"""Newton's method iterative solver"""

import logging
import os

import numpy as np

from .krylov_solver import KrylovSolver
from .solver_base import SolverBase
from .solver_state import action_step_log_wrap
from .utils import class_name, fmt_vals


class NewtonSolver(SolverBase):
    """
    class for applying Newton's method to approximate the solution of a system of
    equations
    """

    def __init__(self, model_state_class, solverinfo, resume, rewind):
        """initialize Newton solver"""

        super().__init__(
            "Newton",
            solverinfo,
            model_state_class.model_config_obj.region_cnt,
            resume,
            rewind,
        )

        step = "Newton iterate 0 written"
        if self._solver_state.step_logged(step, per_iteration=False):
            self._iterate = model_state_class(self._fname("iterate"))
        else:
            self._iterate = model_state_class(solverinfo["init_iterate_fname"])
            caller = f"{class_name(self)}.__init__"
            self._iterate.copy_real_tracers_to_shadow_tracers().dump(
                self._fname("iterate"), caller
            )
            self._solver_state.log_step(step, per_iteration=False)

        self._def_solver_stats_vars(
            self.gen_stats_vars_metadata(), self._iterate.tracer_modules
        )

        self._fcn = self._iterate.comp_fcn(
            self._fname("fcn"), self._solver_state, self._fname("hist")
        )

        self._put_solver_stats_vars(iterate=self._iterate, fcn=self._fcn)

        self._iterate.def_stats_vars(
            self._stats_file, self._fname("hist"), solver_state=self._solver_state
        )
        self._iterate.put_stats_vars_iteration_invariant(
            self._stats_file, self._fname("hist"), solver_state=self._solver_state
        )
        self._iterate.put_stats_vars(
            self._stats_file, self._fname("hist"), solver_state=self._solver_state
        )

    @staticmethod
    def gen_stats_vars_metadata():
        """generate metadata for stats vars from Newton solver"""
        vars_metadata = {}

        var_metadata_template = {
            "category": "model_state",
            "dimensions": ("iteration", "region"),
            "attrs": {
                "long_name": "{method} of {tracer_module_name} Newton {state_name}",
                "units": "{tracer_module_units}",
            },
        }
        for state_name in ["iterate", "fcn", "increment"]:
            repl_dict = {
                "state_name": state_name,
                "method": "{method}",
                "tracer_module_name": "{tracer_module_name}",
                "tracer_module_units": "{tracer_module_units}",
            }
            vars_metadata[state_name] = fmt_vals(var_metadata_template, repl_dict)

        vars_metadata["increment_scalef"] = {
            "category": "per_tracer_module",
            "dimensions": ("iteration",),
            "attrs": {
                "long_name": (
                    "factor applied to {tracer_module_name} Newton increment to "
                    "satisfy bounds"
                ),
                "units": "1",
            },
        }

        vars_metadata["Armijo_factor"] = {
            "category": "per_tracer_module",
            "dimensions": ("iteration", "region"),
            "attrs": {
                "long_name": (
                    "factor applied to {tracer_module_name} Newton increment to "
                    "satisfy Armijo condition"
                ),
                "units": "1",
            },
        }

        vars_metadata["Krylov_iterations"] = {
            "category": "tracer_module_independent",
            "datatype": "i4",
            "dimensions": ("iteration",),
            "attrs": {
                "long_name": "number of iterations in Krylov solver",
                "units": "1",
            },
        }

        return vars_metadata

    def log(self, iterate=None, fcn=None, msg=None):
        """write the state of the instance to the log"""
        if msg is None:
            iteration_p_msg = f"iteration={self.get_iteration():02}"
        else:
            iteration_p_msg = f"iteration={self.get_iteration():02},{msg}"

        log_obj = self._iterate if iterate is None else iterate
        log_obj.log(f"{iteration_p_msg},iterate")

        log_obj = self._fcn if fcn is None else fcn
        log_obj.log(f"{iteration_p_msg},fcn")

    def converged(self):
        """is residual small"""
        rel_tol = self._get_rel_tol()
        return (self.get_iteration() >= self._get_min_iter()) & (
            self._fcn.norm() < rel_tol * self._iterate.norm()
        )

    def _comp_increment(self):
        """
        compute Newton's method increment
        (d(fcn) / d(iterate)) (increment) = -fcn
        """
        logger = logging.getLogger(__name__)
        logger.debug("entering")

        fcn_complete_step = "_comp_increment complete"

        if self._solver_state.step_logged(fcn_complete_step):
            logger.debug('"%s" logged, returning result', fcn_complete_step)
            return type(self._iterate)(self._fname("increment"))
        logger.debug('"%s" not logged, computing increment', fcn_complete_step)

        self._solverinfo["krylov_workdir"] = os.path.join(
            self._get_workdir(), f"krylov_{self.get_iteration():02}"
        )
        step = "KrylovSolver instantiated"
        rewind = self._solver_state.step_was_rewound(step)
        resume = rewind or self._solver_state.step_logged(step)
        if not resume:
            self.log()
        krylov_solver = KrylovSolver(
            self._iterate, self._solverinfo, resume, rewind, self._fname("hist")
        )
        self._solver_state.log_step(step)
        increment = krylov_solver.solve(self._fname("increment"), self._fcn)
        self._put_solver_stats_vars(
            Krylov_iterations=krylov_solver.get_iteration(), increment=increment
        )
        self._solver_state.log_step(fcn_complete_step)
        increment.log(f"Newton increment {self.get_iteration():02}")
        return increment

    @action_step_log_wrap(step="NewtonSolver._armijo_init")
    def _armijo_init(self, solver_state):
        """initialize Armijo factor computation"""
        solver_state.set_value_saved_state(key="armijo_ind", value=0)
        solver_state.set_value_saved_state(
            key="armijo_factor", value=np.where(self.converged(), 0.0, 1.0)
        )

    def _comp_next_iterate(self, increment):
        """compute next Newton iterate"""
        logger = logging.getLogger(__name__)
        logger.debug("entering")

        self._armijo_init(solver_state=self._solver_state)
        armijo_ind = self._solver_state.get_value_saved_state(key="armijo_ind")
        armijo_factor = self._solver_state.get_value_saved_state(key="armijo_factor")

        fcn_complete_step = "_comp_next_iterate complete"

        if self._solver_state.step_logged(fcn_complete_step):
            logger.debug('"%s" logged, returning result', fcn_complete_step)
            return (
                type(self._iterate)(self._fname(f"prov_Armijo_{armijo_ind:02}")),
                type(self._iterate)(self._fname(f"prov_fcn_Armijo_{armijo_ind:02}")),
            )
        logger.debug('"%s" not logged, proceeding', fcn_complete_step)

        caller = f"{class_name(self)}._comp_next_iterate"

        while True:
            # compute provisional candidate for next iterate
            prov = self._iterate + armijo_factor * increment
            prov.dump(self._fname(f"prov_Armijo_{armijo_ind:02}"), caller)
            prov_fcn = prov.comp_fcn(
                self._fname(f"prov_fcn_Armijo_{armijo_ind:02}"),
                self._solver_state,
                self._fname(f"prov_hist_Armijo_{armijo_ind:02}"),
            )

            # at this point in the execution flow, only keep latest Armijo hist file
            if armijo_ind > 0:
                os.remove(self._fname(f"prov_hist_Armijo_{(armijo_ind + 1):02}"))

            logger.info("Armijo_ind=%d", armijo_ind)

            # Determine if Armijo condition is satisfied. Based on Eq. (A.1) of
            # Kelley, C. T., Solving nonlinear equations with Newton's method, 2003.
            fcn_norm = self._fcn.norm()
            prov_fcn_norm = prov_fcn.norm()
            increment.log_vals(
                ["ArmijoFactor", "fcn_norm", "prov_fcn_norm"],
                np.stack((armijo_factor, fcn_norm, prov_fcn_norm)),
            )
            alpha = 1.0e-4
            armijo_cond = (armijo_factor == 0.0) | (
                prov_fcn_norm <= (1.0 - alpha * armijo_factor) * fcn_norm
            )

            if armijo_cond.all():
                logger.info("Armijo condition satisfied")
                self._solver_state.log_step(fcn_complete_step)

                self._put_solver_stats_vars(Armijo_factor=armijo_factor)

                return prov, prov_fcn

            logger.info("Armijo condition not satisfied")
            armijo_factor = np.where(armijo_cond, armijo_factor, 0.5 * armijo_factor)
            armijo_ind += 1
            self._solver_state.set_value_saved_state(key="armijo_ind", value=armijo_ind)
            self._solver_state.set_value_saved_state(
                key="armijo_factor", value=armijo_factor
            )

            if armijo_ind > 10:
                raise RuntimeError("Armijo_ind exceeds limit")

    def step(self):
        """perform a step of Newton's method"""
        logger = logging.getLogger(__name__)
        logger.debug("entering")

        if self.get_iteration() >= int(self._solverinfo["newton_max_iter"]):
            self.log()
            raise RuntimeError("number of maximum Newton iterations exceeded")

        caller = f"{class_name(self)}.step"

        step = "fp iterations started"
        if not self._solver_state.step_logged(step):
            increment = self._comp_increment()

            scalef = increment.apply_limiter(self._iterate)
            self._put_solver_stats_vars(increment_scalef=scalef)

            prov, prov_fcn = self._comp_next_iterate(increment)

            fp_iter = 0
            self._solver_state.set_value_saved_state(key="fp_iter", value=fp_iter)
            prov.copy_shadow_tracers_to_real_tracers()
            prov.dump(self._fname(f"prov_fp_{fp_iter:02}"), caller)
            # Evaluate comp_fcn after copying shadow tracers to their real counterparts.
            # If no shadow tracers are on, then this is the same as the final comp_fcn
            # result from Armijo iterations.
            # Do not preserve hist files from Armijo iterations. Either remove the last
            # one, or rename it to the initial fp hist file (if there are not shadow
            # tracers on).
            armijo_ind = self._solver_state.get_value_saved_state(key="armijo_ind")
            if prov.shadow_tracers_on():
                prov_fcn = prov.comp_fcn(
                    self._fname(f"prov_fcn_fp_{fp_iter:02}"),
                    self._solver_state,
                    self._fname(f"prov_hist_fp_{fp_iter:02}"),
                )
                os.remove(self._fname(f"prov_hist_Armijo_{armijo_ind:02}"))
            else:
                prov_fcn.dump(self._fname(f"prov_fcn_fp_{fp_iter:02}"), caller)
                os.rename(
                    self._fname(f"prov_hist_Armijo_{armijo_ind:02}"),
                    self._fname(f"prov_hist_fp_{fp_iter:02}"),
                )
            self._solver_state.log_step(step)
        else:
            fp_iter = self._solver_state.get_value_saved_state(key="fp_iter")
            prov = type(self._iterate)(self._fname(f"prov_fp_{fp_iter:02}"))
            prov_fcn = type(self._iterate)(self._fname(f"prov_fcn_fp_{fp_iter:02}"))

        while fp_iter < int(self._solverinfo["post_newton_fp_iter"]):
            step = f"prov updated for fp iteration {fp_iter:02}"
            if not self._solver_state.step_logged(step):
                if fp_iter == 0:
                    self.log(prov, prov_fcn, "pre-fp_iter")
                prov += prov_fcn
                prov.copy_shadow_tracers_to_real_tracers()
                prov.dump(self._fname(f"prov_fp_{(fp_iter + 1):02}"), caller)
                self._solver_state.log_step(step)
            else:
                prov = type(self._iterate)(self._fname(f"prov_fp_{(fp_iter + 1):02}"))
            if fp_iter + 1 < int(self._solverinfo["post_newton_fp_iter"]):
                res_fname = self._fname(f"prov_fcn_fp_{(fp_iter + 1):02}")
                hist_fname = self._fname(f"prov_hist_fp_{(fp_iter + 1):02}")
            else:
                self._solver_state.inc_iteration()
                prov.dump(self._fname("iterate"), caller)
                res_fname = self._fname("fcn")
                hist_fname = self._fname("hist")
            prov_fcn = prov.comp_fcn(res_fname, self._solver_state, hist_fname)
            fp_iter += 1
            self._solver_state.set_value_saved_state(key="fp_iter", value=fp_iter)
            self.log(prov, prov_fcn, f"fp_iter={fp_iter:02}")

        self._iterate = prov
        self._fcn = prov_fcn

        self._put_solver_stats_vars(iterate=self._iterate, fcn=self._fcn)
        self._iterate.put_stats_vars(
            self._stats_file,
            hist_fname=self._fname("hist"),
            solver_state=self._solver_state,
        )
