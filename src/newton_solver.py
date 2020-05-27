"""Newton's method iterator solver"""

import logging
import os
import shutil

import numpy as np

from .krylov_solver import KrylovSolver
from .model_config import get_modelinfo
from .region_scalars import to_ndarray, to_region_scalar_ndarray
from .solver import SolverState
from .stats_file import StatsFile
from .utils import mkdir_exist_okay


class NewtonSolver:
    """
    class for applying Newton's method to approximate the solution of system of equations
    """

    def __init__(self, newton_fcn_obj, solverinfo, resume, rewind):
        """initialize Newton solver"""
        logger = logging.getLogger(__name__)
        logger.debug('NewtonSolver, resume="%r", rewind="%r"', resume, rewind)

        # ensure workdir exists
        workdir = solverinfo["workdir"]
        mkdir_exist_okay(workdir)

        self._newton_fcn_obj = newton_fcn_obj
        self._solverinfo = solverinfo
        self._solver_state = SolverState("Newton", workdir, resume, rewind)
        self._stats_file = StatsFile("Newton", workdir, resume)

        # get solver started the first time NewtonSolver is instantiated
        if not resume:
            iterate = self._newton_fcn_obj.model_state_obj(
                get_modelinfo("init_iterate_fname")
            )
            caller = __name__ + ".NewtonSolver.__init__"
            iterate.copy_real_tracers_to_shadow_tracers().dump(
                self._fname("iterate"), caller
            )
            stats_metadata = {
                "iterate_mean_{tr_mod_name}": {
                    "long_name": "mean of {tr_mod_name} iterate"
                },
                "iterate_norm_{tr_mod_name}": {
                    "long_name": "norm of {tr_mod_name} iterate"
                },
                "fcn_mean_{tr_mod_name}": {
                    "long_name": "mean of fcn applied to {tr_mod_name} iterate"
                },
                "fcn_norm_{tr_mod_name}": {
                    "long_name": "norm of fcn applied to {tr_mod_name} iterate"
                },
                "increment_mean_{tr_mod_name}": {
                    "long_name": "mean of {tr_mod_name} Newton increment"
                },
                "increment_norm_{tr_mod_name}": {
                    "long_name": "norm of {tr_mod_name} Newton increment"
                },
                "Armijo_Factor_{tr_mod_name}": {
                    "long_name": "Armijo factor applied to {tr_mod_name} Newton increment"
                },
            }
            self._stats_file.def_vars_generic(stats_metadata)

        self._iterate = self._newton_fcn_obj.model_state_obj(self._fname("iterate"))

        # for iteration == 0, _fcn needs to be computed
        # for iteration >= 1, _fcn is available and stored when iteration is incremented
        if self._solver_state.get_iteration() == 0:
            self._fcn = self._newton_fcn_obj.comp_fcn(
                self._iterate,
                self._fname("fcn"),
                self._solver_state,
                self._fname("hist"),
            )
        else:
            self._fcn = self._newton_fcn_obj.model_state_obj(self._fname("fcn"))

        step = "def_stats_vars called"
        if not self._solver_state.step_logged(step, per_iteration=False):
            self._iterate.def_stats_vars(self._stats_file, self._fname("hist"))
        self._solver_state.log_step(step, per_iteration=False)

    def _fname(self, quantity, iteration=None):
        """construct fname corresponding to particular quantity"""
        if iteration is None:
            iteration = self._solver_state.get_iteration()
        return os.path.join(
            self._solverinfo["workdir"], "%s_%02d.nc" % (quantity, iteration)
        )

    def log(self, iterate=None, fcn=None, msg=None, append_to_stats_file=False):
        """write the state of the instance to the log"""
        iteration = self._solver_state.get_iteration()
        if msg is None:
            iteration_p_msg = "iteration=%02d" % iteration
        else:
            iteration_p_msg = "iteration=%02d,%s" % (iteration, msg)

        stats_info = {
            "append_vals": append_to_stats_file,
            "stats_file_obj": self._stats_file,
            "iteration": iteration,
        }

        log_obj = self._iterate if iterate is None else iterate
        stats_info["varname_root"] = "iterate"
        log_obj.log("%s,iterate" % iteration_p_msg, stats_info=stats_info)

        log_obj = self._fcn if fcn is None else fcn
        stats_info["varname_root"] = "fcn"
        log_obj.log("%s,fcn" % iteration_p_msg, stats_info=stats_info)

        if append_to_stats_file:
            step = "put_stats_vars called"
            if not self._solver_state.step_logged(step):
                self._iterate.put_stats_vars(
                    self._stats_file,
                    self._solver_state.get_iteration(),
                    self._fname("hist"),
                )
            self._solver_state.log_step(step)

    def converged_flat(self):
        """is residual small"""
        rel_tol = self._solverinfo.getfloat("newton_rel_tol")
        return to_ndarray(self._fcn.norm()) < rel_tol * to_ndarray(self._iterate.norm())

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

        krylov_dir = os.path.join(
            self._solverinfo["workdir"],
            "krylov_%02d" % self._solver_state.get_iteration(),
        )
        step = "KrylovSolver instantiated"
        rewind = self._solver_state.step_was_rewound(step)
        resume = rewind or self._solver_state.step_logged(step)
        if not resume:
            self.log(append_to_stats_file=True)
        krylov_solver = KrylovSolver(
            self._newton_fcn_obj,
            self._iterate,
            krylov_dir,
            resume,
            rewind,
            self._fname("hist"),
        )
        self._solver_state.log_step(step)
        increment = krylov_solver.solve(
            self._fname("increment"), self._iterate, self._fcn
        )
        self._solver_state.log_step(fcn_complete_step)
        iteration = self._solver_state.get_iteration()
        stats_info = {
            "append_vals": True,
            "stats_file_obj": self._stats_file,
            "iteration": iteration,
            "varname_root": "increment",
        }
        increment.log("Newton increment %02d" % iteration, stats_info=stats_info)
        return increment

    def _comp_next_iterate(self, increment):
        """compute next Newton iterate"""
        logger = logging.getLogger(__name__)
        logger.debug("entering")

        step = "Armijo factor computation started"
        if not self._solver_state.step_logged(step):
            armijo_ind = 0
            self._solver_state.set_value_saved_state("armijo_ind", armijo_ind)
            armijo_factor_flat = np.where(self.converged_flat(), 0.0, 1.0)
            self._solver_state.set_value_saved_state(
                "armijo_factor_flat", armijo_factor_flat
            )
            self._solver_state.log_step(step)
        else:
            armijo_ind = self._solver_state.get_value_saved_state("armijo_ind")
            armijo_factor_flat = self._solver_state.get_value_saved_state(
                "armijo_factor_flat"
            )

        fcn_complete_step = "_comp_next_iterate complete"

        if self._solver_state.step_logged(fcn_complete_step):
            logger.debug('"%s" logged, returning result', fcn_complete_step)
            return (
                type(self._iterate)(self._fname("prov_Armijo_%02d" % armijo_ind)),
                type(self._iterate)(self._fname("prov_fcn_Armijo_%02d" % armijo_ind)),
            )
        logger.debug('"%s" not logged, proceeding', fcn_complete_step)

        caller = __name__ + ".NewtonSolver._comp_next_iterate"

        while True:
            # compute provisional candidate for next iterate
            armijo_factor = to_region_scalar_ndarray(armijo_factor_flat)
            prov = self._iterate + armijo_factor * increment
            prov.dump(self._fname("prov_Armijo_%02d" % armijo_ind), caller)
            prov_fcn = self._newton_fcn_obj.comp_fcn(
                prov,
                self._fname("prov_fcn_Armijo_%02d" % armijo_ind),
                self._solver_state,
                self._fname("prov_hist_Armijo_%02d" % armijo_ind),
            )

            # at this point in the execution flow, only keep latest Armijo hist file
            if armijo_ind > 0:
                os.remove(self._fname("prov_hist_Armijo_%02d" % (armijo_ind - 1)))

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
            armijo_cond_flat = (armijo_factor_flat == 0.0) | (
                to_ndarray(prov_fcn_norm)
                <= (1.0 - alpha * armijo_factor_flat) * to_ndarray(fcn_norm)
            )

            if armijo_cond_flat.all():
                logger.info("Armijo condition satisfied")
                self._solver_state.log_step(fcn_complete_step)

                # write ArmijoFactor to the stats file
                self._stats_file.put_vars_generic(
                    self._solver_state.get_iteration(),
                    "Armijo_Factor_{tr_mod_name}",
                    armijo_factor,
                )

                return prov, prov_fcn

            logger.info("Armijo condition not satisfied")
            armijo_factor_flat = np.where(
                armijo_cond_flat, armijo_factor_flat, 0.5 * armijo_factor_flat
            )
            armijo_ind += 1
            self._solver_state.set_value_saved_state("armijo_ind", armijo_ind)
            self._solver_state.set_value_saved_state(
                "armijo_factor_flat", armijo_factor_flat
            )

            if armijo_ind > 10:
                msg = "Armijo_ind exceeds limit"
                raise RuntimeError(msg)

    def step(self):
        """perform a step of Newton's method"""
        logger = logging.getLogger(__name__)
        logger.debug("entering")

        if self._solver_state.get_iteration() >= self._solverinfo.getint(
            "newton_max_iter"
        ):
            self.log(append_to_stats_file=True)
            msg = "number of maximum Newton iterations exceeded"
            raise RuntimeError(msg)

        caller = __name__ + "NewtonSolver.step"

        step = "fp iterations started"
        if not self._solver_state.step_logged(step):

            increment = self._comp_increment()

            prov, prov_fcn = self._comp_next_iterate(increment)

            fp_iter = 0
            self._solver_state.set_value_saved_state("fp_iter", fp_iter)
            prov.copy_shadow_tracers_to_real_tracers()
            prov.dump(self._fname("prov_fp_%02d" % fp_iter), caller)
            # Evaluate comp_fcn after copying shadow tracers to their real counterparts.
            # If no shadow tracers are on, then this is the same as the final comp_fcn
            # result from Armijo iterations.
            # Do not preserve hist files from Armijo iterations. Either remove the last
            # one, or rename it to the initial fp hist file (if there are not shadow
            # tracers on).
            armijo_ind = self._solver_state.get_value_saved_state("armijo_ind")
            if prov.shadow_tracers_on():
                prov_fcn = self._newton_fcn_obj.comp_fcn(
                    prov,
                    self._fname("prov_fcn_fp_%02d" % fp_iter),
                    self._solver_state,
                    self._fname("prov_hist_fp_%02d" % fp_iter),
                )
                os.remove(self._fname("prov_hist_Armijo_%02d" % armijo_ind))
            else:
                prov_fcn.dump(self._fname("prov_fcn_fp_%02d" % fp_iter), caller)
                os.rename(
                    self._fname("prov_hist_Armijo_%02d" % armijo_ind),
                    self._fname("prov_hist_fp_%02d" % fp_iter),
                )
            self._solver_state.log_step(step)
        else:
            fp_iter = self._solver_state.get_value_saved_state("fp_iter")
            prov = type(self._iterate)(self._fname("prov_fp_%02d" % fp_iter))
            prov_fcn = type(self._iterate)(self._fname("prov_fcn_fp_%02d" % fp_iter))

        while fp_iter < self._solverinfo.getint("post_newton_fp_iter"):
            step = "prov updated for fp iteration %02d" % fp_iter
            if not self._solver_state.step_logged(step):
                if fp_iter == 0:
                    self.log(prov, prov_fcn, "pre-fp_iter")
                prov += prov_fcn
                prov.copy_shadow_tracers_to_real_tracers()
                prov.dump(self._fname("prov_fp_%02d" % (fp_iter + 1)), caller)
                self._solver_state.log_step(step)
            else:
                prov = type(self._iterate)(self._fname("prov_fp_%02d" % (fp_iter + 1)))
            prov_fcn = self._newton_fcn_obj.comp_fcn(
                prov,
                self._fname("prov_fcn_fp_%02d" % (fp_iter + 1)),
                self._solver_state,
                self._fname("prov_hist_fp_%02d" % (fp_iter + 1)),
            )
            fp_iter += 1
            self._solver_state.set_value_saved_state("fp_iter", fp_iter)
            self.log(prov, prov_fcn, "fp_iter=%02d" % fp_iter)

        shutil.copyfile(
            self._fname("prov_hist_fp_%02d" % fp_iter),
            self._fname("hist", self._solver_state.get_iteration() + 1),
        )

        self._solver_state.inc_iteration()

        self._iterate = prov
        self._iterate.dump(self._fname("iterate"), caller)
        self._fcn = prov_fcn
        self._fcn.dump(self._fname("fcn"), caller)
