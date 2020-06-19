"""Newton's method iterator solver"""

import logging
import os
import shutil

import numpy as np

from .krylov_solver import KrylovSolver
from .model_config import get_modelinfo
from .region_scalars import to_ndarray, to_region_scalar_ndarray
from .solver_state import SolverState, action_step_log_wrap
from .stats_file import StatsFile
from .utils import dict_sel, class_name, mkdir_exist_okay, fmt_vals


class NewtonSolver:
    """
    class for applying Newton's method to approximate the solution of a system of
    equations
    """

    def __init__(self, model_state_class, solverinfo, resume, rewind):
        """initialize Newton solver"""
        logger = logging.getLogger(__name__)
        logger.debug('NewtonSolver, resume="%r", rewind="%r"', resume, rewind)

        # ensure workdir exists
        workdir = solverinfo["workdir"]
        mkdir_exist_okay(workdir)

        self._solverinfo = solverinfo
        self._solver_state = SolverState("Newton", workdir, resume, rewind)
        self._stats_file = StatsFile("Newton", workdir, self._solver_state)

        step = "Newton iterate 0 written"
        if self._solver_state.step_logged(step, per_iteration=False):
            self._iterate = model_state_class(self._fname("iterate"))
        else:
            self._iterate = model_state_class(get_modelinfo("init_iterate_fname"))
            caller = class_name(self) + ".__init__"
            self._iterate.copy_real_tracers_to_shadow_tracers().dump(
                self._fname("iterate"), caller
            )
            self._solver_state.log_step(step, per_iteration=False)

        self._stats_vars_metadata = self.gen_stats_vars_metadata()
        self._def_solver_stats_vars(solver_state=self._solver_state)

        # for iteration == 0, _fcn needs to be computed
        # for iteration >= 1, _fcn is available and stored when iteration is incremented
        if self._solver_state.get_iteration() == 0:
            self._fcn = self._iterate.comp_fcn(
                self._fname("fcn"), self._solver_state, self._fname("hist")
            )
        else:
            self._fcn = model_state_class(self._fname("fcn"))

        self._put_solver_stats_vars(
            model_state={"iterate": self._iterate, "fcn": self._fcn}
        )

        self._iterate.def_stats_vars(
            self._stats_file, self._fname("hist"), solver_state=self._solver_state
        )
        self._iterate.put_stats_vars_iteration_invariant(
            self._stats_file, self._fname("hist"), solver_state=self._solver_state
        )
        self._iterate.put_stats_vars(
            self._stats_file, self._fname("hist"), solver_state=self._solver_state
        )

    def gen_stats_vars_metadata(self):
        """generate metadata for stats vars from Newton solver"""
        vars_metadata = {}
        var_metadata_template = {
            "{state_name}_{method}_{tracer_module_name}": {
                "category": "model_state",
                "state_name": "{state_name}",
                "tracer_module_name": "{tracer_module_name}",
                "method": "{method}",
                "dimensions": ("iteration", "region"),
                "attrs": {
                    "long_name": "{method} of {tracer_module_name} Newton {state_name}",
                },
            },
        }
        state_names = ["iterate", "fcn", "increment"]
        methods = ["mean", "norm"]
        for tracer_module in self._iterate.tracer_modules:
            repl_dict = {"tracer_module_name": tracer_module.name}
            for metadata in var_metadata_template.values():
                metadata["attrs"]["units"] = tracer_module.units
            for state_name in state_names:
                repl_dict.update({"state_name": state_name})
                for method in methods:
                    repl_dict.update({"method": method})
                    var_metadata = fmt_vals(var_metadata_template, repl_dict)
                    vars_metadata.update(var_metadata)
        var_metadata_template = {
            "Armijo_factor_{tracer_module_name}": {
                "category": "scalar",
                "scalar_name": "Armijo_factor",
                "tracer_module_name": "{tracer_module_name}",
                "dimensions": ("iteration", "region"),
                "attrs": {
                    "long_name": (
                        "Armijo factor applied to {tracer_module_name} Newton increment"
                    ),
                    "units": "1",
                },
            },
        }
        for tracer_module in self._iterate.tracer_modules:
            repl_dict = {"tracer_module_name": tracer_module.name}
            var_metadata = fmt_vals(var_metadata_template, repl_dict)
            vars_metadata.update(var_metadata)
        return vars_metadata

    @action_step_log_wrap(
        step="NewtonSolver._def_solver_stats_vars", per_iteration=False
    )
    # pylint: disable=unused-argument
    def _def_solver_stats_vars(self, solver_state):
        """define stats vars from Newton solver"""
        self._stats_file.def_vars(self._stats_vars_metadata)

    def _put_solver_stats_vars(self, **kwargs):
        """write vals corresponding to kwargs for all tracer modules to stats file"""

        # dict of varname and values to be written
        # collect together before opening file and writing
        varname_vals_all = {}

        for category, vals_dict in kwargs.items():
            vars_metadata = dict_sel(self._stats_vars_metadata, category=category)
            if category == "model_state":
                for state_name, model_state in vals_dict.items():
                    vars_metadata_sub = dict_sel(vars_metadata, state_name=state_name)
                    for method in ["mean", "norm"]:
                        step = "write %s %s vals to stats file" % (state_name, method)
                        if self._solver_state.step_logged(step):
                            continue
                        if method == "mean":
                            vals_reduced = model_state.mean()
                        else:
                            vals_reduced = model_state.norm()
                        varname_vals_scalar = self._gen_varname_vals_scalar(
                            vars_metadata_sub, vals_reduced, method=method
                        )
                        varname_vals_all.update(varname_vals_scalar)
                        self._solver_state.log_step(step)
            elif category == "scalar":
                for scalar_name, scalar_val in vals_dict.items():
                    varname_vals_scalar = self._gen_varname_vals_scalar(
                        vars_metadata, scalar_val, scalar_name=scalar_name
                    )
                    varname_vals_all.update(varname_vals_scalar)
            else:
                msg = "unknown category %s" % category
                raise ValueError(msg)

        self._stats_file.put_vars(self._solver_state.get_iteration(), varname_vals_all)

    def _gen_varname_vals_scalar(self, vars_metadata, scalar_var, **kwargs):
        """return varname_vals dict of scalar values for all tracer modules"""
        varname_vals = {}
        for ind, tracer_module in enumerate(self._iterate.tracer_modules):
            vars_metadata_sub = dict_sel(
                vars_metadata, tracer_module_name=tracer_module.name, **kwargs
            )
            for varname in vars_metadata_sub:
                varname_vals[varname] = scalar_var[ind].vals()
        return varname_vals

    def _fname(self, quantity, iteration=None):
        """construct fname corresponding to particular quantity"""
        if iteration is None:
            iteration = self._solver_state.get_iteration()
        return os.path.join(
            self._solverinfo["workdir"], "%s_%02d.nc" % (quantity, iteration)
        )

    def log(self, iterate=None, fcn=None, msg=None):
        """write the state of the instance to the log"""
        iteration = self._solver_state.get_iteration()
        if msg is None:
            iteration_p_msg = "iteration=%02d" % iteration
        else:
            iteration_p_msg = "iteration=%02d,%s" % (iteration, msg)

        log_obj = self._iterate if iterate is None else iterate
        log_obj.log("%s,iterate" % iteration_p_msg)

        log_obj = self._fcn if fcn is None else fcn
        log_obj.log("%s,fcn" % iteration_p_msg)

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
            self.log()
        krylov_solver = KrylovSolver(
            self._iterate, krylov_dir, resume, rewind, self._fname("hist"),
        )
        self._solver_state.log_step(step)
        increment = krylov_solver.solve(
            self._fname("increment"), self._iterate, self._fcn
        )
        self._put_solver_stats_vars(model_state={"increment": increment})
        self._solver_state.log_step(fcn_complete_step)
        iteration = self._solver_state.get_iteration()
        increment.log("Newton increment %02d" % iteration)
        return increment

    @action_step_log_wrap(step="NewtonSolver._armijo_init")
    def _armijo_init(self, solver_state):
        """initialize Armijo factor computation"""
        solver_state.set_value_saved_state(key="armijo_ind", value=0)
        solver_state.set_value_saved_state(
            key="armijo_factor_flat", value=np.where(self.converged_flat(), 0.0, 1.0)
        )

    def _comp_next_iterate(self, increment):
        """compute next Newton iterate"""
        logger = logging.getLogger(__name__)
        logger.debug("entering")

        self._armijo_init(solver_state=self._solver_state)
        armijo_ind = self._solver_state.get_value_saved_state(key="armijo_ind")
        armijo_factor_flat = self._solver_state.get_value_saved_state(
            key="armijo_factor_flat"
        )

        fcn_complete_step = "_comp_next_iterate complete"

        if self._solver_state.step_logged(fcn_complete_step):
            logger.debug('"%s" logged, returning result', fcn_complete_step)
            return (
                type(self._iterate)(self._fname("prov_Armijo_%02d" % armijo_ind)),
                type(self._iterate)(self._fname("prov_fcn_Armijo_%02d" % armijo_ind)),
            )
        logger.debug('"%s" not logged, proceeding', fcn_complete_step)

        caller = class_name(self) + "._comp_next_iterate"

        while True:
            # compute provisional candidate for next iterate
            armijo_factor = to_region_scalar_ndarray(armijo_factor_flat)
            prov = self._iterate + armijo_factor * increment
            prov.dump(self._fname("prov_Armijo_%02d" % armijo_ind), caller)
            prov_fcn = prov.comp_fcn(
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

                self._put_solver_stats_vars(scalar={"Armijo_factor": armijo_factor})

                return prov, prov_fcn

            logger.info("Armijo condition not satisfied")
            armijo_factor_flat = np.where(
                armijo_cond_flat, armijo_factor_flat, 0.5 * armijo_factor_flat
            )
            armijo_ind += 1
            self._solver_state.set_value_saved_state(key="armijo_ind", value=armijo_ind)
            self._solver_state.set_value_saved_state(
                key="armijo_factor_flat", value=armijo_factor_flat
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
            self.log()
            msg = "number of maximum Newton iterations exceeded"
            raise RuntimeError(msg)

        caller = class_name(self) + ".step"

        step = "fp iterations started"
        if not self._solver_state.step_logged(step):

            increment = self._comp_increment()

            prov, prov_fcn = self._comp_next_iterate(increment)

            fp_iter = 0
            self._solver_state.set_value_saved_state(key="fp_iter", value=fp_iter)
            prov.copy_shadow_tracers_to_real_tracers()
            prov.dump(self._fname("prov_fp_%02d" % fp_iter), caller)
            # Evaluate comp_fcn after copying shadow tracers to their real counterparts.
            # If no shadow tracers are on, then this is the same as the final comp_fcn
            # result from Armijo iterations.
            # Do not preserve hist files from Armijo iterations. Either remove the last
            # one, or rename it to the initial fp hist file (if there are not shadow
            # tracers on).
            armijo_ind = self._solver_state.get_value_saved_state(key="armijo_ind")
            if prov.shadow_tracers_on():
                prov_fcn = prov.comp_fcn(
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
            fp_iter = self._solver_state.get_value_saved_state(key="fp_iter")
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
            prov_fcn = prov.comp_fcn(
                self._fname("prov_fcn_fp_%02d" % (fp_iter + 1)),
                self._solver_state,
                self._fname("prov_hist_fp_%02d" % (fp_iter + 1)),
            )
            fp_iter += 1
            self._solver_state.set_value_saved_state(key="fp_iter", value=fp_iter)
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

        self._put_solver_stats_vars(
            model_state={"iterate": self._iterate, "fcn": self._fcn}
        )
        self._iterate.put_stats_vars(
            self._stats_file,
            hist_fname=self._fname("hist"),
            solver_state=self._solver_state,
        )
