"""generic iterative solver infrastructure"""

import logging
import os

from .solver_state import SolverState
from .stats_file import StatsFile
from .utils import fmt_vals, mkdir_exist_okay


class SolverBase:
    """
    generic iterative solver class
    base class for NewtonSolver and KrylovSolver
    """

    def __init__(self, solver_name, solverinfo, region_cnt, resume, rewind):
        """initialize solver"""
        logger = logging.getLogger(__name__)

        logger.debug(
            'solver_name=""%s", resume="%r", rewind="%r"', solver_name, resume, rewind
        )

        self._solver_name = solver_name
        self._solverinfo = solverinfo

        workdir = self._get_workdir()
        logger.debug('%s solver workdir="%s"', solver_name, workdir)
        mkdir_exist_okay(workdir)

        self._solver_state = SolverState(self._solver_name, workdir, resume, rewind)

        self._stats_file = StatsFile(
            self._solver_name, workdir, region_cnt, self._solver_state
        )
        self._stats_vars_put_metadata = {}

    def get_iteration(self):
        """get current iteration"""
        return self._solver_state.get_iteration()

    def _get_workdir(self):
        """get name of workdir from solverinfo"""
        key = f"{self._solver_name}_workdir"
        if key not in self._solverinfo:
            key = "workdir"
        return self._solverinfo[key]

    def _fname(self, quantity, iteration=None):
        """construct fname corresponding to particular quantity"""
        if iteration is None:
            iteration = self.get_iteration()
        return os.path.join(self._get_workdir(), f"{quantity}_{iteration:02}.nc")

    def _get_rel_tol(self):
        """get solver's relative tolerance from solverinfo"""
        key = f"{self._solver_name}_rel_tol"
        return float(self._solverinfo[key])

    def _get_min_iter(self):
        """
        get minimum number iterations from solverinfo, if present
        return 0 otherwise
        """
        key = f"{self._solver_name}_min_iter"
        if key not in self._solverinfo:
            return 0
        return int(self._solverinfo[key])

    def _def_solver_stats_vars(self, stats_vars_dict, tracer_modules):
        """
        define stats file vars for solver
        store info easing writing stats vars in self._stats_vars_put_metadata
        """

        # self._stats_vars_put_metadata needs to be generated on reinvocations
        vars_def_metadata = {}
        for key, metadata in stats_vars_dict.items():
            # verify iteration dimension is first, if present
            dimensions = metadata["dimensions"]
            if "iteration" in dimensions and dimensions[0] != "iteration":
                raise ValueError("iteration must be first dimension, if present")
            category = metadata["category"]
            self._stats_vars_put_metadata[key] = {
                "category": category,
                "dimensions": dimensions,
            }
            if category == "model_state":
                stats_varnames = {"mean": [], "norm": []}
                for method, varnames in stats_varnames.items():
                    repl_dict = {"method": method}
                    for tracer_module in tracer_modules:
                        repl_dict["tracer_module_name"] = tracer_module.name
                        repl_dict["tracer_module_units"] = tracer_module.units
                        stats_varname = f"{key}_{method}_{tracer_module.name}"
                        vars_def_metadata[stats_varname] = fmt_vals(metadata, repl_dict)
                        attrs = vars_def_metadata[stats_varname]["attrs"]
                        if attrs["units"] == "None":
                            attrs["units"] = None
                        varnames.append(stats_varname)
                self._stats_vars_put_metadata[key]["stats_varnames"] = stats_varnames
            elif category == "per_tracer_module":
                stats_varnames = []
                for tracer_module in tracer_modules:
                    repl_dict = {"tracer_module_name": tracer_module.name}
                    repl_dict["tracer_module_units"] = tracer_module.units
                    stats_varname = f"{key}_{tracer_module.name}"
                    vars_def_metadata[stats_varname] = fmt_vals(metadata, repl_dict)
                    attrs = vars_def_metadata[stats_varname]["attrs"]
                    if attrs["units"] == "None":
                        attrs["units"] = None
                    stats_varnames.append(stats_varname)
                self._stats_vars_put_metadata[key]["stats_varnames"] = stats_varnames
            elif category == "tracer_module_independent":
                vars_def_metadata[key] = metadata
            else:
                raise ValueError(f"unknown category {category}")

        # use step-log to avoid attempting to redefine stats file vars
        step = f"define {self._solver_name} solver stats file vars"
        if not self._solver_state.step_logged(step, per_iteration=False):
            self._stats_file.def_vars(vars_def_metadata)
        self._solver_state.log_step(step, per_iteration=False)

    def _put_solver_stats_vars_iteration_independent(self, **kwargs):
        """write vals corresponding to kwargs to stats file"""

        # dict of varname and values to be written
        # collect together before opening file and writing
        vals_dict = {}

        for key, vals in kwargs.items():
            var_put_metadata = self._stats_vars_put_metadata[key]
            # verify iteration dimension not present
            if "iteration" in var_put_metadata["dimensions"]:
                raise ValueError(
                    "_put_solver_stats_vars should be used "
                    "for vars with the iteration dimension"
                )
            # use step-log to avoid rewriting vals to the stats file
            step = f"write {key} vals to stats file"
            if self._solver_state.step_logged(step, per_iteration=False):
                continue
            category = var_put_metadata["category"]
            if category == "per_tracer_module":
                for ind, stats_varname in enumerate(var_put_metadata["stats_varnames"]):
                    vals_dict[stats_varname] = vals[ind]
            elif category == "tracer_module_independent":
                vals_dict[key] = vals
            else:
                raise ValueError(f"unknown category {category}")
            self._solver_state.log_step(step, per_iteration=False)

        self._stats_file.put_vars_iteration_invariant(vals_dict)

    def _put_solver_stats_vars(self, **kwargs):
        """write vals corresponding to kwargs to stats file"""

        # dict of varname and values to be written
        # collect together before opening file and writing
        vals_dict = {}

        for key, vals in kwargs.items():
            var_put_metadata = self._stats_vars_put_metadata[key]
            # verify iteration dimension present
            if "iteration" not in var_put_metadata["dimensions"]:
                raise ValueError(
                    "_put_solver_stats_vars_iteration_independent should be used "
                    "for vars lacking the iteration dimension"
                )
            # use step-log to avoid rewriting vals to the stats file
            step = f"write {key} vals to stats file"
            if self._solver_state.step_logged(step):
                continue
            category = var_put_metadata["category"]
            if category == "model_state":
                for method in ["mean", "norm"]:
                    vals_reduced = vals.mean() if method == "mean" else vals.norm()
                    for ind, stats_varname in enumerate(
                        var_put_metadata["stats_varnames"][method]
                    ):
                        vals_dict[stats_varname] = vals_reduced[ind]
            elif category == "per_tracer_module":
                for ind, stats_varname in enumerate(var_put_metadata["stats_varnames"]):
                    vals_dict[stats_varname] = vals[ind]
            elif category == "tracer_module_independent":
                vals_dict[key] = vals
            else:
                raise ValueError(f"unknown category {category}")
            self._solver_state.log_step(step)

        self._stats_file.put_vars(self.get_iteration(), vals_dict)
