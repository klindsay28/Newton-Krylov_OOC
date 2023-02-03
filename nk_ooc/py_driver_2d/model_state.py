"""py_driver_2d model specifics for ModelStateBase"""

import copy
import logging
import subprocess
from datetime import datetime
from inspect import signature

import numpy as np
from netCDF4 import Dataset
from scipy import integrate

from ..model_state_base import ModelStateBase
from ..spatial_axis import spatial_axis_from_file
from ..utils import class_name, create_dimensions_verify, create_vars, strtobool
from .advection import Advection
from .horiz_mix import HorizMix
from .vert_mix import VertMix


class ModelState(ModelStateBase):
    """py_driver_2d model specifics for ModelStateBase"""

    # give ModelState operators higher priority than those of numpy
    __array_priority__ = 100

    class_vars_set = False

    def __init__(self, fname):
        # confirm that model_config_obj has been set for this class
        if ModelState.model_config_obj is None:
            raise RuntimeError("ModelState.model_config_obj is None")

        # Call _set_class_vars before super().__init__ to ensure
        # that the axis class variables are available in super().__init__.
        # These are used when generating intial values from tracer module metadata,
        # or (potentially) regridding from input datasets to the axes.
        self._set_class_vars(self.model_config_obj.modelinfo)

        super().__init__(fname)

    @staticmethod
    def _set_class_vars(modelinfo):
        """set (time-invariant) class variables"""

        if ModelState.class_vars_set:
            return

        ModelState.time_range = (0.0, 365.0 * 86400.0)
        ModelState.depth = spatial_axis_from_file(
            fname=modelinfo["grid_weight_fname"], axisname="depth"
        )
        ModelState.ypos = spatial_axis_from_file(
            fname=modelinfo["grid_weight_fname"], axisname="ypos"
        )
        ModelState.processes = {}
        ModelState.processes["advection"] = Advection(
            ModelState.depth, ModelState.ypos, modelinfo
        )
        ModelState.processes["horiz_mix"] = HorizMix(
            ModelState.depth, ModelState.ypos, modelinfo
        )
        ModelState.processes["vert_mix"] = VertMix(ModelState.depth, ModelState.ypos)

        ModelState.class_vars_set = True

    def get_tracer_vals_all(self):
        """get all tracer values"""
        res_vals = np.empty((self.tracer_cnt, len(self.depth), len(self.ypos)))
        ind0 = 0
        for tracer_module in self.tracer_modules:
            cnt = tracer_module.tracer_cnt
            res_vals[ind0 : ind0 + cnt, :] = tracer_module.get_tracer_vals_all()
            ind0 = ind0 + cnt
        return res_vals

    def set_tracer_vals_all(self, vals, reseat_vals=False):
        """set all tracer values"""
        ind0 = 0
        if reseat_vals:
            tracer_modules_orig = self.tracer_modules
            self.tracer_modules = np.empty(len(tracer_modules_orig), dtype=object)
            for ind, tracer_module_orig in enumerate(tracer_modules_orig):
                self.tracer_modules[ind] = copy.copy(tracer_module_orig)
                cnt = tracer_module_orig.tracer_cnt
                self.tracer_modules[ind].set_tracer_vals_all(
                    vals[ind0 : ind0 + cnt, :], reseat_vals=reseat_vals
                )
                ind0 = ind0 + cnt
        else:
            for tracer_module in self.tracer_modules:
                cnt = tracer_module.tracer_cnt
                tracer_module.set_tracer_vals_all(
                    vals[ind0 : ind0 + cnt, :], reseat_vals=reseat_vals
                )
                ind0 = ind0 + cnt

    def comp_fcn(self, res_fname, solver_state, hist_fname=None):
        """evalute function being solved with Newton's method"""
        logger = logging.getLogger(__name__)
        logger.debug('res_fname="%s", hist_fname="%s"', res_fname, hist_fname)

        if solver_state is not None:
            fcn_complete_step = "comp_fcn complete for %s" % res_fname
            if solver_state.step_logged(fcn_complete_step):
                logger.debug('"%s" logged, returning result', fcn_complete_step)
                return ModelState(res_fname)
            logger.debug('"%s" not logged, proceeding', fcn_complete_step)

        # get dense output, if requested
        if hist_fname is not None:
            t_eval = np.linspace(self.time_range[0], self.time_range[1], 61)
        else:
            t_eval = np.array(self.time_range)

        # memory for result, use it initially for passing initial value to solve_ivp
        res_vals = self.get_tracer_vals_all()

        fptr_hist = self._hist_def_dimensions(hist_fname)
        self._hist_def_vars_tracer_module_independent(fptr_hist)

        shape = (len(self.depth), len(self.ypos))

        # solve ODEs for each tracer module independently, using scipy.integrate
        ind0 = 0
        for tracer_module in self.tracer_modules:
            self._hist_def_vars(tracer_module, fptr_hist)
            cnt = tracer_module.tracer_cnt
            tracer_vals_init = res_vals[ind0 : ind0 + cnt, :].reshape(-1)
            # assume sparsity pattern does not change in time
            jac_sparsity = tracer_module.comp_jacobian_sparsity(
                self.time_range[0], tracer_vals_init, self.processes
            )
            sol = integrate.solve_ivp(
                tracer_module.comp_tend,
                self.time_range,
                tracer_vals_init,
                "Radau",
                t_eval,
                max_step=(self.time_range[1] - self.time_range[0]) * 0.01,
                atol=1.0e-6,
                rtol=1.0e-6,
                args=(self.processes,),
                jac=tracer_module.comp_jacobian,
                jac_sparsity=jac_sparsity,
            )
            if ind0 == 0:
                self._hist_write_tracer_module_independent(sol, fptr_hist)
            self._hist_write(tracer_module, sol, fptr_hist)
            res_vals[ind0 : ind0 + cnt, :] = (
                sol.y[:, -1].reshape((cnt,) + shape) - res_vals[ind0 : ind0 + cnt, :]
            )
            ind0 = ind0 + cnt

        if fptr_hist is not None:
            fptr_hist.close()

        # ModelState instance for result
        res_ms = copy.copy(self)
        res_ms.set_tracer_vals_all(res_vals, reseat_vals=True)

        caller = class_name(self) + ".comp_fcn"
        res_ms.comp_fcn_postprocess(res_fname, caller)

        if solver_state is not None:
            solver_state.log_step(fcn_complete_step)
            modelinfo = self.model_config_obj.modelinfo
            if strtobool(modelinfo["reinvoke"]):
                cmd = [modelinfo["invoker_script_fname"], "--resume"]
                logger.info('cmd="%s"', " ".join(cmd))
                # use Popen instead of run because we don't want to wait
                subprocess.Popen(cmd)
                raise SystemExit

        return res_ms

    def _hist_def_dimensions(self, hist_fname):
        """define hist dimensions"""
        if hist_fname is None:
            return None

        # create the hist file
        fptr_hist = Dataset(hist_fname, mode="w", format="NETCDF3_64BIT_OFFSET")
        datestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        name = __name__ + "._gen_hist"
        fptr_hist.history = datestamp + ": created by " + name

        # define dimensions
        dimensions = {"time": None}
        for axis in [self.depth, self.ypos]:
            dimensions.update(axis.dump_dimensions())
        create_dimensions_verify(fptr_hist, dimensions)

        return fptr_hist

    def _hist_def_vars_tracer_module_independent(self, fptr_hist):
        """define hist vars that are independent of tracer modules"""
        if fptr_hist is None:
            return

        # define dict of variable metadata

        hist_vars_metadata = {}
        hist_vars_metadata["time"] = {
            "dimensions": ("time",),
            "attrs": {
                "long_name": "time",
                "units": "seconds since 0001-01-01",
                "calendar": "noleap",
            },
        }

        for axis in [self.depth, self.ypos]:
            hist_vars_metadata.update(axis.dump_vars_metadata())

        for process in self.processes.values():
            hist_vars_metadata.update(process.get_hist_vars_metadata())

        # set cell_methods attribute and define hist vars
        for varname, metadata in hist_vars_metadata.items():
            if varname != "time" and "time" in metadata["dimensions"]:
                metadata["attrs"]["cell_methods"] = "time: point"

        create_vars(fptr_hist, hist_vars_metadata)

        fptr_hist.sync()

    @staticmethod
    def _hist_def_vars(tracer_module, fptr_hist):
        """define hist vars for tracer_module"""
        if fptr_hist is None:
            return

        hist_vars_metadata = tracer_module.hist_vars_metadata()

        # set cell_methods attribute and define hist vars
        for metadata in hist_vars_metadata.values():
            if "time" in metadata["dimensions"]:
                metadata["attrs"]["cell_methods"] = "time: point"

        create_vars(fptr_hist, hist_vars_metadata)

        fptr_hist.sync()

    def _hist_write_tracer_module_independent(self, sol, fptr_hist):
        """define hist vars that are independent of tracer modules"""
        if fptr_hist is None:
            return

        fptr_hist.variables["time"][:] = sol.t

        for axis in [self.depth, self.ypos]:
            axis.dump_write(fptr_hist)

        for process in self.processes.values():
            process.hist_write(sol, fptr_hist)

    def _hist_write(self, tracer_module, sol, fptr_hist):
        """write hist vars for tracer_module"""
        if fptr_hist is None:
            return

        # write tracer module hist vars, providing appropriate segment of sol.y
        tracer_vals_all = sol.y.reshape(
            (tracer_module.tracer_cnt, len(self.depth), len(self.ypos), -1)
        )
        tracer_module.write_hist_vars(fptr_hist, tracer_vals_all)

        fptr_hist.sync()

    def apply_precond_jacobian(self, precond_fname, res_fname, solver_state):
        """apply preconditioner of jacobian of comp_fcn to model state object, self"""
        logger = logging.getLogger(__name__)
        logger.debug('precond_fname="%s", res_fname="%s"', precond_fname, res_fname)

        if solver_state is not None:
            fcn_complete_step = "apply_precond_jacobian complete for %s" % res_fname
            if solver_state.step_logged(fcn_complete_step):
                logger.debug('"%s" logged, returning result', fcn_complete_step)
                return ModelState(res_fname)
            logger.debug('"%s" not logged, proceeding', fcn_complete_step)

        # ModelState instance for result
        res_ms = copy.deepcopy(self)

        with Dataset(precond_fname, mode="r") as fptr:
            for tracer_module_ind, tracer_module in enumerate(self.tracer_modules):
                # pass fptr_precond, if it is present as an argument
                kwargs = {}
                if (
                    "fptr_precond"
                    in signature(tracer_module.apply_precond_jacobian).parameters
                ):
                    kwargs["fptr_precond"] = fptr
                tracer_module.apply_precond_jacobian(
                    self.time_range,
                    res_ms.tracer_modules[tracer_module_ind],
                    ModelState.processes,
                    **kwargs
                )

        if solver_state is not None:
            solver_state.log_step(fcn_complete_step)

        caller = class_name(self) + ".apply_precond_jacobian"
        return res_ms.dump(res_fname, caller)
