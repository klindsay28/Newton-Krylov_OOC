"""test_problem model specifics for TracerModuleStateBase"""

import logging

from netCDF4 import Dataset
import numpy as np

from test_problem.src.spatial_axis import SpatialAxis

from ..tracer_module_state_base import TracerModuleStateBase
from ..model_config import get_modelinfo


################################################################################


class TracerModuleState(TracerModuleStateBase):
    """
    Derived class for representing a collection of model tracers.
    It implements _read_vals and dump.
    """

    def _read_vals(self, tracer_module_name, fname):
        """return tracer values and dimension names and lengths, read from fname)"""
        logger = logging.getLogger(__name__)
        logger.debug(
            'tracer_module_name="%s", fname="%s"', tracer_module_name, fname,
        )
        if fname == "gen_init_iterate":
            depth = SpatialAxis("depth", get_modelinfo("depth_fname"))
            vals = np.empty((len(self._tracer_module_def), depth.nlevs))
            for tracer_ind, tracer_metadata in enumerate(
                self._tracer_module_def.values()
            ):
                if "init_iterate_vals" in tracer_metadata:
                    vals[tracer_ind, :] = np.interp(
                        depth.mid,
                        tracer_metadata["init_iterate_val_depths"],
                        tracer_metadata["init_iterate_vals"],
                    )
                elif "shadows" in tracer_metadata:
                    shadowed_tracer = tracer_metadata["shadows"]
                    shadow_tracer_metadata = self._tracer_module_def[shadowed_tracer]
                    vals[tracer_ind, :] = np.interp(
                        depth.mid,
                        shadow_tracer_metadata["init_iterate_val_depths"],
                        shadow_tracer_metadata["init_iterate_vals"],
                    )
                else:
                    msg = (
                        "gen_init_iterate failure for %s"
                        % self.tracer_names()[tracer_ind]
                    )
                    raise ValueError(msg)
            return vals, {"depth": depth.nlevs}
        dims = {}
        with Dataset(fname, mode="r") as fptr:
            fptr.set_auto_mask(False)
            # get dims from first variable
            dimnames0 = fptr.variables[self.tracer_names()[0]].dimensions
            for dimname in dimnames0:
                dims[dimname] = fptr.dimensions[dimname].size
            # all tracers are stored in a single array
            # tracer index is the leading index
            vals = np.empty((self.tracer_cnt(),) + tuple(dims.values()))
            # check that all vars have the same dimensions
            for tracer_name in self.tracer_names():
                if fptr.variables[tracer_name].dimensions != dimnames0:
                    msg = (
                        "not all vars have same dimensions"
                        ", tracer_module_name=%s, fname=%s"
                        % (tracer_module_name, fname)
                    )
                    raise ValueError(msg)
            # read values
            if len(dims) > 3:
                msg = (
                    "ndim too large (for implementation of dot_prod)"
                    "tracer_module_name=%s, fname=%s, ndim=%s"
                    % (tracer_module_name, fname, len(dims))
                )
                raise ValueError(msg)
            for tracer_ind, tracer_name in enumerate(self.tracer_names()):
                varid = fptr.variables[tracer_name]
                vals[tracer_ind, :] = varid[:]
        return vals, dims

    def dump(self, fptr, action):
        """
        perform an action (define or write) of dumping a TracerModuleState object
        to an open file
        """
        if action == "define":
            for dimname, dimlen in self._dims.items():
                try:
                    if fptr.dimensions[dimname].size != dimlen:
                        msg = (
                            "dimname already exists and has wrong size"
                            "tracer_module_name=%s, dimname=%s"
                            % (self._tracer_module_name, dimname)
                        )
                        raise ValueError(msg)
                except KeyError:
                    fptr.createDimension(dimname, dimlen)
            dimnames = tuple(self._dims.keys())
            # define all tracers
            for tracer_name in self.tracer_names():
                fptr.createVariable(tracer_name, "f8", dimensions=dimnames)
        elif action == "write":
            # write all tracers
            for tracer_ind, tracer_name in enumerate(self.tracer_names()):
                fptr.variables[tracer_name][:] = self._vals[tracer_ind, :]
        else:
            msg = "unknown action=%s", action
            raise ValueError(msg)
        return self
