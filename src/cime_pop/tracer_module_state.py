"""test_problem model specifics for TracerModuleStateBase"""

import logging

import numpy as np
from netCDF4 import Dataset

from .. import model_config
from ..tracer_module_state_base import TracerModuleStateBase
from ..utils import (
    create_dimensions_verify,
    create_vars,
    datatype_sname,
    extract_dimensions,
)


class TracerModuleState(TracerModuleStateBase):
    """
    Derived class for representing a collection of model tracers.
    It implements _read_vals and dump.
    """

    def _read_vals(self, fname):
        """return tracer values and dimension names and lengths, read from fname)"""
        logger = logging.getLogger(__name__)
        logger.debug('tracer_module_name="%s", fname="%s"', self.name, fname)
        suffix = "_CUR"
        with Dataset(fname, mode="r") as fptr:
            fptr.set_auto_mask(False)
            # get dimensions from first variable
            varname = self.tracer_names()[0] + suffix
            dimensions = extract_dimensions(fptr, varname)
            # all tracers are stored in a single array
            # tracer index is the leading index
            vals = np.empty((self.tracer_cnt,) + tuple(dimensions.values()))
            # check that all vars have the same dimensions
            for tracer_name in self.tracer_names():
                if extract_dimensions(fptr, tracer_name + suffix) != dimensions:
                    msg = (
                        "not all vars have same dimensions"
                        ", tracer_module_name=%s, fname=%s" % (self.name, fname)
                    )
                    raise ValueError(msg)
            # read values
            if len(dimensions) > 3:
                msg = (
                    "ndim too large (for implementation of dot_prod)"
                    "tracer_module_name=%s, fname=%s, ndim=%s"
                    % (self.name, fname, len(dimensions))
                )
                raise ValueError(msg)
            for tracer_ind, tracer_name in enumerate(self.tracer_names()):
                var = fptr.variables[tracer_name + suffix]
                vals[tracer_ind, :] = var[:]
        return vals, dimensions

    def dump(self, fptr, action):
        """
        perform an action (define or write) of dumping a TracerModuleState object
        to an open file
        """
        if action == "define":
            create_dimensions_verify(fptr, self._dimensions)
            dimnames = tuple(self._dimensions.keys())
            # define all tracers, with _CUR and _OLD suffixes
            vars_metadata = {}
            for tracer_name in self.tracer_names():
                for suffix in ["_CUR", "_OLD"]:
                    vars_metadata[tracer_name + suffix] = {"dimensions": dimnames}
            create_vars(fptr, vars_metadata)
        elif action == "write":
            # write all tracers, with _CUR and _OLD suffixes
            for tracer_ind, tracer_name in enumerate(self.tracer_names()):
                for suffix in ["_CUR", "_OLD"]:
                    fptr.variables[tracer_name + suffix][:] = self._vals[tracer_ind, :]
        else:
            msg = "unknown action=", action
            raise ValueError(msg)
        return self

    @staticmethod
    def stats_dimensions(fptr):
        """return dimensions to be used in stats file for this tracer module"""
        dimnames = ["z_t", "nlat"]
        return {dimname: len(fptr.dimensions[dimname]) for dimname in dimnames}

    def stats_vars_metadata(self, fptr_hist):
        """
        return dict of metadata for vars to appear in the stats file for this tracer
        module
        """
        res = {}

        for dimname in ["z_t"]:
            attrs = fptr_hist.variables[dimname].__dict__
            attrs["_FillValue"] = None
            res[dimname] = {"dimensions": (dimname,), "attrs": attrs}

        # add metadata for tracer-like variables

        for tracer_name in self.stats_vars_tracer_like():
            datatype = datatype_sname(fptr_hist.variables[tracer_name])

            attrs = fptr_hist.variables[tracer_name].__dict__
            del attrs["cell_methods"]
            del attrs["coordinates"]
            del attrs["grid_loc"]

            # grid-i average
            varname_stats = "_".join([tracer_name, "mean", "nlon"])
            res[varname_stats] = {
                "datatype": datatype,
                "dimensions": ("iteration", "region", "z_t", "nlat"),
                "attrs": attrs,
            }

            # grid-ij average
            varname_stats = "_".join([tracer_name, "mean", "nlat", "nlon"])
            res[varname_stats] = {
                "datatype": datatype,
                "dimensions": ("iteration", "region", "z_t"),
                "attrs": attrs,
            }
        return res

    @staticmethod
    def stats_vars_vals_iteration_invariant(fptr_hist):
        """return iteration-invariant tracer module specific stats variables"""
        res = {}
        for varname in ["z_t"]:
            res[varname] = fptr_hist.variables[varname][:]
        return res

    def stats_vars_vals(self, fptr_hist):
        """return tracer module specific stats variables for the current iteration"""

        # return values for tracer-like variables

        grid_weight = model_config.model_config_obj.grid_weight

        denom_nlon = grid_weight.sum(axis=-1)
        numer_nlon = np.empty(denom_nlon.shape)

        denom_nlat_nlon = grid_weight.sum(axis=(-2, -1))
        numer_nlat_nlon = np.empty(denom_nlat_nlon.shape)

        res = {}
        for tracer_name in self.stats_vars_tracer_like():
            tracer = fptr_hist.variables[tracer_name]
            fill_value = tracer._FillValue  # pylint: disable=protected-access
            tracer_vals = tracer[:]

            # grid-i average
            varname_stats = "_".join([tracer_name, "mean", "nlon"])
            numer_nlon[:] = (grid_weight * tracer_vals).sum(axis=-1)
            vals_nlon = np.full(numer_nlon.shape, fill_value)
            np.divide(numer_nlon, denom_nlon, out=vals_nlon, where=(denom_nlon != 0.0))
            res[varname_stats] = vals_nlon

            # grid-ij average
            varname_stats = "_".join([tracer_name, "mean", "nlat", "nlon"])
            numer_nlat_nlon[:] = (grid_weight * tracer_vals).sum(axis=(-2, -1))
            vals_nlat_nlon = np.full(numer_nlat_nlon.shape, fill_value)
            np.divide(
                numer_nlat_nlon,
                denom_nlat_nlon,
                out=vals_nlat_nlon,
                where=(denom_nlat_nlon != 0.0),
            )
            res[varname_stats] = vals_nlat_nlon
        return res
