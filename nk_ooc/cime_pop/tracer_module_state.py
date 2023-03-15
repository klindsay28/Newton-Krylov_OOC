"""test_problem model specifics for TracerModuleStateBase"""

import logging

import numpy as np
from netCDF4 import Dataset

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
                        "not all vars have same dimensions, tracer_module_name="
                        f"{self.name}, fname={fname}"
                    )
                    raise ValueError(msg)
            # read values
            if len(dimensions) > 3:
                msg = (
                    "ndim too large (for implementation of dot_prod), "
                    f"tracer_module_name={self.name}, fname={fname}, "
                    f"ndim={len(dimensions)}"
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

    def stats_dimnames(self, fptr):
        """return dimnames to be used in stats file for this tracer module"""
        # base result on first tracer, assume they are the same for all tracers
        tracer_name = self.tracer_names()[0]
        # omit dimension[-1], which is reduced over in stats file
        dimnames = fptr.variables[tracer_name].dimensions[:-1]
        # drop dimension[0] if it is time
        if dimnames[0] == "time":
            dimnames = dimnames[1:]
        return dimnames

    def stats_dimension_varnames(self, fptr):
        """
        return varnames associated with dimensions to be used in stats file
        include associated bounds variables
        """
        varnames = []
        for dimname in self.stats_dimnames(fptr):
            if dimname in fptr.variables:
                varnames.append(dimname)
                if hasattr(fptr.variables[dimname], "bounds"):
                    varnames.append(fptr.variables[dimname].bounds)
        return varnames

    def stats_dimensions(self, fptr):
        """return dimensions to be used in stats file for this tracer module"""
        dimensions = extract_dimensions(fptr, self.stats_dimnames(fptr))
        # include dimensions from associated variables
        dimensions.update(extract_dimensions(fptr, self.stats_dimension_varnames(fptr)))
        return dimensions

    def stats_vars_metadata(self, fptr_hist):
        """
        return dict of metadata for vars to appear in the stats file for this tracer
        module
        """
        res = {}

        # add metadata for coordinate and associated variables
        for varname in self.stats_dimension_varnames(fptr_hist):
            var = fptr_hist.variables[varname]
            attrs = var.__dict__
            attrs["_FillValue"] = None
            res[varname] = {"dimensions": var.dimensions, "attrs": attrs}

        # add metadata for tracer-like variables

        for tracer_name in self.stats_vars_tracer_like():
            tracer = fptr_hist.variables[tracer_name]
            dimensions = tracer.dimensions
            datatype = datatype_sname(tracer)

            attrs = tracer.__dict__
            for attr_name in ["cell_methods", "coordinates", "grid_loc"]:
                if attr_name in attrs:
                    del attrs[attr_name]

            # drop dimensions[0] if it is time
            if dimensions[0] == "time":
                dimensions = dimensions[1:]

            # grid-i average
            varname_stats = "_".join([tracer_name, "mean", dimensions[-1]])
            res[varname_stats] = {
                "datatype": datatype,
                "dimensions": ("iteration", "region") + dimensions[:-1],
                "attrs": attrs,
            }

            # grid-ij average
            varname_stats = "_".join(
                [tracer_name, "mean", dimensions[-2], dimensions[-1]]
            )
            res[varname_stats] = {
                "datatype": datatype,
                "dimensions": ("iteration", "region") + dimensions[:-2],
                "attrs": attrs,
            }
        return res

    def stats_vars_vals_iteration_invariant(self, fptr_hist):
        """return iteration-invariant tracer module specific stats variables"""
        res = {}
        for varname in self.stats_dimension_varnames(fptr_hist):
            res[varname] = fptr_hist.variables[varname][:]
        return res

    def stats_vars_vals(self, fptr_hist):
        """return tracer module specific stats variables for the current iteration"""

        # return values for tracer-like variables

        grid_weight = self.model_config_obj.grid_weight
        region_mask = self.model_config_obj.region_mask

        # allocate space for grid-i average computations
        isum_shape = (self.model_config_obj.region_cnt,) + grid_weight.shape[:-1]
        denom_isum = np.empty(isum_shape)
        numer_isum = np.empty(isum_shape)
        vals_isum = np.empty(isum_shape)

        # allocate space for grid-ij average computations
        ijsum_shape = (self.model_config_obj.region_cnt,) + grid_weight.shape[:-2]
        denom_ijsum = np.empty(ijsum_shape)
        numer_ijsum = np.empty(ijsum_shape)
        vals_ijsum = np.empty(ijsum_shape)

        # compute denominators outside tracer loop,
        # as they are independent of the tracer
        for region_ind in range(self.model_config_obj.region_cnt):
            denom_isum[region_ind, :] = np.where(
                region_mask == region_ind + 1, grid_weight, 0.0
            ).sum(axis=-1)
        denom_ijsum[:] = denom_isum[:].sum(axis=-1)

        res = {}
        for tracer_name in self.stats_vars_tracer_like():
            tracer = fptr_hist.variables[tracer_name]
            dimensions = tracer.dimensions
            fill_value = getattr(tracer, "_FillValue")
            tracer_vals = tracer[:]

            # compute grid-i average, store in result dictionary
            weighted_vals = grid_weight * tracer_vals
            for region_ind in range(self.model_config_obj.region_cnt):
                numer_isum[region_ind, :] = np.where(
                    region_mask == region_ind + 1, weighted_vals, 0.0
                ).sum(axis=-1)
            vals_isum[:] = fill_value
            np.divide(numer_isum, denom_isum, out=vals_isum, where=(denom_isum != 0.0))
            varname_stats = "_".join([tracer_name, "mean", dimensions[-1]])
            res[varname_stats] = vals_isum

            # compute grid-ij average, store in result dictionary
            numer_ijsum[:] = numer_isum[:].sum(axis=-1)
            vals_ijsum[:] = fill_value
            np.divide(
                numer_ijsum, denom_ijsum, out=vals_ijsum, where=(denom_ijsum != 0.0)
            )
            varname_stats = "_".join(
                [tracer_name, "mean", dimensions[-2], dimensions[-1]]
            )
            res[varname_stats] = vals_ijsum

        return res
