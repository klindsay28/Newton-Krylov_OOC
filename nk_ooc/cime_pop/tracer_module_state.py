"""test_problem model specifics for TracerModuleStateBase"""

import logging

import numpy as np

from ..cime import cime_yr_cnt
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
    It implements _load_dataset and dump.
    """

    def _load_dataset(self, fname):
        """return xarray Dataset of tracer module tracers"""
        logger = logging.getLogger(__name__)
        logger.debug('tracer_module_name="%s", fname="%s"', self.name, fname)
        self._tracer_varname_suffix = "CUR"
        return super()._load_dataset(fname)

    def dump(self, fptr, action):
        """
        perform an action (define or write) of dumping a TracerModuleState object
        to an open file
        """
        if action == "define":
            create_dimensions_verify(fptr, dict(self._dataset.dims))
            # define all tracers, with CUR and OLD suffixes
            vars_metadata = {}
            for tracer_name in self._tracer_module_def["tracers"]:
                dimnames = self._dataset[tracer_name].dims
                for suffix in ["CUR", "OLD"]:
                    varname = f"{tracer_name}_{suffix}"
                    vars_metadata[varname] = {"dimensions": dimnames}
            create_vars(fptr, vars_metadata)
        elif action == "write":
            # write all tracers, with CUR and OLD suffixes
            for tracer_name in self._tracer_module_def["tracers"]:
                tracer_vals = self.get_tracer_vals(tracer_name)
                for suffix in ["CUR", "OLD"]:
                    varname = f"{tracer_name}_{suffix}"
                    fptr.variables[varname][:] = tracer_vals
        else:
            raise ValueError(f"unknown action={action}")
        return self

    def stats_dimnames(self, fptr):
        """return dimnames to be used in stats file for this tracer module"""
        # base result on first tracer, assume they are the same for all tracers
        tracer_name = list(self._tracer_module_def["tracers"])[0]
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
            varname_stats = f"{tracer_name}_mean_{dimensions[-1]}"
            res[varname_stats] = {
                "datatype": datatype,
                "dimensions": ("iteration", "region") + dimensions[:-1],
                "attrs": attrs,
            }

            # grid-ij average
            varname_stats = f"{tracer_name}_mean_{dimensions[-2]}_{dimensions[-1]}"
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

        # base result on first tracer, assume they are the same for all tracers
        tracer_name = list(self._tracer_module_def["tracers"])[0]
        grid_vars = self.get_grid_vars(tracer_name)
        grid_weight = grid_vars["grid_weight"]
        region_mask = grid_vars["region_mask"]

        # compute denominators outside tracer loop,
        # as they are independent of the tracer
        isum_shape = (self.model_config_obj.region_cnt,) + grid_weight.shape[:-1]
        denom_isum = np.empty(isum_shape)
        for region_ind in range(self.model_config_obj.region_cnt):
            denom_isum[region_ind, :] = np.where(
                region_mask == region_ind + 1, grid_weight, 0.0
            ).sum(axis=-1)
        denom_ijsum = denom_isum.sum(axis=-1)

        # allocate space for numerators, which is shared across all tracers
        numer_isum = np.empty(denom_isum.shape)
        numer_ijsum = np.empty(denom_ijsum.shape)

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
            quo_i = np.full(denom_isum.shape, fill_value)
            np.divide(numer_isum, denom_isum, out=quo_i, where=denom_isum != 0.0)
            varname_stats = f"{tracer_name}_mean_{dimensions[-1]}"
            res[varname_stats] = quo_i

            # compute grid-ij average, store in result dictionary
            numer_ijsum[:] = numer_isum[:].sum(axis=-1)
            quo_ij = np.full(denom_ijsum.shape, fill_value)
            np.divide(numer_ijsum, denom_ijsum, out=quo_ij, where=denom_ijsum != 0.0)
            varname_stats = f"{tracer_name}_mean_{dimensions[-2]}_{dimensions[-1]}"
            res[varname_stats] = quo_ij

        return res

    def apply_tracers_sflux_term(self, tracer_names_subset, precond_fptr):
        """
        Apply surface flux term of tracers in tracer_names_subset to subsequent
        tracer_names, if there is a dependency.
        Return value is boolen, reporting if a term was applied.
        """
        logger = logging.getLogger(__name__)
        logger.debug(
            'tracer_names_subset="%s", precond_fname="%s"',
            tracer_names_subset,
            precond_fptr.filepath(),
        )
        term_applied = False
        delta_time = 365.0 * 86400.0 * cime_yr_cnt(self.model_config_obj.modelinfo)
        tracer_names_all = list(self._tracer_module_def["tracers"])
        for tracer_name_src in tracer_names_subset:
            try:
                tracer_name_src_ind = tracer_names_all.index(tracer_name_src)
            except ValueError:
                continue
            for tracer_name_dst in tracer_names_all[tracer_name_src_ind + 1 :]:
                partial_deriv_varname = f"d_SF_{tracer_name_dst}_d_{tracer_name_src}"
                if partial_deriv_varname in precond_fptr.variables:
                    logger.info('applying "%s"', partial_deriv_varname)
                    partial_deriv = precond_fptr.variables[partial_deriv_varname]
                    # replace _FillValue vals with 0.0
                    partial_deriv_vals = partial_deriv[:].filled(0.0)
                    src = self.get_tracer_vals(tracer_name_src)
                    dst = self.get_tracer_vals(tracer_name_dst)
                    dst[0, :] -= (
                        delta_time
                        / precond_fptr.variables["dz"][0].data
                        * partial_deriv_vals
                        * src[0, :]
                    )
                    self.set_tracer_vals(tracer_name_dst, dst)
                    term_applied = True
        return term_applied
