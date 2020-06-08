"""abio_dic_dic14 subclass of cime_pop's TracerModuleState"""

import numpy as np

from .. import model_config

from .tracer_module_state import TracerModuleState


class abio_dic_dic14(TracerModuleState):  # pylint: disable=invalid-name
    """abio_dic_dic14 tracer module specifics for TracerModuleState"""

    def stats_vars_metadata(self, fptr_hist):
        """
        return dict of metadata for vars to appear in the stats file for this tracer
        module
        """
        res = super().stats_vars_metadata(fptr_hist)

        # add metadata for FG_ABIO_DIC
        varname_hist = "FG_ABIO_DIC"
        varname_stats = "FG_ABIO_DIC_int_nlat_nlon"
        attr_names = ["_FillValue", "missing_value"]
        attrs = {
            attr_name: getattr(fptr_hist.variables[varname_hist], attr_name)
            for attr_name in attr_names
        }
        attrs["long_name"] = "integrated surface gas flux of abiotic DIC"
        attrs["units"] = "Pg/year"
        res[varname_stats] = {
            "dimensions": ("iteration", "region"),
            "attrs": attrs,
        }

        return res

    def stats_vars_tracer_like(self):
        """
        return list of tracer-like vars in hist file to be processed for the stats file
        """
        res = super().stats_vars_tracer_like()
        res.append("ABIO_D14Cocn")
        return res

    def stats_vars_vals(self, fptr_hist):
        """return tracer module specific stats variables for the current iteration"""
        res = super().stats_vars_vals(fptr_hist)

        # add region dimension to surface version of region_mask
        # assume surf region_cnt at surf is same as full-depth region_cnt
        region_mask_no_region_dim = model_config.model_config_obj.region_mask[0, :, :]
        region_cnt = region_mask_no_region_dim.max()
        region_mask = np.empty((region_cnt,) + region_mask_no_region_dim.shape)
        for region_ind in range(region_cnt):
            region_mask[region_ind, :] = np.where(
                region_mask_no_region_dim == region_ind + 1, 1.0, 0.0
            )

        tarea = fptr_hist.variables["TAREA"][:]

        # add values for FG_ABIO_DIC, dropping singleton time dimension
        varname_hist = "FG_ABIO_DIC"
        varname_stats = "FG_ABIO_DIC_int_nlat_nlon"
        var_vals = fptr_hist.variables[varname_hist][0, :]
        stats_vals = (region_mask * (tarea * var_vals)).sum(axis=(-2, -1))
        # convert to desired units
        stats_vals *= 1.0e-9 * 12.0 * 1.0e-15 * 86400.0 * 365.0
        res[varname_stats] = stats_vals

        return res
