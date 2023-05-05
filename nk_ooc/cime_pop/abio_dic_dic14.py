"""abio_dic_dic14 subclass of cime_pop's TracerModuleState"""

import numpy as np

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
        datatype = fptr_hist.variables["FG_ABIO_DIC"].dtype
        attrs = fptr_hist.variables["FG_ABIO_DIC"].__dict__
        del attrs["cell_methods"]
        del attrs["coordinates"]
        del attrs["grid_loc"]
        attrs["long_name"] = "integrated surface gas flux of abiotic DIC"
        attrs["units"] = "Pg/year"

        res["FG_ABIO_DIC_int_nlat_nlon"] = {
            "datatype": datatype,
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

        # base result on first tracer, assume they are the same for all tracers
        tracer_name = list(self._tracer_module_def["tracers"])[0]
        region_mask_surf = self.get_grid_vars(tracer_name)["region_mask"][0, :, :]

        # confirm that surf region_cnt is same as full-depth region_cnt
        if region_mask_surf.max() != self.model_config_obj.region_cnt:
            raise RuntimeError("region_cnt_surf != region_cnt")

        tarea = fptr_hist.variables["TAREA"][:]

        # add values for FG_ABIO_DIC, dropping singleton time dimension
        hist_var_vals = tarea * fptr_hist.variables["FG_ABIO_DIC"][0, :]
        stats_var_vals = np.empty(self.model_config_obj.region_cnt)
        for region_ind in range(self.model_config_obj.region_cnt):
            stats_var_vals[region_ind] = np.where(
                region_mask_surf == region_ind + 1, hist_var_vals, 0.0
            ).sum()
        # convert to desired units
        stats_var_vals *= 1.0e-9 * 12.0 * 1.0e-15 * 86400.0 * 365.0
        res["FG_ABIO_DIC_int_nlat_nlon"] = stats_var_vals

        return res
