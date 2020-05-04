"""class to hold model configuration info"""

import logging

import numpy as np
from netCDF4 import Dataset
import yaml

# model configuration info
model_config_obj = None

# functions to commonly accessed vars in model_config_obj
def get_region_cnt():
    """return number of regions specified by region_mask"""
    return model_config_obj.region_cnt


def get_precond_matrix_def(matrix_name):
    """return an entry from precond_matrix_defs"""
    return model_config_obj.precond_matrix_defs[matrix_name]


def get_modelinfo(key):
    """return value associated in modelinfo with key"""
    return model_config_obj.modelinfo[key]


################################################################################


class ModelConfig:
    """class to hold model configuration info"""

    def __init__(self, modelinfo, lvl=logging.DEBUG):
        logger = logging.getLogger(__name__)
        logger.debug("ModelConfig")

        # store modelinfo for later use
        self.modelinfo = modelinfo

        # load content from tracer_module_defs_fname
        fname = modelinfo["tracer_module_defs_fname"]
        logger.log(lvl, "loading content from %s", fname)
        with open(fname, mode="r") as fptr:
            file_contents = yaml.safe_load(fptr)
        self.tracer_module_defs = file_contents["tracer_module_defs"]
        check_shadow_tracers(self.tracer_module_defs, lvl)
        self.precond_matrix_defs = pad_defs(
            file_contents["precond_matrix_defs"],
            "precond matrix",
            {"hist_to_precond_var_names": "list"},
            lvl,
        )
        check_precond_matrix_defs(self.precond_matrix_defs)

        # extract grid_weight from modelinfo config object
        fname = modelinfo["grid_weight_fname"]
        varname = modelinfo["grid_weight_varname"]
        logger.log(lvl, "reading %s from %s for grid_weight", varname, fname)
        with Dataset(fname, mode="r") as fptr:
            fptr.set_auto_mask(False)
            grid_weight_no_region_dim = fptr.variables[varname][:]

        # extract region_mask from modelinfo config object
        fname = modelinfo["region_mask_fname"]
        varname = modelinfo["region_mask_varname"]
        if not fname == "None" and not varname == "None":
            logger.log(lvl, "reading %s from %s for region_mask", varname, fname)
            with Dataset(fname, mode="r") as fptr:
                fptr.set_auto_mask(False)
                self.region_mask = fptr.variables[varname][:]
                if self.region_mask.shape != grid_weight_no_region_dim.shape:
                    msg = "region_mask and grid_weight must have the same shape"
                    raise RuntimeError(msg)
        else:
            self.region_mask = np.ones_like(grid_weight_no_region_dim, dtype=np.int32)

        # enforce that region_mask and grid_weight and both 0 where one of them is
        self.region_mask = np.where(
            grid_weight_no_region_dim == 0.0, 0, self.region_mask
        )
        grid_weight_no_region_dim = np.where(
            self.region_mask == 0, 0.0, grid_weight_no_region_dim
        )

        self.region_cnt = self.region_mask.max()

        # add region dimension to grid_weight and normalize
        self.grid_weight = np.empty(
            (self.region_cnt,) + grid_weight_no_region_dim.shape
        )
        for region_ind in range(self.region_cnt):
            self.grid_weight[region_ind, :] = np.where(
                self.region_mask == region_ind + 1, grid_weight_no_region_dim, 0.0
            )
            # normalize grid_weight so that its sum is 1.0 over each region
            self.grid_weight[region_ind, :] *= 1.0 / np.sum(
                self.grid_weight[region_ind, :]
            )

        # store contents in module level var, to enable use elsewhere
        global model_config_obj  # pylint: disable=W0603
        model_config_obj = self


def check_shadow_tracers(tracer_module_defs, lvl):
    """Confirm that tracers specified in shadow_tracers are also in tracer_names."""
    # This check is done for all entries in tracer_module_defs,
    # whether they are being used or not.
    logger = logging.getLogger(__name__)
    for tracer_module_name, tracer_module_def in tracer_module_defs.items():
        # Verify that shadow_tracer_name and real_tracer_name are known tracer names.
        for tracer_name, tracer_metadata in tracer_module_def.items():
            if "shadows" in tracer_metadata:
                if tracer_metadata["shadows"] not in tracer_module_def:
                    msg = "shadows value %s for %s in tracer module %s not known" % (
                        tracer_metadata["shadows"],
                        tracer_name,
                        tracer_module_name,
                    )
                    raise ValueError(msg)
                logger.log(
                    lvl,
                    "tracer module %s has %s as a shadow for %s",
                    tracer_module_name,
                    tracer_name,
                    tracer_metadata["shadows"],
                )


def pad_defs(def_dict, obj_desc, def_entries, lvl):
    """
    Place emtpy objects in dict of definitions where they do not exist.
    This is to ease subsequent coding.
    """
    logger = logging.getLogger(__name__)
    for def_dict_key, def_dict_value in def_dict.items():
        for entry, entry_type in def_entries.items():
            if entry not in def_dict_value:
                logger.log(lvl, "%s %s has no entry %s", obj_desc, def_dict_key, entry)
                def_dict_value[entry] = [] if entry_type == "list" else {}
    return def_dict


def check_precond_matrix_defs(precond_matrix_defs):
    """Perform basic vetting of precond_matrix_defs"""
    # This check is done for all entries in def_dict,
    # whether they are being used or not.
    for precond_matrix_name, precond_matrix_def in precond_matrix_defs.items():
        # verify that suffixes in hist_to_precond_var_names are recognized
        for hist_var in precond_matrix_def["hist_to_precond_var_names"]:
            _, _, time_op = hist_var.partition(":")
            if time_op not in ["avg", "log_avg", "copy", ""]:
                msg = "unknown time_op=%s in %s from %s" % (
                    time_op,
                    hist_var,
                    precond_matrix_name,
                )
                raise ValueError(msg)
