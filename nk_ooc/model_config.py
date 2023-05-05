"""class to hold model configuration info"""

import copy
import logging

import numpy as np
import scipy
import scipy.sparse
import yaml
from netCDF4 import Dataset
from packaging.version import Version

from .share import repro_fname
from .utils import fmt_vals


class ModelConfig:
    """class to hold model configuration info"""

    def __init__(self, modelinfo, lvl=logging.DEBUG):
        logger = logging.getLogger(__name__)
        logger.debug("ModelConfig")

        # store modelinfo for later use
        self.modelinfo = modelinfo

        # load content from tracer_module_defs_fname
        fname = modelinfo["tracer_module_defs_fname"]
        logger.log(lvl, "loading content from %s", repro_fname(modelinfo, fname))
        with open(fname, mode="r") as fptr:
            file_contents = yaml.safe_load(fptr)
        self.tracer_module_defs = file_contents["tracer_module_defs"]
        check_shadow_tracers(self.tracer_module_defs, lvl)
        check_tracer_module_suffs(self.tracer_module_defs)
        check_tracer_module_names(
            modelinfo["tracer_module_names"], self.tracer_module_defs
        )
        self.precond_matrix_defs = file_contents["precond_matrix_defs"]
        propagate_base_matrix_defs_to_all(self.precond_matrix_defs)
        check_precond_matrix_defs(self.precond_matrix_defs)

        modelinfo["tracer_module_names"] = self.tracer_module_expand_all(
            modelinfo["tracer_module_names"]
        )

        # confirm that region_mask_varname is specified for all tracers
        # add region_mask_varname to variable metadata, to ease subsequent lookup
        # create set of unique region_mask_varnames
        region_mask_varnames = set()
        for tracer_module_name in modelinfo["tracer_module_names"].split(","):
            tracer_module_def = self.tracer_module_defs[tracer_module_name]
            for tracer_name, tracer_metadata in tracer_module_def["tracers"].items():
                if "region_mask_varname" not in tracer_metadata:
                    if "region_mask_varname" in tracer_module_def:
                        region_mask_varname = tracer_module_def["region_mask_varname"]
                    else:
                        raise RuntimeError(
                            f"region_mask_varname not known for {tracer_name} in"
                            f"{tracer_module_name}"
                        )
                    tracer_metadata["region_mask_varname"] = region_mask_varname
                region_mask_varnames.add(tracer_metadata["region_mask_varname"])

        # generate dictionary of grid_vars
        self.grid_vars = {
            region_mask_varname: gen_grid_vars(
                lvl, modelinfo["grid_vars_fname"], region_mask_varname
            )
            for region_mask_varname in region_mask_varnames
        }

        # confirm that all region_masks have the same number of regions
        region_cnts = set(
            grid_vars["region_cnt"] for grid_vars in self.grid_vars.values()
        )
        if len(region_cnts) != 1:
            raise RuntimeError("not all region_masks have the same region_cnt")
        self.region_cnt = region_cnts.pop()

    def tracer_module_expand_all(self, tracer_module_names):
        """
        Perform substitution/expansion on parameterized tracer modules.
        Generate new tracer module definitions.
        """

        tracer_module_names_new = []
        for tracer_module_name in tracer_module_names.split(","):
            if ":" not in tracer_module_name:
                tracer_module_names_new.append(tracer_module_name)
                continue
            (tracer_module_name_root, _, suffs) = tracer_module_name.partition(":")
            for suff in suffs.split(":"):
                tracer_module_name_new = self.tracer_module_expand_one(
                    tracer_module_name_root, suff
                )
                tracer_module_names_new.append(tracer_module_name_new)
        return ",".join(tracer_module_names_new)

    def tracer_module_expand_one(self, tracer_module_name_root, suff):
        """
        Perform substitution/expansion on parameterized tracer modules.
        Generate new tracer module definitions.
        """

        fmt = {"suff": suff}

        tracer_module_name_new = tracer_module_name_root.format(**fmt)
        # construct new tracer_module_def
        # with {suff} replaced with suff throughout metadata
        tracer_module_def_root = self.tracer_module_defs[tracer_module_name_root]
        tracer_module_def = fmt_vals(tracer_module_def_root, fmt)
        self.tracer_module_defs[tracer_module_name_new] = tracer_module_def

        # apply replacement to referenced precond matrices,
        # if their name is parameterized
        for tracer_metadata in tracer_module_def_root["tracers"].values():
            if "precond_matrix" in tracer_metadata:
                matrix_name = tracer_metadata["precond_matrix"]
                matrix_name_new = matrix_name.format(**fmt)
                if matrix_name_new != matrix_name:
                    self.precond_matrix_defs[matrix_name_new] = fmt_vals(
                        self.precond_matrix_defs[matrix_name], fmt
                    )

        return tracer_module_name_new


def check_tracer_module_names(tracer_module_names, tracer_module_defs):
    """
    Confirm that tracer_module_names names exist in tracer_module_defs and that
    parameterized tracer modules in tracer_module_names are provided a suffix.
    """

    fmt = {"suff": "suff"}  # dummy suff replacement

    for tracer_module_name in tracer_module_names.split(","):
        has_suff = ":" in tracer_module_name
        if has_suff:
            tracer_module_name = tracer_module_name.partition(":")[0]
        if tracer_module_name not in tracer_module_defs:
            raise ValueError(f"unknown tracer module name {tracer_module_name}")
        if has_suff == (tracer_module_name.format(**fmt) == tracer_module_name):
            verb = "doesn't expect" if has_suff else "expects"
            raise ValueError(f"{tracer_module_name} {verb} suff")


def check_shadow_tracers(tracer_module_defs, lvl):
    """Confirm that tracers specified in shadow_tracers are also in tracer_names."""
    # This check is done for all entries in tracer_module_defs,
    # whether they are being used or not.
    logger = logging.getLogger(__name__)
    for tracer_module_name, tracer_module_def in tracer_module_defs.items():
        shadowed_tracers = []
        # Verify that shadows is a known tracer names.
        # Verify that no tracer is shadowed multiple times.
        for tracer_name, tracer_metadata in tracer_module_def["tracers"].items():
            if "shadows" in tracer_metadata:
                if tracer_metadata["shadows"] not in tracer_module_def["tracers"]:
                    raise ValueError(
                        f'shadows value {tracer_metadata["shadows"]} for {tracer_name} '
                        f"in tracer module {tracer_module_name} not known"
                    )
                logger.log(
                    lvl,
                    "tracer module %s has %s as a shadow for %s",
                    tracer_module_name,
                    tracer_name,
                    tracer_metadata["shadows"],
                )
                if tracer_metadata["shadows"] in shadowed_tracers:
                    raise ValueError(
                        f'{tracer_metadata["shadows"]} shadowed multiple times in '
                        f"tracer module {tracer_module_name}"
                    )
                shadowed_tracers.append(tracer_metadata["shadows"])


def check_tracer_module_suffs(tracer_module_defs):
    """
    Confirm that tracer module names with a suff correspond to tracer module defs with a
    suff. Confirm that tracer names in tracer modules with a suff have a suff.
    """

    fmt = {"suff": "suff"}  # dummy suff replacement

    for name, metadata in tracer_module_defs.items():
        name_has_suff = name.format(**fmt) != name
        metadata_has_suff = fmt_vals(metadata, fmt) != metadata
        if name_has_suff != metadata_has_suff:
            raise ValueError(f"{name}: name_has_suff must equal metadata_has_suff")
        if name_has_suff:
            for tracer_name in metadata["tracers"]:
                if tracer_name.format(**fmt) == tracer_name:
                    raise ValueError(f"{name}: tracer {tracer_name} must have suff")


def propagate_base_matrix_defs_to_all(matrix_defs):
    """propagate matrix_defs from matrix_def 'base' to all other matrix_defs"""
    logger = logging.getLogger(__name__)
    if "base" not in matrix_defs:
        return
    for matrix_name, matrix_def in matrix_defs.items():
        if matrix_name != "base":
            logger.debug("propagating matrix def to %s", matrix_name)
            propagate_base_matrix_defs_to_one(matrix_defs["base"], matrix_def)


def propagate_base_matrix_defs_to_one(base_def, matrix_def):
    """propagate matrix_defs from base_def to one matrix_def"""
    for base_def_key, base_def_value in base_def.items():
        if base_def_key not in matrix_def:
            matrix_def[base_def_key] = copy.deepcopy(base_def_value)
        else:
            matrix_def_value = matrix_def[base_def_key]
            if isinstance(base_def_value, list):
                # generate list of 1st words of opts from matrix_def_value
                matrix_def_value_word0 = [opt.split()[0] for opt in matrix_def_value]
                for opt in base_def_value:
                    # only append opt to matrix_def_value if opt's 1st word isn't
                    # already present in list of 1st words
                    if opt.split()[0] not in matrix_def_value_word0:
                        matrix_def_value.append(opt)
            elif isinstance(base_def_value, dict):
                for key in base_def_value:
                    if key not in matrix_def_value:
                        matrix_def_value[key] = base_def_value[key]
            else:
                raise TypeError(f"base defn type {type(base_def_value)} not supported")


def check_precond_matrix_defs(precond_matrix_defs):
    """Perform basic vetting of precond_matrix_defs"""
    # This check is done for all entries in def_dict,
    # whether they are being used or not.
    logger = logging.getLogger(__name__)
    for precond_matrix_name, precond_matrix_def in precond_matrix_defs.items():
        logger.debug("checking precond_matrix_def for %s", precond_matrix_name)
        # verify that suffixes in hist_to_precond_varnames are recognized
        if "hist_to_precond_varnames" in precond_matrix_def:
            for hist_var in precond_matrix_def["hist_to_precond_varnames"]:
                _, _, time_op = hist_var.partition(":")
                if time_op not in ["mean", "log_mean", ""]:
                    raise ValueError(
                        f"unknown time_op={time_op} in {hist_var} from "
                        f"{precond_matrix_name}"
                    )


def gen_grid_vars(lvl, grid_vars_fname, region_mask_varname):
    """
    Return dict of grid vars related to region_mask_varname,
    reading fields from grid_vars_fname. Grid vars that are generated/read are
    region_mask: region indices (1, 2, ...), read from region_mask_varname
    grid_weight: cell weights for averaging, read from variable whose name is
        determined from cell_measures attribute of region_mask
    region_comp_mean_matrix: sparse matrix to compute regional means, generated
        from region_mask and grid_weight
    region_cnt: number of regions
    """
    logger = logging.getLogger(__name__)
    logger.log(
        lvl, "reading grid_vars for %s from %s", region_mask_varname, grid_vars_fname
    )

    res = {}

    with Dataset(grid_vars_fname, mode="r") as fptr:
        fptr.set_auto_mask(False)
        region_mask_var = fptr.variables[region_mask_varname]
        res["region_mask"] = region_mask_var[:]
        cell_measures = region_mask_var.cell_measures
        cell_measures_split = cell_measures.split(":")
        if len(cell_measures_split) != 2:
            raise RuntimeError(
                f"unexpected number of words in {region_mask_varname}:cell_measures"
            )
        grid_weight_varname = cell_measures_split[-1].split()[0]
        res["grid_weight"] = fptr.variables[grid_weight_varname][:]

    # enforce that region_mask and grid_weight and both 0 where one of them is
    res["region_mask"][:] = np.where(res["grid_weight"] == 0.0, 0, res["region_mask"])
    res["grid_weight"][:] = np.where(res["region_mask"] == 0, 0.0, res["grid_weight"])

    res["region_cnt"] = res["region_mask"].max()
    res["region_comp_mean_matrix"] = gen_region_mean_sparse(
        res["region_mask"], res["region_cnt"], res["grid_weight"]
    )

    return res


def gen_region_mean_sparse(region_mask, region_cnt, grid_weight):
    """Generate sparse matrix used for computing means over regions."""

    region_mask_flat = region_mask.reshape(-1)
    grid_weight_flat = grid_weight.reshape(-1)

    indices = []
    indptr = [0]
    data = []

    for region_ind in range(region_cnt):
        indices.extend(np.nonzero(region_mask_flat == region_ind + 1)[0])
        indptr.append(len(indices))
        data_row_raw = grid_weight_flat[indices[indptr[-2] : indptr[-1]]]
        data_row_raw_sum_r = 1.0 / sum(data_row_raw)
        data.extend([data_row_raw_sum_r * val for val in data_row_raw])

    arg1 = (data, indices, indptr)
    shape = (region_cnt, len(grid_weight_flat))
    return (
        scipy.sparse.csr_array(arg1=arg1, shape=shape)
        if Version(scipy.__version__) >= Version("1.8.0")
        else scipy.sparse.csr_matrix(arg1=arg1, shape=shape)
    )
