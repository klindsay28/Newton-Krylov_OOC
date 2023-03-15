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

        # extract grid_weight from modelinfo config object
        fname = modelinfo["grid_weight_fname"]
        varname = modelinfo["grid_weight_varname"]
        logger.log(
            lvl,
            "reading %s from %s for grid_weight",
            varname,
            repro_fname(modelinfo, fname),
        )
        with Dataset(fname, mode="r") as fptr:
            fptr.set_auto_mask(False)
            self.grid_weight = fptr.variables[varname][:]

        # extract region_mask from modelinfo config object
        fname = modelinfo["region_mask_fname"]
        varname = modelinfo["region_mask_varname"]
        if fname is not None and varname is not None:
            logger.log(
                lvl,
                "reading %s from %s for region_mask",
                varname,
                repro_fname(modelinfo, fname),
            )
            with Dataset(fname, mode="r") as fptr:
                fptr.set_auto_mask(False)
                self.region_mask = fptr.variables[varname][:]
                if self.region_mask.shape != self.grid_weight.shape:
                    msg = "region_mask and grid_weight must have the same shape"
                    raise RuntimeError(msg)
        else:
            self.region_mask = np.ones_like(self.grid_weight, dtype=np.int32)

        # enforce that region_mask and grid_weight and both 0 where one of them is
        self.region_mask = np.where(self.grid_weight == 0.0, 0, self.region_mask)
        self.grid_weight = np.where(self.region_mask == 0, 0.0, self.grid_weight)

        self.region_mean_sparse = self.gen_region_mean_sparse()
        self.region_cnt = self.region_mean_sparse.shape[0]

    def gen_region_mean_sparse(self):
        """Generate sparse matrix used for computing means over regions."""

        region_mask_flat = self.region_mask.reshape(-1)
        grid_weight_flat = self.grid_weight.reshape(-1)

        indices = []
        indptr = [0]
        data = []

        region_cnt = self.region_mask.max()

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
            msg = f"unknown tracer module name {tracer_module_name}"
            raise ValueError(msg)
        if has_suff == (tracer_module_name.format(**fmt) == tracer_module_name):
            if has_suff:
                msg = f"{tracer_module_name} doesn't expect suff"
            else:
                msg = f"{tracer_module_name} expects suff"
            raise ValueError(msg)


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
                    msg = (
                        f'shadows value {tracer_metadata["shadows"]} for {tracer_name} '
                        f"in tracer module {tracer_module_name} not known"
                    )
                    raise ValueError(msg)
                logger.log(
                    lvl,
                    "tracer module %s has %s as a shadow for %s",
                    tracer_module_name,
                    tracer_name,
                    tracer_metadata["shadows"],
                )
                if tracer_metadata["shadows"] in shadowed_tracers:
                    msg = (
                        f'{tracer_metadata["shadows"]} shadowed multiple times in '
                        f"tracer module {tracer_module_name}"
                    )
                    raise ValueError(msg)
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
            msg = f"{name}: name_has_suff must equal metadata_has_suff"
            raise ValueError(msg)
        if name_has_suff:
            for tracer_name in metadata["tracers"]:
                if tracer_name.format(**fmt) == tracer_name:
                    msg = f"{name}: tracer {tracer_name} must have suff"
                    raise ValueError(msg)


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
                msg = f"base defn type {type(base_def_value)} not supported"
                raise TypeError(msg)


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
                    msg = (
                        f"unknown time_op={time_op} in {hist_var} from "
                        f"{precond_matrix_name}"
                    )
                    raise ValueError(msg)
