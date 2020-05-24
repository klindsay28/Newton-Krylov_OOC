#!/usr/bin/env python
"""set up files needed to run NK solver for cime_pop"""

from datetime import datetime
from distutils.util import strtobool
import logging
import os
import sys

from netCDF4 import Dataset
import numpy as np

from ..cime import cime_xmlquery, cime_yr_cnt
from .. import gen_invoker_script
from ..model_config import ModelConfig
from ..share import parse_args_common, read_cfg_file
from ..utils import mkdir_exist_okay, ann_files_to_mean_file, mon_files_to_mean_file


def _parse_args():
    """parse command line arguments"""
    parser = parse_args_common("setup cime_pop", model_name="cime_pop")
    parser.add_argument(
        "--skip_irf_gen",
        help="skip generating irf file if it exists, default is to overwrite it",
        action="store_true",
    )

    args = parser.parse_args()

    # replace {model_name} with specified model
    args.cfg_fname = args.cfg_fname.replace("{model_name}", args.model_name)

    return args


def main(args):
    """set up files needed to run NK solver for cime_pop"""

    config = read_cfg_file(args)
    solverinfo = config["solverinfo"]

    logging_format = "%(asctime)s:%(process)s:%(filename)s:%(funcName)s:%(message)s"
    logging.basicConfig(
        stream=sys.stdout, format=logging_format, level=solverinfo["logging_level"]
    )
    logger = logging.getLogger(__name__)

    logger.info('args.model_name="%s"', args.model_name)
    logger.info('args.cfg_fname="%s"', args.cfg_fname)

    modelinfo = config["modelinfo"]

    # generate irf file
    irf_fname = modelinfo["irf_fname"]
    if os.path.exists(irf_fname) and args.skip_irf_gen:
        logger.info('irf_fname="%s" exists, skipping generation', irf_fname)
    else:
        logger.info('generating irf_fname="%s"', irf_fname)
        mkdir_exist_okay(os.path.dirname(irf_fname))
        gen_irf_file(modelinfo)

    # generate grid files from irf file
    grid_weight_fname = modelinfo["grid_weight_fname"]
    logger.info('grid_weight_fname="%s"', grid_weight_fname)
    mkdir_exist_okay(os.path.dirname(grid_weight_fname))
    gen_grid_weight_file(modelinfo)

    region_mask_fname = modelinfo["region_mask_fname"]
    logger.info('region_mask_fname="%s"', region_mask_fname)
    mkdir_exist_okay(os.path.dirname(region_mask_fname))
    gen_region_mask_file(modelinfo)

    # confirm that model configuration works with generated file
    # ModelState relies on model being configured
    ModelConfig(modelinfo)

    # generate invoker script
    gen_invoker_script.main(args)


def gen_irf_file(modelinfo):
    """generate irf file, based on modelinfo"""

    irf_hist_freq_opt = modelinfo["irf_hist_freq_opt"]

    if irf_hist_freq_opt not in ["nyear", "nmonth"]:
        msg = "irf_hist_freq_opt = %s not implemented" % irf_hist_freq_opt
        raise NotImplementedError(msg)

    hist_dir = modelinfo["irf_hist_dir"]

    # get start date for date range getting averaged into irf file

    # fallbacks values if they are not specified in the cfg file
    if modelinfo["irf_hist_start_date"] is None:
        caseroot = modelinfo["caseroot"]
        if cime_xmlquery("RUN_TYPE", caseroot=caseroot) == "branch":
            irf_hist_start_date = cime_xmlquery("RUN_REFDATE", caseroot=caseroot)
        else:
            irf_hist_start_date = cime_xmlquery("RUN_STARTDATE", caseroot=caseroot)
    else:
        irf_hist_start_date = modelinfo["irf_hist_start_date"]

    (irf_hist_year0, irf_hist_month0, irf_hist_day0) = irf_hist_start_date.split("-")

    # basic error checking

    if irf_hist_day0 != "01":
        msg = "irf_hist_day0 = %s not implemented" % irf_hist_day0
        raise NotImplementedError(msg)

    if irf_hist_freq_opt == "nyear" and irf_hist_month0 != "01":
        msg = (
            "irf_hist_month0 = %s not implemented for nyear tavg output"
            % irf_hist_month0
        )
        raise NotImplementedError(msg)

    # get duration of date range getting averaged into irf file

    if modelinfo["irf_hist_yr_cnt"] is None:
        irf_hist_yr_cnt = cime_yr_cnt(modelinfo)
    else:
        irf_hist_yr_cnt = modelinfo["irf_hist_yr_cnt"]

    caller = "src.cime_pop.setup_solver.gen_irf_file"

    if irf_hist_freq_opt == "nyear":
        fname_fmt = modelinfo["irf_case"] + ".pop.h.{year:04d}.nc"
        ann_files_to_mean_file(
            hist_dir,
            fname_fmt,
            int(irf_hist_year0),
            int(irf_hist_yr_cnt),
            modelinfo["irf_fname"],
            caller,
        )

    if irf_hist_freq_opt == "nmonth":
        fname_fmt = modelinfo["irf_case"] + ".pop.h.{year:04d}-{month:02d}.nc"
        mon_files_to_mean_file(
            hist_dir,
            fname_fmt,
            int(irf_hist_year0),
            int(irf_hist_month0),
            12 * int(irf_hist_yr_cnt),
            modelinfo["irf_fname"],
            caller,
        )


def gen_grid_weight_file(modelinfo):
    """generate weight file from irf file, based on modelinfo"""

    weight_dimnames = ("z_t", "nlat", "nlon")

    with Dataset(modelinfo["irf_fname"], mode="r") as fptr_in:
        if hasattr(fptr_in, "history"):
            history_in = getattr(fptr_in, "history")
        else:
            history_in = None
        # generate weight
        dz = 1.0e-2 * fptr_in.variables["dz"][:]  # convert from cm to m
        tarea = 1.0e-4 * fptr_in.variables["TAREA"][:]  # convert from cm2 to m2
        kmt = fptr_in.variables["KMT"][:]
        region_mask = fptr_in.variables["REGION_MASK"][:]

        surf_mask = region_mask > 0
        if strtobool(modelinfo["include_black_sea"]):
            surf_mask = surf_mask | (region_mask == -13)

        weight_shape = tuple(
            fptr_in.dimensions[dimname].size for dimname in weight_dimnames
        )
        weight = np.empty(weight_shape)
        for k in range(weight_shape[0]):
            weight[k, :, :] = dz[k] * np.where((k < kmt) & surf_mask, tarea, 0.0)

    mode_out = "w"

    with Dataset(
        modelinfo["grid_weight_fname"], mode=mode_out, format="NETCDF3_64BIT_OFFSET"
    ) as fptr_out:
        datestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        name = "src.cime_pop.setup_solver.gen_grid_weight_file"
        msg = datestamp + ": created by " + name + " from " + modelinfo["irf_fname"]
        if history_in is not None:
            msg = msg + "\n" + history_in
        setattr(fptr_out, "history", msg)

        # propagate dimension sizes from fptr_in to fptr_out
        for dimind, dimname in enumerate(weight_dimnames):
            if dimname not in fptr_out.dimensions:
                fptr_out.createDimension(dimname, weight_shape[dimind])

        varname = modelinfo["grid_weight_varname"]
        var = fptr_out.createVariable(varname, weight.dtype, dimensions=weight_dimnames)
        setattr(var, "long_name", "Ocean Grid-Cell Volume")
        setattr(var, "units", "m3")

        var[:] = weight


def gen_region_mask_file(modelinfo):
    """generate region_mask file from irf file, based on modelinfo"""

    mask_dimnames = ("z_t", "nlat", "nlon")

    with Dataset(modelinfo["irf_fname"], mode="r") as fptr_in:
        history_in = getattr(fptr_in, "history")
        # generate mask
        kmt = fptr_in.variables["KMT"][:]
        region_mask = fptr_in.variables["REGION_MASK"][:]

        mask_shape = tuple(
            fptr_in.dimensions[dimname].size for dimname in mask_dimnames
        )
        mask = np.empty(mask_shape, dtype=kmt.dtype)
        for k in range(mask_shape[0]):
            mask[k, :] = np.where((k < kmt) & (region_mask > 0), 1, 0)

        if strtobool(modelinfo["include_black_sea"]):
            for k in range(mask_shape[0]):
                mask[k, :] = np.where((k < kmt) & (region_mask == -13), 2, mask[k, :])

    mode_out = (
        "a" if modelinfo["region_mask_fname"] == modelinfo["grid_weight_fname"] else "w"
    )

    with Dataset(
        modelinfo["region_mask_fname"], mode=mode_out, format="NETCDF3_64BIT_OFFSET"
    ) as fptr_out:
        datestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        name = "src.cime_pop.setup_solver.gen_region_mask_file"
        msg = datestamp + ": "
        if mode_out == "a":
            history_in = getattr(fptr_out, "history")
            vars_appended = ",".join([modelinfo["region_mask_varname"], "DYN_REGMASK"])
            msg = msg + vars_appended + " appended by " + name
        else:
            msg = msg + "created by " + name + " from " + modelinfo["irf_fname"]
        msg = msg + "\n" + history_in
        setattr(fptr_out, "history", msg)

        # propagate dimension sizes from fptr_in to fptr_out
        for dimind, dimname in enumerate(mask_dimnames):
            if dimname not in fptr_out.dimensions:
                fptr_out.createDimension(dimname, mask_shape[dimind])

        varname = modelinfo["region_mask_varname"]
        var = fptr_out.createVariable(varname, mask.dtype, dimensions=mask_dimnames)
        setattr(var, "long_name", "Region Mask")

        var[:] = mask

        varname = "DYN_REGMASK"
        var = fptr_out.createVariable(varname, mask.dtype, dimensions=mask_dimnames[1:])
        setattr(var, "long_name", "Region Mask")

        var[:] = mask[0, :]


################################################################################

if __name__ == "__main__":
    main(_parse_args())
