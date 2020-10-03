#!/usr/bin/env python
"""set up files needed to run NK solver for cime_pop"""

import glob
import logging
import os
import shutil
import sys
from datetime import datetime
from distutils.util import strtobool

import numpy as np
from netCDF4 import Dataset

from .. import gen_invoker_script
from ..cime import cime_xmlquery, cime_yr_cnt
from ..model_config import ModelConfig, get_modelinfo
from ..share import (
    args_replace,
    common_args,
    logging_config,
    read_cfg_file,
    repro_fname,
)
from ..utils import (
    ann_files_to_mean_file,
    create_dimensions_verify,
    create_vars,
    extract_dimensions,
    mkdir_exist_okay,
    mon_files_to_mean_file,
)


def parse_args(args_list_in=None):
    """parse command line arguments"""

    args_list = [] if args_list_in is None else args_list_in
    parser, args_remaining = common_args(
        "setup solver for cime_pop model", "cime_pop", args_list
    )

    parser.add_argument(
        "--skip_irf_gen",
        help="skip generating irf file if it exists, default is to overwrite it",
        action="store_true",
    )

    return args_replace(parser.parse_args(args_remaining))


def main(args):
    """set up files needed to run NK solver for cime_pop"""

    config = read_cfg_file(args)
    solverinfo = config["solverinfo"]

    logging_config(args, solverinfo, filemode="w")
    logger = logging.getLogger(__name__)

    logger.info('args.cfg_fname="%s"', repro_fname(solverinfo, args.cfg_fname))

    # ensure workdir exists
    mkdir_exist_okay(solverinfo["workdir"])

    # copy rpointer files from RUNDIR to rpointer_dir
    rundir = cime_xmlquery("RUNDIR")
    rpointer_dir = get_modelinfo("rpointer_dir")
    mkdir_exist_okay(rpointer_dir)
    for src in glob.glob(os.path.join(rundir, "rpointer.*")):
        shutil.copy(src, rpointer_dir)

    # generate invoker script
    args.model_name = "cime_pop"
    gen_invoker_script.main(args)

    modelinfo = config["modelinfo"]

    # generate irf file
    irf_fname = modelinfo["irf_fname"]
    if os.path.exists(irf_fname) and args.skip_irf_gen:
        logger.info(
            'irf_fname="%s" exists, skipping generation',
            repro_fname(modelinfo, irf_fname),
        )
    else:
        logger.info('generating irf_fname="%s"', repro_fname(modelinfo, irf_fname))
        mkdir_exist_okay(os.path.dirname(irf_fname))
        gen_irf_file(modelinfo)

    # generate grid files from irf file
    grid_weight_fname = modelinfo["grid_weight_fname"]
    logger.info('grid_weight_fname="%s"', repro_fname(modelinfo, grid_weight_fname))
    mkdir_exist_okay(os.path.dirname(grid_weight_fname))
    gen_grid_weight_file(modelinfo)

    region_mask_fname = modelinfo["region_mask_fname"]
    logger.info('region_mask_fname="%s"', repro_fname(modelinfo, region_mask_fname))
    mkdir_exist_okay(os.path.dirname(region_mask_fname))
    gen_region_mask_file(modelinfo)

    ModelConfig(modelinfo)


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

    with Dataset(modelinfo["irf_fname"], mode="r") as fptr_in:
        history_in = getattr(fptr_in, "history", None)
        # generate weight
        thickness = 1.0e-2 * fptr_in.variables["dz"][:]  # convert from cm to m
        area = 1.0e-4 * fptr_in.variables["TAREA"][:]  # convert from cm2 to m2
        kmt = fptr_in.variables["KMT"][:]
        region_mask = fptr_in.variables["REGION_MASK"][:]

        surf_mask = region_mask > 0
        if strtobool(modelinfo["include_black_sea"]):
            surf_mask = surf_mask | (region_mask == -13)

        weight_dimensions = extract_dimensions(fptr_in, ["dz", "TAREA"])
        weight = np.empty(tuple(weight_dimensions.values()))
        for k in range(weight.shape[0]):
            weight[k, :, :] = thickness[k] * np.where((k < kmt) & surf_mask, area, 0.0)

    with Dataset(
        modelinfo["grid_weight_fname"], mode="w", format="NETCDF3_64BIT_OFFSET"
    ) as fptr_out:
        datestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        name = "src.cime_pop.setup_solver.gen_grid_weight_file"
        msg = datestamp + ": created by " + name + " from " + modelinfo["irf_fname"]
        fptr_out.history = msg if history_in is None else "\n".join([msg, history_in])

        # propagate dimension sizes from fptr_in to fptr_out
        create_dimensions_verify(fptr_out, weight_dimensions)

        vars_metadata = {}
        vars_metadata[modelinfo["grid_weight_varname"]] = {
            "datatype": weight.dtype,
            "dimensions": tuple(weight_dimensions),
            "attrs": {"long_name": "Ocean Grid-Cell Volume", "units": "m^3"},
        }
        create_vars(fptr_out, vars_metadata)

        fptr_out.variables[modelinfo["grid_weight_varname"]][:] = weight


def gen_region_mask_file(modelinfo):
    """generate region_mask file from irf file, based on modelinfo"""

    with Dataset(modelinfo["irf_fname"], mode="r") as fptr_in:
        history_in = getattr(fptr_in, "history", None)
        # generate mask
        kmt = fptr_in.variables["KMT"][:]
        region_mask = fptr_in.variables["REGION_MASK"][:]

        mask_dimensions = extract_dimensions(fptr_in, ["z_t", "KMT"])
        mask = np.empty(tuple(mask_dimensions.values()), dtype=kmt.dtype)
        for k in range(mask.shape[0]):
            mask[k, :] = np.where((k < kmt) & (region_mask > 0), 1, 0)

        if strtobool(modelinfo["include_black_sea"]):
            for k in range(mask.shape[0]):
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
            history_in = getattr(fptr_out, "history", None)
            vars_appended = ",".join([modelinfo["region_mask_varname"], "DYN_REGMASK"])
            msg = msg + vars_appended + " appended by " + name
        else:
            msg = msg + "created by " + name + " from " + modelinfo["irf_fname"]
        fptr_out.history = msg if history_in is None else "\n".join([msg, history_in])

        # propagate dimension sizes from fptr_in to fptr_out
        create_dimensions_verify(fptr_out, mask_dimensions)

        vars_metadata = {}
        vars_metadata[modelinfo["region_mask_varname"]] = {
            "datatype": mask.dtype,
            "dimensions": tuple(mask_dimensions),
            "attrs": {"long_name": "Region Mask"},
        }
        vars_metadata["DYN_REGMASK"] = {
            "datatype": mask.dtype,
            "dimensions": tuple(mask_dimensions)[1:],
            "attrs": {"long_name": "Region Mask"},
        }
        create_vars(fptr_out, vars_metadata)

        fptr_out.variables[modelinfo["region_mask_varname"]][:] = mask

        fptr_out.variables["DYN_REGMASK"][:] = mask[0, :]


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
