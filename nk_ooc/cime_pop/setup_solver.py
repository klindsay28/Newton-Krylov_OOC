#!/usr/bin/env python
"""set up files needed to run NK solver for cime_pop"""

import glob
import logging
import os
import shutil
import sys
from datetime import datetime

import numpy as np
from netCDF4 import Dataset

from .. import gen_invoker_script
from ..cime import cime_xmlquery, cime_yr_cnt
from ..model_config import ModelConfig
from ..share import (
    args_replace,
    common_args,
    logging_config,
    read_cfg_files,
    repro_fname,
)
from ..utils import (
    ann_files_to_mean_file,
    create_dimensions_verify,
    create_vars,
    extract_dimensions,
    mkdir_exist_okay,
    mon_files_to_mean_file,
    strtobool,
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

    config = read_cfg_files(args)
    solverinfo = config["solverinfo"]
    modelinfo = config["modelinfo"]

    logging_config(solverinfo, filemode="w")
    logger = logging.getLogger(__name__)

    logger.info('args.cfg_fnames="%s"', repro_fname(solverinfo, args.cfg_fnames))

    # ensure workdir exists
    mkdir_exist_okay(solverinfo["workdir"])

    # copy rpointer files from RUNDIR to rpointer_dir
    rundir = cime_xmlquery(modelinfo["caseroot"], "RUNDIR")
    rpointer_dir = modelinfo["rpointer_dir"]
    mkdir_exist_okay(rpointer_dir)
    for src in glob.glob(os.path.join(rundir, "rpointer.*")):
        shutil.copy(src, rpointer_dir)

    # generate invoker script
    args.model_name = "cime_pop"
    gen_invoker_script.main(args)

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

    # generate grid_vars file
    grid_vars_fname = modelinfo["grid_vars_fname"]
    logger.info('grid_vars_fname="%s"', repro_fname(modelinfo, grid_vars_fname))
    mkdir_exist_okay(os.path.dirname(grid_vars_fname))
    gen_grid_vars_file(modelinfo)

    # confirm that generated files can be read
    ModelConfig(modelinfo)


def gen_irf_file(modelinfo):
    """generate irf file, based on modelinfo"""

    irf_hist_freq_opt = modelinfo["irf_hist_freq_opt"]

    if irf_hist_freq_opt not in ["nyear", "nmonth"]:
        raise NotImplementedError(
            f"irf_hist_freq_opt={irf_hist_freq_opt} not implemented"
        )

    # get start date for date range getting averaged into irf file

    # fallbacks values if they are not specified in the cfg file
    if modelinfo["irf_hist_start_date"] is None:
        if cime_xmlquery(modelinfo["caseroot"], "RUN_TYPE") == "branch":
            varname = "RUN_REFDATE"
        else:
            varname = "RUN_STARTDATE"
        irf_hist_start_date = cime_xmlquery(modelinfo["caseroot"], varname)
    else:
        irf_hist_start_date = modelinfo["irf_hist_start_date"]

    (irf_hist_year0, irf_hist_month0, irf_hist_day0) = irf_hist_start_date.split("-")

    # basic error checking

    if irf_hist_day0 != "01":
        raise NotImplementedError(f"irf_hist_day0={irf_hist_day0} not implemented")

    if irf_hist_freq_opt == "nyear" and irf_hist_month0 != "01":
        raise NotImplementedError(
            f"irf_hist_month0={irf_hist_month0} not implemented for nyear tavg output"
        )

    # get duration of date range getting averaged into irf file

    if modelinfo["irf_hist_yr_cnt"] is None:
        irf_hist_yr_cnt = cime_yr_cnt(modelinfo)
    else:
        irf_hist_yr_cnt = modelinfo["irf_hist_yr_cnt"]

    caller = "nk_ooc.cime_pop.setup_solver.gen_irf_file"

    irf_case = modelinfo["irf_case"]

    if irf_hist_freq_opt == "nyear":
        fname_fmt = f"{irf_case}.pop.h.{{year:04}}.nc"
        ann_files_to_mean_file(
            modelinfo["irf_hist_dir"],
            fname_fmt,
            int(irf_hist_year0),
            int(irf_hist_yr_cnt),
            modelinfo["irf_fname"],
            caller,
        )

    if irf_hist_freq_opt == "nmonth":
        fname_fmt = f"{irf_case}.pop.h.{{year:04}}-{{month:02}}.nc"
        mon_files_to_mean_file(
            modelinfo["irf_hist_dir"],
            fname_fmt,
            int(irf_hist_year0),
            int(irf_hist_month0),
            12 * int(irf_hist_yr_cnt),
            modelinfo["irf_fname"],
            caller,
        )


def gen_grid_vars_file(modelinfo):
    """generate grid vars file from irf file, based on modelinfo"""

    irf_fname = modelinfo["irf_fname"]

    # read required variables and dimensions from irf file
    with Dataset(irf_fname, mode="r") as fptr_in:
        history_in = getattr(fptr_in, "history", None)
        thickness = 1.0e-2 * fptr_in.variables["dz"][:]  # convert from cm to m
        area = 1.0e-4 * fptr_in.variables["TAREA"][:]  # convert from cm2 to m2
        kmt = fptr_in.variables["KMT"][:]
        region_mask = fptr_in.variables["REGION_MASK"][:]
        dimensions_3d = extract_dimensions(fptr_in, ["z_t", "KMT"])

    # generate mask
    mask = np.empty(tuple(dimensions_3d.values()), dtype=kmt.dtype)
    for k in range(mask.shape[0]):
        mask[k, :] = np.where((k < kmt) & (region_mask > 0), 1, 0)

    if strtobool(modelinfo["include_black_sea"]):
        for k in range(mask.shape[0]):
            mask[k, :] = np.where((k < kmt) & (region_mask == -13), 2, mask[k, :])

    # generate weight
    weight = np.empty(tuple(dimensions_3d.values()))
    for k in range(weight.shape[0]):
        weight[k, :, :] = thickness[k] * np.where(mask[k, :, :] > 0, area, 0.0)

    with Dataset(
        modelinfo["grid_vars_fname"], mode="w", format="NETCDF3_64BIT_OFFSET"
    ) as fptr_out:
        datestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        name = "nk_ooc.cime_pop.setup_solver.gen_grid_vars_file"
        msg = f"{datestamp}: created by {name} from {irf_fname}"
        fptr_out.history = msg if history_in is None else "\n".join([msg, history_in])

        # propagate dimension sizes from fptr_in to fptr_out
        create_dimensions_verify(fptr_out, dimensions_3d)

        vars_metadata = {}
        vars_metadata["region_mask"] = {
            "datatype": mask.dtype,
            "dimensions": tuple(dimensions_3d),
            "attrs": {
                "long_name": "Region Mask",
                "cell_measures": "volume: grid_weight",
            },
        }
        vars_metadata["DYN_REGMASK"] = {
            "datatype": mask.dtype,
            "dimensions": tuple(dimensions_3d)[1:],
            "attrs": {"long_name": "Surface Region Mask"},
        }
        vars_metadata["grid_weight"] = {
            "datatype": weight.dtype,
            "dimensions": tuple(dimensions_3d),
            "attrs": {"long_name": "Ocean Grid-Cell Volume", "units": "m^3"},
        }
        create_vars(fptr_out, vars_metadata)

        fptr_out.variables["region_mask"][:] = mask

        fptr_out.variables["DYN_REGMASK"][:] = mask[0, :]

        fptr_out.variables["grid_weight"][:] = weight


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
