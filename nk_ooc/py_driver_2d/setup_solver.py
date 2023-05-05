#!/usr/bin/env python
"""set up files needed to run NK solver for py_driver_2d"""

import cProfile
import logging
import os
import pstats
import sys
from datetime import datetime

import numpy as np
from netCDF4 import Dataset

from .. import gen_invoker_script
from ..model_config import ModelConfig
from ..share import (
    args_replace,
    common_args,
    logging_config,
    read_cfg_files,
    repro_fname,
)
from ..spatial_axis import spatial_axis_defn_dict, spatial_axis_from_defn_dict
from ..utils import create_dimensions_verify, create_vars, mkdir_exist_okay
from .model_state import ModelState


def parse_args(args_list_in=None):
    """parse command line arguments"""

    args_list = [] if args_list_in is None else args_list_in
    parser, args_remaining = common_args(
        "setup solver for py_driver_2d model", "py_driver_2d", args_list
    )

    parser.add_argument(
        "--init_iterate_opt",
        help="option for specifying initial iterate",
        default="gen_init_iterate",
    )
    parser.add_argument(
        "--fp_cnt",
        type=int,
        help="number of fixed point iterations to apply to init_iterate",
        default=2,
    )
    parser.add_argument(
        "--prof_comp_fcn_fname",
        help="profile comp_fcn call; write output to provided argument",
        default=None,
    )
    parser.add_argument(
        "--deprecation_warning_to_error",
        help="treat DeprecationWarning warnings as errors",
        action="store_true",
    )

    return args_replace(parser.parse_args(args_remaining))


def main(args):
    """set up files needed to run NK solver for py_driver_2d"""

    config = read_cfg_files(args)
    solverinfo = config["solverinfo"]

    logging_config(solverinfo, filemode="w")
    logger = logging.getLogger(__name__)

    logger.info('args.cfg_fnames="%s"', repro_fname(solverinfo, args.cfg_fnames))

    # ensure workdir exists
    mkdir_exist_okay(solverinfo["workdir"])

    # generate invoker script
    args.model_name = "py_driver_2d"
    gen_invoker_script.main(args)

    modelinfo = config["modelinfo"]

    caller = "nk_ooc.py_driver_2d.setup_solver.main"

    # generate grid_vars file
    grid_vars_fname = modelinfo["grid_vars_fname"]
    logger.info('grid_vars_fname="%s"', repro_fname(modelinfo, grid_vars_fname))
    mkdir_exist_okay(os.path.dirname(grid_vars_fname))
    gen_grid_vars_file(args, modelinfo)

    # confirm that model configuration works with generated file
    # ModelState relies on model being configured
    ModelState.model_config_obj = ModelConfig(modelinfo)

    # generate initial condition
    init_iterate = ModelState(args.init_iterate_opt)

    if args.prof_comp_fcn_fname is not None:
        cProfile.runctx(
            "init_iterate.comp_fcn(res_fname=None, solver_state=None, hist_fname=None)",
            globals=None,
            locals={"init_iterate": init_iterate},
            filename=args.prof_comp_fcn_fname,
        )
        stats_obj = pstats.Stats(args.prof_comp_fcn_fname)
        stats_obj.strip_dirs().sort_stats("time").print_stats(20)
        return

    # perform fixed point iteration(s) on init_iterate
    if args.fp_cnt > 0:
        workdir = solverinfo["workdir"]
        gen_init_iterate_workdir = os.path.join(workdir, "gen_init_iterate")
        mkdir_exist_okay(gen_init_iterate_workdir)

        for fp_iter in range(args.fp_cnt):
            logger.info("fp_iter=%d", fp_iter)
            init_iterate.dump(
                os.path.join(gen_init_iterate_workdir, f"init_iterate_{fp_iter:04}.nc"),
                caller,
            )
            init_iterate_fcn = init_iterate.comp_fcn(
                os.path.join(gen_init_iterate_workdir, f"fcn_{fp_iter:04}.nc"),
                None,
                os.path.join(gen_init_iterate_workdir, f"hist_{fp_iter:04}.nc"),
            )
            init_iterate += init_iterate_fcn
            init_iterate.copy_shadow_tracers_to_real_tracers()

    # write generated init_iterate to where solver expects it to be
    init_iterate_fname = solverinfo["init_iterate_fname"]
    logger.info('init_iterate_fname="%s"', repro_fname(solverinfo, init_iterate_fname))
    mkdir_exist_okay(os.path.dirname(init_iterate_fname))
    init_iterate.dump(init_iterate_fname, caller)


def gen_grid_vars_file(args, modelinfo):
    """generate grid vars file based on args and modelinfo"""

    # generate axes from args and modelinfo
    axisnames = ["depth", "ypos"]
    axes = {axisname: gen_axis(axisname, args, modelinfo) for axisname in axisnames}

    with Dataset(
        modelinfo["grid_vars_fname"], mode="w", format="NETCDF3_64BIT_OFFSET"
    ) as fptr:
        datestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        name = "nk_ooc.py_driver_2d.setup_solver.gen_grid_vars_file"
        fptr.history = f"{datestamp}: created by {name}"

        for axis in axes.values():
            create_dimensions_verify(fptr, axis.dump_dimensions())
            create_vars(fptr, axis.dump_vars_metadata())

        vars_metadata = {}
        vars_metadata["grid_weight"] = {
            "dimensions": tuple(axisnames),
            "attrs": {"long_name": "grid-cell area", "units": "m^2"},
        }
        vars_metadata["region_mask"] = {
            "datatype": "i4",
            "dimensions": tuple(axisnames),
            "attrs": {
                "long_name": "Region Mask",
                "cell_measures": "area: grid_weight",
            },
        }
        create_vars(fptr, vars_metadata)

        for axis in axes.values():
            axis.dump_write(fptr)

        weight = np.outer(axes["depth"].delta, axes["ypos"].delta)
        fptr.variables["grid_weight"][:] = weight

        max_abs_vvel = float(modelinfo["max_abs_vvel"])
        horiz_mix_coeff = float(modelinfo["horiz_mix_coeff"])
        if max_abs_vvel == 0.0 and horiz_mix_coeff == 0.0:
            mask = np.empty(weight.shape, dtype=np.int32)
            for ypos_i in range(weight.shape[1]):
                mask[:, ypos_i] = ypos_i + 1
        else:
            mask = np.ones(weight.shape, dtype=np.int32)

        fptr.variables["region_mask"][:] = mask


def gen_axis(axisname, args, modelinfo):
    """
    generate axis object based on args and modelinfo
    """

    defn_dict = {}
    for key, defn in spatial_axis_defn_dict(axisname=axisname).items():
        axis_key = "_".join([axisname, key])
        if axis_key in modelinfo:
            defn_dict[key] = (defn["type"])(modelinfo[axis_key])
        if hasattr(args, axis_key):
            defn_dict[key] = getattr(args, axis_key)

    return spatial_axis_from_defn_dict(defn_dict=spatial_axis_defn_dict(**defn_dict))


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
