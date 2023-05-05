#!/usr/bin/env python
"""set up files needed to run NK solver for test_problem"""

import cProfile
import logging
import os
import pstats
import sys

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
from ..utils import create_vars, mkdir_exist_okay
from .model_state import ModelState


def parse_args(args_list_in=None):
    """parse command line arguments"""

    args_list = [] if args_list_in is None else args_list_in
    parser, args_remaining = common_args(
        "setup solver for test_problem model", "test_problem", args_list
    )

    # axis related arguments
    defn = spatial_axis_defn_dict(axisname="depth")["nlevs"]
    parser.add_argument(
        "--depth_nlevs", type=defn["type"], help=defn["help"], default=defn["value"]
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
    """set up files needed to run NK solver for test_problem"""

    config = read_cfg_files(args)
    solverinfo = config["solverinfo"]

    logging_config(solverinfo, filemode="w")
    logger = logging.getLogger(__name__)

    logger.info('args.cfg_fnames="%s"', repro_fname(solverinfo, args.cfg_fnames))

    # ensure workdir exists
    mkdir_exist_okay(solverinfo["workdir"])

    # generate invoker script
    args.model_name = "test_problem"
    gen_invoker_script.main(args)

    modelinfo = config["modelinfo"]
    depth_axisname = modelinfo["depth_axisname"]

    # generate depth axis from args and modelinfo
    defn_dict = {}
    for key, defn in spatial_axis_defn_dict(axisname=depth_axisname).items():
        depth_key = f"{depth_axisname}_{key}"
        if depth_key in modelinfo:
            defn_dict[key] = (defn["type"])(modelinfo[depth_key])
        if hasattr(args, depth_key):
            defn_dict[key] = getattr(args, depth_key)
    depth = spatial_axis_from_defn_dict(defn_dict=spatial_axis_defn_dict(**defn_dict))

    caller = "nk_ooc.test_problem.setup_solver.main"

    # generate grid_vars file
    grid_vars_fname = modelinfo["grid_vars_fname"]
    logger.info('grid_vars_fname="%s"', repro_fname(modelinfo, grid_vars_fname))
    mkdir_exist_okay(os.path.dirname(grid_vars_fname))
    depth.dump(grid_vars_fname, caller)

    # add region_mask to grid_vars file
    depth_delta_name = depth.dump_names["delta"]
    vars_metadata = {
        "region_mask": {
            "datatype": "i4",
            "dimensions": depth_axisname,
            "attrs": {
                "long_name": "Region Mask",
                "cell_measures": f"thickness: {depth_delta_name}",
            },
        }
    }
    with Dataset(grid_vars_fname, mode="a") as fptr:
        create_vars(fptr, vars_metadata)
        fptr.variables["region_mask"][:] = 1

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
                os.path.join(gen_init_iterate_workdir, f"init_iterate_{fp_iter:02}.nc"),
                caller,
            )
            init_iterate_fcn = init_iterate.comp_fcn(
                os.path.join(gen_init_iterate_workdir, f"fcn_{fp_iter:02}.nc"),
                None,
                os.path.join(gen_init_iterate_workdir, f"hist_{fp_iter:02}.nc"),
            )
            init_iterate += init_iterate_fcn
            init_iterate.copy_shadow_tracers_to_real_tracers()

    # write generated init_iterate to where solver expects it to be
    init_iterate_fname = solverinfo["init_iterate_fname"]
    logger.info('init_iterate_fname="%s"', repro_fname(solverinfo, init_iterate_fname))
    mkdir_exist_okay(os.path.dirname(init_iterate_fname))
    init_iterate.dump(init_iterate_fname, caller)


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
