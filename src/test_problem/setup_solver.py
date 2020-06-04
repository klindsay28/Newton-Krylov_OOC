#!/usr/bin/env python
"""set up files needed to run NK solver for test_problem"""

import logging
import os
import sys

from .. import gen_invoker_script
from ..model_config import ModelConfig
from ..share import args_replace, common_args, read_cfg_file
from ..utils import mkdir_exist_okay

from .model_state import ModelState
from .spatial_axis import SpatialAxis


def depth_defn_dict(trap_unknown=True, **kwargs):
    """return a defn_dict for depth axis defaults"""
    defn_dict = {
        "name": "depth",
        "units": "m",
        "nlevs": 30,
        "edge_start": 0.0,
        "edge_end": 900.0,
        "delta_ratio_max": 5.0,
    }
    for key, value in kwargs.items():
        if key in defn_dict:
            defn_dict[key] = value
        elif trap_unknown:
            msg = "unknown key %s" % key
            raise ValueError(msg)
    return defn_dict


def parse_args(args_list_in=None):
    """parse command line arguments"""

    args_list = [] if args_list_in is None else args_list_in
    parser, args_remaining = common_args(
        "setup test_problem", "test_problem", args_list
    )

    axis_defaults = depth_defn_dict()
    parser.add_argument(
        "--axisname", help="axis name", default=axis_defaults["name"],
    )
    parser.add_argument(
        "--units", help="axis units", default=axis_defaults["units"],
    )
    parser.add_argument(
        "--nlevs", type=int, help="number of layers", default=axis_defaults["nlevs"],
    )
    parser.add_argument(
        "--edge_start",
        type=float,
        help="start of edges",
        default=axis_defaults["edge_start"],
    )
    parser.add_argument(
        "--edge_end",
        type=float,
        help="end of edges",
        default=axis_defaults["edge_end"],
    )
    parser.add_argument(
        "--delta_ratio_max",
        type=float,
        help="maximum ratio of layer thicknesses",
        default=axis_defaults["delta_ratio_max"],
    )
    parser.add_argument(
        "--fp_cnt",
        type=int,
        help="number of fixed point iterations to apply to init_iterate",
        default=2,
    )

    return args_replace(parser.parse_args(args_remaining))


def main(args):
    """set up files needed to run NK solver for test_problem"""

    config = read_cfg_file(args)
    solverinfo = config["solverinfo"]

    logging_format = "%(asctime)s:%(process)s:%(filename)s:%(funcName)s:%(message)s"
    logging.basicConfig(
        stream=sys.stdout, format=logging_format, level=solverinfo["logging_level"]
    )
    logger = logging.getLogger(__name__)

    logger.info('args.cfg_fname="%s"', args.cfg_fname)

    # generate invoker script
    args.model_name = "test_problem"
    gen_invoker_script.main(args)

    # generate depth axis from args
    defn_dict = depth_defn_dict(trap_unknown=False, **(args.__dict__))
    depth = SpatialAxis(defn_dict=defn_dict)

    modelinfo = config["modelinfo"]

    caller = "src.test_problem.setup_solver.main"

    # write depth axis
    grid_weight_fname = modelinfo["grid_weight_fname"]
    logger.info('grid_weight_fname="%s"', grid_weight_fname)
    mkdir_exist_okay(os.path.dirname(grid_weight_fname))
    depth.dump(grid_weight_fname, caller)

    # confirm that model configuration works with generated file
    # ModelState relies on model being configured
    ModelConfig(modelinfo)

    # generate initial condition
    init_iterate = ModelState("gen_init_iterate")

    # perform fixed point iteration(s) on init_iterate
    if args.fp_cnt > 0:
        workdir = config["solverinfo"]["workdir"]
        gen_init_iterate_workdir = os.path.join(workdir, "gen_init_iterate")
        mkdir_exist_okay(gen_init_iterate_workdir)

        for fp_iter in range(args.fp_cnt):
            logger.info("fp_iter=%d", fp_iter)
            init_iterate.dump(
                os.path.join(
                    gen_init_iterate_workdir, "init_iterate_%02d.nc" % fp_iter
                ),
                caller,
            )
            init_iterate_fcn = init_iterate.comp_fcn(
                os.path.join(gen_init_iterate_workdir, "fcn_%02d.nc" % fp_iter),
                None,
                os.path.join(gen_init_iterate_workdir, "hist_%02d.nc" % fp_iter),
            )
            init_iterate += init_iterate_fcn
            init_iterate.copy_shadow_tracers_to_real_tracers()

    # write generated init_iterate to where solver expects it to be
    init_iterate_fname = modelinfo["init_iterate_fname"]
    logger.info('init_iterate_fname="%s"', init_iterate_fname)
    mkdir_exist_okay(os.path.dirname(init_iterate_fname))
    init_iterate.dump(init_iterate_fname, caller)


if __name__ == "__main__":
    main(parse_args(sys.argv[1:],))
