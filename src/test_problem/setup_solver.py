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


def _parse_args():
    """parse command line arguments"""
    parser = common_args("setup test_problem", "test_problem")
    parser.add_argument(
        "--axisname", help="axis name", default="depth",
    )
    parser.add_argument(
        "--units", help="axis units", default="m",
    )
    parser.add_argument(
        "--nlevs", type=int, help="number of layers", default=30,
    )
    parser.add_argument(
        "--edge_start", type=float, help="start of edges", default=0.0,
    )
    parser.add_argument(
        "--edge_end", type=float, help="end of edges", default=900.0,
    )
    parser.add_argument(
        "--delta_ratio_max",
        type=float,
        help="maximum ratio of layer thicknesses",
        default=5.0,
    )
    parser.add_argument(
        "--fp_cnt",
        type=int,
        help="number of fixed point iterations to apply to init_iterate",
        default=2,
    )

    return args_replace(parser.parse_args(), model_name="test_problem")


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

    # generate depth axis
    defn_dict = {
        "units": args.units,
        "nlevs": args.nlevs,
        "edge_start": args.edge_start,
        "edge_end": args.edge_end,
        "delta_ratio_max": args.delta_ratio_max,
    }
    depth = SpatialAxis(axisname=args.axisname, defn_dict=defn_dict)

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
    main(_parse_args())
