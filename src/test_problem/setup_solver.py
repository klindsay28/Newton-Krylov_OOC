#!/usr/bin/env python
"""set up files needed to run NK solver for test_problem"""

import argparse
import logging
import os
import sys

from test_problem.src.spatial_axis import SpatialAxis

from .. import gen_invoker_script
from ..model_config import ModelConfig
from .newton_fcn import ModelState, NewtonFcn
from ..utils import mkdir_exist_okay, read_cfg_file


def _parse_args():
    """parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="setup test_problem",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_name",
        help="name of model that solver is being applied to",
        default="test_problem",
    )
    parser.add_argument(
        "--cfg_fname",
        help="name of configuration file",
        default="models/{model_name}/newton_krylov.cfg",
    )
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
        "--edge_end", type=float, help="end of edges", default=675.0,
    )
    parser.add_argument(
        "--delta_start", type=float, help="thickness of first layer", default=10.0,
    )
    parser.add_argument(
        "--fp_cnt",
        type=int,
        help="number of fixed point iterations to apply to ic",
        default=2,
    )

    args = parser.parse_args()

    # replace {model_name} with specified model
    args.cfg_fname = args.cfg_fname.replace("{model_name}", args.model_name)

    return args


def main(args):
    """set up files needed to run NK solver for test_problem"""

    config = read_cfg_file(args.cfg_fname)
    solverinfo = config["solverinfo"]

    logging_format = "%(asctime)s:%(process)s:%(filename)s:%(funcName)s:%(message)s"
    logging.basicConfig(
        stream=sys.stdout, format=logging_format, level=solverinfo["logging_level"]
    )
    logger = logging.getLogger(__name__)

    logger.info('args.model_name="%s"', args.model_name)
    logger.info('args.cfg_fname="%s"', args.cfg_fname)

    # generate depth axis
    defn_dict = {
        "units": args.units,
        "nlevs": args.nlevs,
        "edge_start": args.edge_start,
        "edge_end": args.edge_end,
        "delta_start": args.delta_start,
    }
    depth = SpatialAxis(axisname=args.axisname, defn_dict=defn_dict)

    modelinfo = config["modelinfo"]

    # write depth axis
    grid_weight_fname = modelinfo["grid_weight_fname"]
    logger.info('grid_weight_fname="%s"', grid_weight_fname)
    mkdir_exist_okay(os.path.dirname(grid_weight_fname))
    depth.dump(grid_weight_fname)

    # confirm that model configuration works with generated file
    # ModelState relies on model being configured
    ModelConfig(modelinfo)

    # generate initial condition
    ic = ModelState(vals_fname="gen_ic")

    # perform fixed point iteration(s) on ic
    if args.fp_cnt > 0:
        workdir = config["solverinfo"]["workdir"]
        gen_ic_workdir = os.path.join(workdir, "gen_ic")
        mkdir_exist_okay(gen_ic_workdir)

        newton_fcn = NewtonFcn()
        for fp_iter in range(args.fp_cnt):
            logger.info("fp_iter=%d", fp_iter)
            ic.dump(os.path.join(gen_ic_workdir, "ic_%02d.nc" % fp_iter))
            ic_fcn = newton_fcn.comp_fcn(
                ic,
                os.path.join(gen_ic_workdir, "fcn_%02d.nc" % fp_iter),
                None,
                os.path.join(gen_ic_workdir, "hist_%02d.nc" % fp_iter),
            )
            ic += ic_fcn
            ic.copy_shadow_tracers_to_real_tracers()

    # write generated ic to where solver expects it to be
    init_iterate_fname = modelinfo["init_iterate_fname"]
    logger.info('init_iterate_fname="%s"', init_iterate_fname)
    mkdir_exist_okay(os.path.dirname(init_iterate_fname))
    ic.dump(init_iterate_fname)

    # generate invoker script
    gen_invoker_script.main(args)


################################################################################

if __name__ == "__main__":
    main(_parse_args())
