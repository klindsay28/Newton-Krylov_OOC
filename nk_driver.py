#!/usr/bin/env python
"""driver for Newton-Krylov solver"""

import argparse
import importlib
import logging
import os
import sys

from src.model_config import ModelConfig
from src.newton_solver import NewtonSolver
from src.share import args_replace, common_args, read_cfg_file


def parse_args():
    """parse command line arguments"""

    # process --model_name so that it can be passed to common_args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        help="name of model that solver is being applied to",
        default="test_problem",
    )
    args, args_remaining = parser.parse_known_args()

    parser = common_args("invoke Newton-Krylov solver", args.model_name)

    parser.add_argument(
        "--resume",
        help="resume Newton's method from solver's saved state",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--rewind",
        help="rewind last step to recover from error",
        action="store_true",
        default=False,
    )

    return args_replace(parser.parse_args(args_remaining))


def main(args):
    """driver for Newton-Krylov solver"""

    config = read_cfg_file(args)
    solverinfo = config["solverinfo"]

    logging_format = "%(asctime)s:%(process)s:%(filename)s:%(funcName)s:%(message)s"
    logging.basicConfig(
        filename=solverinfo["logging_fname"],
        filemode="a" if args.resume else "w",
        format=logging_format,
        level=solverinfo["logging_level"],
    )
    sys.stdout = open(solverinfo["logging_fname"], "a")
    sys.stderr = open(solverinfo["logging_fname"], "a")
    logger = logging.getLogger(__name__)

    if os.path.exists("KILL"):
        logger.warning("KILL file detected, exiting")
        raise SystemExit

    # store cfg_fname in modelinfo, to ease access to its values elsewhere
    config["modelinfo"]["cfg_fname"] = args.cfg_fname

    ModelConfig(config["modelinfo"], logging.DEBUG if args.resume else logging.INFO)

    # import module with ModelState class
    logger.debug('model_state_modname="%s"', config["modelinfo"]["model_state_modname"])
    model_state_mod = importlib.import_module(
        config["modelinfo"]["model_state_modname"]
    )

    newton_solver = NewtonSolver(
        model_state_mod.ModelState,
        solverinfo=solverinfo,
        resume=args.resume,
        rewind=args.rewind,
    )

    while True:
        if newton_solver.converged_flat().all():
            logger.info("convergence criterion satisfied")
            newton_solver.log(append_to_stats_file=True)
            break
        newton_solver.step()


if __name__ == "__main__":
    main(parse_args())
