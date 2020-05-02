#!/usr/bin/env python
"""driver for Newton-Krylov solver"""

import argparse
import configparser
import importlib
import logging
import os
import sys

from src.model_config import ModelConfig
from src.newton_solver import NewtonSolver
from src.gen_invoker_script import invoker_script_fname


def parse_args():
    """parse command line arguments"""

    parser = argparse.ArgumentParser(description="Newton's method example")
    parser.add_argument(
        "--cfg_fname", help="name of configuration file", default="newton_krylov.cfg"
    )
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

    return parser.parse_args()


def main(args):
    """driver for Newton-Krylov solver"""

    config = configparser.ConfigParser(os.environ)
    config.read_file(open(args.cfg_fname))
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

    # store cfg_fname and invoker_script_fname in modelinfo,
    # to ease access to their values elsewhere
    config["modelinfo"]["cfg_fname"] = args.cfg_fname
    config["modelinfo"]["invoker_script_fname"] = invoker_script_fname(
        config["solverinfo"]["workdir"],
    )

    ModelConfig(config["modelinfo"], logging.DEBUG if args.resume else logging.INFO)

    # import module with NewtonFcn class
    logger.debug('newton_fcn_modname="%s"', config["modelinfo"]["newton_fcn_modname"])
    newton_fcn_mod = importlib.import_module(config["modelinfo"]["newton_fcn_modname"])

    newton_solver = NewtonSolver(
        newton_fcn_obj=newton_fcn_mod.NewtonFcn(),
        solverinfo=solverinfo,
        resume=args.resume,
        rewind=args.rewind,
    )

    while True:
        if newton_solver.converged_flat().all():
            logger.info("convergence criterion satisfied")
            newton_solver.log()
            break
        newton_solver.step()


if __name__ == "__main__":
    main(parse_args())
