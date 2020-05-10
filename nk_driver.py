#!/usr/bin/env python
"""driver for Newton-Krylov solver"""

import argparse
import configparser
import importlib
import logging
import os
import sys

import git

from src.model_config import ModelConfig
from src.newton_solver import NewtonSolver
from src.gen_invoker_script import invoker_script_fname


def parse_args():
    """parse command line arguments"""

    parser = argparse.ArgumentParser(
        description="Newton's method example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        help="name of model that solver is being applied to",
        default="test_problem",
    )
    parser.add_argument(
        "--cfg_fname",
        help="name of configuration file",
        default="models/{model}/newton_krylov.cfg",
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

    parsed_args = parser.parse_args()

    # replace {model} with specified model
    parsed_args.cfg_fname = parsed_args.cfg_fname.replace("{model}", parsed_args.model)

    return parsed_args


def main(args):
    """driver for Newton-Krylov solver"""

    defaults = os.environ
    defaults["repo_root"] = git.Repo(search_parent_directories=True).working_dir
    config = configparser.ConfigParser(defaults)
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
            newton_solver.log(append_to_stats_file=True)
            break
        newton_solver.step()


if __name__ == "__main__":
    main(parse_args())
