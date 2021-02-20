#!/usr/bin/env python
"""driver for Newton-Krylov solver"""

import logging
import os
import sys

from .model_config import ModelConfig, get_modelinfo
from .model_state_base import ModelStateBase
from .newton_solver import NewtonSolver
from .share import args_replace, common_args, logging_config, read_cfg_files
from .utils import get_subclasses


def parse_args(args_list_in=None):
    """parse command line arguments"""

    args_list = [] if args_list_in is None else args_list_in
    parser, args_remaining = common_args(
        "invoke Newton-Krylov solver", "test_problem", args_list
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

    return args_replace(parser.parse_args(args_remaining))


def main(args):
    """driver for Newton-Krylov solver"""

    config = read_cfg_files(args)
    solverinfo = config["solverinfo"]

    logging_config(args, solverinfo, filemode="a")
    logger = logging.getLogger(__name__)

    if os.path.exists("KILL"):
        logger.warning("KILL file detected, exiting")
        raise SystemExit

    lvl = logging.DEBUG if args.resume else logging.INFO
    ModelConfig(config["modelinfo"], lvl)

    model_state_class = get_model_state_class()
    logger.log(
        lvl,
        "using class %s from %s for model state",
        model_state_class.__name__,
        model_state_class.__module__,
    )

    newton_solver = NewtonSolver(
        model_state_class, solverinfo=solverinfo, resume=args.resume, rewind=args.rewind
    )

    while True:
        if newton_solver.converged_flat().all():
            logger.info("convergence criterion satisfied")
            newton_solver.log()
            break
        newton_solver.step()


def get_model_state_class():
    """return tracer module state class for tracer_module_name"""

    model_state_class = ModelStateBase

    # look for model specific derived class
    mod_name = ".".join(["src", get_modelinfo("model_name"), "model_state"])
    subclasses = get_subclasses(mod_name, model_state_class)
    if len(subclasses) > 0:
        model_state_class = subclasses[0]

    return model_state_class


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
