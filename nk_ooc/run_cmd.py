#!/usr/bin/env python
"""standalone driver for model_state methods"""

import logging
import os
import sys

from .model_config import ModelConfig
from .share import (
    args_replace,
    common_args,
    get_model_state_class,
    logging_config,
    read_cfg_files,
)


def parse_args(args_list_in=None):
    """parse command line arguments"""

    args_list = [] if args_list_in is None else args_list_in
    parser, args_remaining = common_args(
        "standalone driver for model_state methods", "test_problem", args_list
    )
    parser.add_argument(
        "cmd",
        choices=["comp_fcn", "gen_precond_jacobian", "apply_precond_jacobian"],
        help="command to run",
    )
    parser.add_argument(
        "--fname_dir",
        help="directory that relative fname arguments are relative to",
        default=".",
    )
    parser.add_argument("--hist_fname", help="name of history file", default=None)
    parser.add_argument("--precond_fname", help="name of precond file", default=None)
    parser.add_argument("--in_fname", help="name of file with input")
    parser.add_argument("--res_fname", help="name of file for result")

    return args_replace(parser.parse_args(args_remaining))


def _resolve_fname(fname_dir, fname):
    """prepend fname_dir to fname, if fname is a relative path"""
    if fname is None or os.path.isabs(fname):
        return fname
    return os.path.join(fname_dir, fname)


def main(args):
    """standalone driver for model_state methods"""

    config = read_cfg_files(args)
    solverinfo = config["solverinfo"]

    logging_config(solverinfo, filemode="a")
    logger = logging.getLogger(__name__)

    logger.info('args.cmd="%s"', args.cmd)

    lvl = logging.INFO

    model_state_class = get_model_state_class(config["DEFAULT"]["model_name"], lvl)

    # configure model and attach model_config_obj to model_state_class
    model_state_class.model_config_obj = ModelConfig(config["modelinfo"], lvl)

    ms_in = model_state_class(_resolve_fname(args.fname_dir, args.in_fname))
    if args.cmd == "comp_fcn":
        ms_in.log("state_in")
        ms_in.comp_fcn(
            _resolve_fname(args.fname_dir, args.res_fname),
            solver_state=None,
            hist_fname=_resolve_fname(args.fname_dir, args.hist_fname),
        ).log("fcn")
    elif args.cmd == "gen_precond_jacobian":
        ms_in.gen_precond_jacobian(
            _resolve_fname(args.fname_dir, args.hist_fname),
            _resolve_fname(args.fname_dir, args.precond_fname),
            solver_state=None,
        )
    elif args.cmd == "apply_precond_jacobian":
        ms_in.log("state_in")
        ms_in.apply_precond_jacobian(
            _resolve_fname(args.fname_dir, args.precond_fname),
            _resolve_fname(args.fname_dir, args.res_fname),
            solver_state=None,
        ).log("precond_res")
    else:
        msg = "unknown cmd=%s" % args.cmd
        raise ValueError(msg)

    logger.info("done")


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
