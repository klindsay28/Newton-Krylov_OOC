#!/usr/bin/env python
"""generate script for invoking nk_driver.py"""

import argparse
import logging
import os
import stat
import sys

from .share import args_replace, common_args, read_cfg_file, cfg_override_args
from .utils import mkdir_exist_okay


def gen_invoker_script(args, modelinfo, repo_root):
    """
    generate script for invoking nk_driver.py with optional arguments
    """

    invoker_script_fname = modelinfo["invoker_script_fname"]
    mkdir_exist_okay(os.path.dirname(invoker_script_fname))

    logger = logging.getLogger(__name__)
    logger.info("generating %s", invoker_script_fname)

    with open(invoker_script_fname, mode="w") as fptr:
        fptr.write("#!/bin/bash\n")
        fptr.write("cd %s\n" % repo_root)
        fptr.write("source scripts/newton_krylov_env_cmds\n")
        if "mpi_cmd_env_cmds_fname" in modelinfo:
            if modelinfo["mpi_cmd_env_cmds_fname"] is not None:
                fptr.write("source %s\n" % modelinfo["mpi_cmd_env_cmds_fname"])

        # construct invocation command
        line = 'python -m src.nk_driver --cfg_fname "%s" ' % modelinfo["cfg_fname"]
        if "model_name" in args:
            line = line + '--model_name "%s" ' % args.model_name
        for argname, metadata in cfg_override_args.items():
            # skip conditional overrides that were not added
            if argname not in args:
                continue
            if "action" not in metadata:
                if getattr(args, argname) is not None:
                    line = line + '--%s "%s" ' % (argname, getattr(args, argname))
            elif metadata["action"] == "store_true":
                if getattr(args, argname):
                    line = line + "--%s " % argname
            else:
                msg = "action = %s not implemented" % metadata["action"]
                raise NotImplementedError(msg)
        line = line + '"$@"\n'
        fptr.write(line)

    # ensure script is executable by the user, while preserving other permissions
    fstat = os.stat(invoker_script_fname)
    os.chmod(invoker_script_fname, fstat.st_mode | stat.S_IXUSR)


def parse_args(args_list_in=None):
    """parse command line arguments"""

    # process --model_name so that it can be passed to common_args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        help="name of model that solver is being applied to",
        default="test_problem",
    )

    args_list = [] if args_list_in is None else args_list_in
    args, args_remaining = parser.parse_known_args(args_list)

    parser = common_args("generate script for invoking nk_driver.py", args.model_name)

    return args_replace(parser.parse_args(args_remaining))


def main(args):
    """driver for Newton-Krylov solver"""

    config = read_cfg_file(args)

    # store cfg_fname in modelinfo, to follow what is done in other scripts
    config["modelinfo"]["cfg_fname"] = args.cfg_fname

    gen_invoker_script(
        args, config["modelinfo"], config["DEFAULT"]["repo_root"],
    )


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
