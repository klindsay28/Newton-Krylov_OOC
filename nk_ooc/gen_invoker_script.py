#!/usr/bin/env python
"""generate script for invoking nk_driver.py"""

import logging
import os
import stat
import sys

from .share import (
    args_replace,
    cfg_override_args,
    common_args,
    read_cfg_files,
    repro_fname,
)
from .utils import mkdir_exist_okay


def gen_invoker_script(args, modelinfo, repo_root):
    """
    generate script for invoking nk_driver.py with optional arguments
    """

    invoker_script_fname = modelinfo["invoker_script_fname"]
    mkdir_exist_okay(os.path.dirname(invoker_script_fname))

    logger = logging.getLogger(__name__)
    logger.info("generating %s", repro_fname(modelinfo, invoker_script_fname))

    with open(invoker_script_fname, mode="w") as fptr:
        fptr.write("#!/bin/bash\n")
        fptr.write(f"cd {repo_root}\n")
        fptr.write("source scripts/newton_krylov_env_cmds\n")
        if getattr(args, "deprecation_warning_to_error", False):
            fptr.write("export PYTHONWARNINGS=error::DeprecationWarning\n")
        mpi_cmd_env_cmds_fname = modelinfo.get("mpi_cmd_env_cmds_fname", None)
        if mpi_cmd_env_cmds_fname is not None:
            fptr.write(f"source {mpi_cmd_env_cmds_fname}\n")

        # construct invocation command
        line = f'python -m nk_ooc.nk_driver --cfg_fnames "{args.cfg_fnames}" '
        if "model_name" in args:
            line = f'{line}--model_name "{args.model_name}" '
        for argname, metadata in cfg_override_args.items():
            # skip conditional overrides that were not added
            if argname not in args:
                continue
            if "action" not in metadata:
                if getattr(args, argname) is not None:
                    line = f'{line}--{argname} "{getattr(args, argname)}" '
            elif metadata["action"] == "store_true":
                if getattr(args, argname):
                    line = f"{line}--{argname} "
            else:
                msg = f'action = {metadata["action"]} not implemented'
                raise NotImplementedError(msg)
        line = f'{line}"$@"\n'
        fptr.write(line)

    # ensure script is executable by the user, while preserving other permissions
    fstat = os.stat(invoker_script_fname)
    os.chmod(invoker_script_fname, fstat.st_mode | stat.S_IXUSR)


def parse_args(args_list_in=None):
    """parse command line arguments"""

    args_list = [] if args_list_in is None else args_list_in
    parser, args_remaining = common_args(
        "generate script for invoking nk_driver.py", "test_problem", args_list
    )

    return args_replace(parser.parse_args(args_remaining))


def main(args):
    """driver for Newton-Krylov solver"""

    config = read_cfg_files(args)

    gen_invoker_script(args, config["modelinfo"], config["DEFAULT"]["repo_root"])


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
