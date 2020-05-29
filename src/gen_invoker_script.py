#!/usr/bin/env python
"""generate script for invoking nk_driver.py"""

import logging
import os
import stat

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
        fptr.write("source src/newton_krylov_env_cmds\n")
        if "mpi_cmd_env_cmds_fname" in modelinfo:
            if modelinfo["mpi_cmd_env_cmds_fname"] is not None:
                fptr.write("source %s\n" % modelinfo["mpi_cmd_env_cmds_fname"])
        fptr.write("if [ -z ${PYTHONPATH+x} ]; then\n")
        fptr.write("    export PYTHONPATH=models\n")
        fptr.write("else\n")
        fptr.write("    export PYTHONPATH=models:$PYTHONPATH\n")
        fptr.write("fi\n")

        # construct invocation command
        line = './nk_driver.py --cfg_fname "%s" ' % modelinfo["cfg_fname"]
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


def parse_args():
    """parse command line arguments"""

    parser = common_args("generate script for invoking nk_driver.py")

    parser.add_argument(
        "--model_name",
        help="name of model that solver is being applied to",
        default="test_problem",
    )

    return args_replace(parser.parse_args())


def main(args):
    """driver for Newton-Krylov solver"""

    config = read_cfg_file(args)

    # store cfg_fname in modelinfo, to follow what is done in other scripts
    config["modelinfo"]["cfg_fname"] = args.cfg_fname

    gen_invoker_script(
        args, config["modelinfo"], config["DEFAULT"]["repo_root"],
    )


if __name__ == "__main__":
    main(parse_args())
