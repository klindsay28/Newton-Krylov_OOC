#!/usr/bin/env python
"""generate script for invoking nk_driver.py"""

import argparse
import logging
import os
import stat

from .utils import mkdir_exist_okay, read_cfg_file


def gen_invoker_script(modelinfo, repo_root):
    """
    generate script for invoking nk_driver.py with optional arguments
    """

    invoker_script_fname = modelinfo["invoker_script_fname"]
    mkdir_exist_okay(os.path.dirname(invoker_script_fname))

    logger = logging.getLogger(__name__)
    logger.info("generating %s", invoker_script_fname)

    with open(invoker_script_fname, mode="w") as fptr:
        fptr.write("#!/bin/bash\n")
        fptr.write("source %s\n" % modelinfo["newton_krylov_env_cmds_fname"])
        fptr.write("if [ -z ${PYTHONPATH+x} ]; then\n")
        fptr.write("    export PYTHONPATH=models\n")
        fptr.write("else\n")
        fptr.write("    export PYTHONPATH=models:$PYTHONPATH\n")
        fptr.write("fi\n")
        fptr.write("cd %s\n" % repo_root)
        fptr.write('./nk_driver.py --cfg_fname %s "$@"\n' % modelinfo["cfg_fname"])

    # ensure script is executable by the user, while preserving other permissions
    fstat = os.stat(invoker_script_fname)
    os.chmod(invoker_script_fname, fstat.st_mode | stat.S_IXUSR)


def parse_args():
    """parse command line arguments"""

    parser = argparse.ArgumentParser(
        description="generate script for invoking nk_driver.py",
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

    args = parser.parse_args()

    # replace {model_name} with specified model
    args.cfg_fname = args.cfg_fname.replace("{model_name}", args.model_name)

    return args


def main(args):
    """driver for Newton-Krylov solver"""

    config = read_cfg_file(args.cfg_fname)

    # store cfg_fname in modelinfo, to follow what is done in other scripts
    config["modelinfo"]["cfg_fname"] = args.cfg_fname

    gen_invoker_script(
        config["modelinfo"], config["DEFAULT"]["repo_root"],
    )


if __name__ == "__main__":
    main(parse_args())
