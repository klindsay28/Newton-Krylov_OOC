#!/usr/bin/env python
"""generate script for invoking nk_driver.py"""

import argparse
import configparser
import os
import stat


def gen_nk_driver_invoker_script(modelinfo):
    """
    generate script for invoking nk_driver.py with optional arguments
    return the name of the generated script
    """

    cwd = os.path.dirname(os.path.realpath(__file__))
    script_fname = os.path.join(cwd, "generated_scripts", "nk_driver.sh")

    with open(script_fname, mode="w") as fptr:
        fptr.write("#!/bin/bash\n")
        fptr.write("cd %s\n" % cwd)
        fptr.write("source %s\n" % modelinfo["newton_krylov_env_cmds_fname"])
        fptr.write('./nk_driver.py --cfg_fname %s "$@"\n' % modelinfo["cfg_fname"])

    # ensure script_fname is executable by the user, while preserving other permissions
    fstat = os.stat(script_fname)
    os.chmod(script_fname, fstat.st_mode | stat.S_IXUSR)

    return script_fname


def parse_args():
    """parse command line arguments"""

    parser = argparse.ArgumentParser(description="Newton's method example")

    parser.add_argument(
        "--cfg_fname", help="name of configuration file", default="newton_krylov.cfg"
    )

    return parser.parse_args()


def main(args):
    """driver for Newton-Krylov solver"""

    config = configparser.ConfigParser()
    config.read(args.cfg_fname)

    # store cfg_fname in modelinfo, to follow what is done in other scripts
    config["modelinfo"]["cfg_fname"] = args.cfg_fname

    gen_nk_driver_invoker_script(config["modelinfo"])


if __name__ == "__main__":
    main(parse_args())
