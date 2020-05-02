#!/usr/bin/env python
"""generate script for invoking nk_driver.py"""

import argparse
import configparser
import errno
import os
import stat


def mkdir_exist_okay(path):
    """
    Create a directory named path.
    It is okay if it already exists.
    """
    try:
        os.mkdir(path)
    except OSError as err:
        if err.errno == errno.EEXIST:
            pass
        else:
            raise


def invoker_script_fname(workdir):
    """
    full path of script for invoking nk_driver.py
    """
    return os.path.join(workdir, "nk_driver.sh")


def gen_invoker_script(workdir, toplevel_dir, modelinfo):
    """
    generate script for invoking nk_driver.py with optional arguments
    return the name of the generated script
    """

    mkdir_exist_okay(workdir)
    script_fname = invoker_script_fname(workdir)
    print("generating %s" % script_fname)

    with open(script_fname, mode="w") as fptr:
        fptr.write("#!/bin/bash\n")
        fptr.write("source %s\n" % modelinfo["newton_krylov_env_cmds_fname"])
        fptr.write("if [ -z ${PYTHONPATH+x} ]; then\n")
        fptr.write("    export PYTHONPATH=models\n")
        fptr.write("else\n")
        fptr.write("    export PYTHONPATH=models:$PYTHONPATH\n")
        fptr.write("fi\n")
        fptr.write("cd %s\n" % toplevel_dir)
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

    config = configparser.ConfigParser(os.environ)
    config.read_file(open(args.cfg_fname))

    # store cfg_fname in modelinfo, to follow what is done in other scripts
    config["modelinfo"]["cfg_fname"] = args.cfg_fname

    gen_invoker_script(
        config["solverinfo"]["workdir"],
        config["DEFAULT"]["toplevel_dir"],
        config["modelinfo"],
    )


if __name__ == "__main__":
    main(parse_args())
