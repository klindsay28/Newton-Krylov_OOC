"""general purpose utility functions"""

import argparse
import configparser
from datetime import datetime
import errno
import logging
import os
import subprocess

import git
from netCDF4 import Dataset


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


def parse_args_common(description, model_name="test_problem"):
    """instantiate and return a parser, using common options"""
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_name",
        help="name of model that solver is being applied to",
        default=model_name,
    )
    parser.add_argument(
        "--cfg_fname",
        help="name of configuration file",
        default="models/{model_name}/newton_krylov.cfg",
    )
    parser.add_argument(
        "--workdir", help="override workdir from cfg file", default=None,
    )
    parser.add_argument(
        "--tracer_module_names",
        help="override tracer_module_names from cfg file",
        default=None,
    )
    if model_name == "test_problem":
        parser.add_argument(
            "--persist", help="override reinvoke from cfg file", action="store_true",
        )
    return parser


def read_cfg_file(args):
    """
    read cfg file
    set defaults common to all occurrances
    """
    cfg_fname = args.cfg_fname

    defaults = {key: os.environ[key] for key in ["HOME", "USER"]}
    defaults["repo_root"] = git.Repo(search_parent_directories=True).working_dir
    config = configparser.ConfigParser(defaults, allow_no_value=True)
    config.read_file(open(cfg_fname))

    # verify that only names in no_value_allowed have no value
    # no_value_allowed is allowed to have no value or not be present
    if "no_value_allowed" in config["DEFAULT"]:
        no_value_allowed = config["DEFAULT"]["no_value_allowed"]
    else:
        no_value_allowed = None
    nva_list = [] if no_value_allowed is None else no_value_allowed.split(",")
    nva_list.append("no_value_allowed")
    for section in config.sections():
        for name in config[section]:
            if config[section][name] is None and name not in nva_list:
                msg = "%s not allowed to be empty in cfg file %s" % (name, cfg_fname)
                raise ValueError(msg)

    if args.workdir is not None:
        config["DEFAULT"]["workdir"] = args.workdir

    if args.tracer_module_names is not None:
        config["modelinfo"]["tracer_module_names"] = args.tracer_module_names

    if "persist" in args and args.persist:
        config["modelinfo"]["reinvoke"] = "False"

    cfg_out_fname = config["solverinfo"]["cfg_out_fname"]
    if cfg_out_fname is not None:
        mkdir_exist_okay(os.path.dirname(cfg_out_fname))
        with open(cfg_out_fname, "w") as fptr:
            config.write(fptr)

    return config


def ann_files_to_mean_file(dir_in, fname_fmt, year0, cnt, fname_out, caller):
    """
    average cnt number of files of annual means

    fname_fmt is a string format specifying the filenames,
    relative to dir_in, of the annual means, with year as a field
    e.g., fname_fmt = "casename.pop.h.{year:04d}.nc"

    the mean is written to fname_out
    """

    cmd = [
        "ncra",
        "-O",
        "-o",
        fname_out,
        "-p",
        dir_in,
    ]

    fnames = [fname_fmt.format(year=year0 + inc) for inc in range(cnt)]

    cmd.extend(fnames)

    logger = logging.getLogger(__name__)
    logger.debug('cmd = "%s"', " ".join(cmd))

    subprocess.run(cmd, check=True)

    with Dataset(os.path.join(dir_in, fname_out), mode="a") as fptr:
        datestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        name = "src.utils.ann_files_to_mean_file"
        msg = datestamp + ": ncra called from " + name + " called from " + caller
        msg = msg + "\n" + getattr(fptr, "history")
        setattr(fptr, "history", msg)


def mon_files_to_mean_file(dir_in, fname_fmt, year0, month0, cnt, fname_out, caller):
    """
    average cnt number of files of monthly means

    fname_fmt is a string format specifying the filenames,
    relative to dir_in, of the monthly means, with year and month as fields
    e.g., fname_fmt = "casename.pop.h.{year:04d}-{month:02d}.nc"

    the mean is written to fname_out

    it is okay for month0 to not be 1
    cnt does not need to be a multiple of 12
    noleap days in month weights are applied in the averaging
    """

    # construct averaging weights
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    days_all = [days_in_month[(month0 - 1 + inc) % 12] for inc in range(cnt)]
    days_all_str = ",".join(["%d" % wval for wval in days_all])

    # generate filenames of input monthly means
    yr_vals = [year0 + (month0 - 1 + inc) // 12 for inc in range(cnt)]
    month_vals = [(month0 - 1 + inc) % 12 + 1 for inc in range(cnt)]
    fnames = [
        fname_fmt.format(year=yr_vals[inc], month=month_vals[inc]) for inc in range(cnt)
    ]

    cmd = [
        "ncra",
        "-O",
        "-w",
        days_all_str,
        "-o",
        fname_out,
        "-p",
        dir_in,
    ]
    cmd.extend(fnames)

    logger = logging.getLogger(__name__)
    logger.debug('cmd = "%s"', " ".join(cmd))

    subprocess.run(cmd, check=True)

    with Dataset(os.path.join(dir_in, fname_out), mode="a") as fptr:
        datestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        name = "src.utils.mon_files_to_mean_file"
        msg = datestamp + ": ncra called from " + name + " called from " + caller
        msg = msg + "\n" + getattr(fptr, "history")
        setattr(fptr, "history", msg)
