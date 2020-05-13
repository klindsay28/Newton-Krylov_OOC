"""general purpose utility functions"""

import configparser
import errno
import logging
import os
import subprocess

import git


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


def read_cfg_file(cfg_fname):
    """
    read cfg_fname
    set defaults common to all occurrances
    """
    defaults = os.environ
    defaults["repo_root"] = git.Repo(search_parent_directories=True).working_dir
    config = configparser.ConfigParser(defaults)
    config.read_file(open(cfg_fname))
    return config


def ann_files_to_mean_file(fname_in0, cnt, fname_out):
    """
    average cnt files of annual means, starting with fname_in0
    the mean is written to fname_out

    assumes that year is specified in fname_in0 as YYYY
    at the end of fname_in0s root
    """

    cmd = [
        "ncra",
        "-O",
        "-n",
        "%d,4" % cnt,
        "-o",
        fname_out,
        fname_in0,
    ]
    logger = logging.getLogger(__name__)
    logger.debug('cmd = "%s"', " ".join(cmd))
    subprocess.run(cmd, check=True)


def mon_files_to_mean_file(fname_in0, yr_cnt, fname_out):
    """
    average yr_cnt years of files of monthly means, starting with fname_in0
    the mean is written to fname_out

    assumes that year and month are specified in fname_in0 as YYYY-MM
    at the end of fname_in0s root

    generates annual means in the directory containing fname_out
    and calls ann_files_to_mean on the annual means
    """

    (fname_in0_root, fname_in0_ext) = os.path.splitext(fname_in0)
    fname_in0_nodate = fname_in0_root[:-7]
    yr0 = int(fname_in0_root[-7:-3])
    basename_nodate = os.path.basename(fname_in0_nodate)

    dir_out = os.path.dirname(fname_out)

    for yr in range(yr0, yr0 + yr_cnt):  # pylint: disable=C0103
        yyyy = "%04d" % yr
        yr_fname_in0 = fname_in0_nodate + yyyy + "-01" + fname_in0_ext
        yr_fname_out = os.path.join(dir_out, basename_nodate + yyyy + fname_in0_ext)
        mon_files_to_ann_file(yr_fname_in0, yr_fname_out)

    yyyy = "%04d" % yr0
    yr0_ann_fname = os.path.join(dir_out, basename_nodate + yyyy + fname_in0_ext)
    ann_files_to_mean_file(yr0_ann_fname, yr_cnt, fname_out)


def mon_files_to_ann_file(fname_in0, fname_out):
    """
    average a year of files of monthly means, starting with fname_in0
    the mean is written to fname_out

    assumes that year and month are specified in fname_in0 as YYYY-MM
    at the end of fname_in0s root
    """

    cmd = [
        "ncra",
        "-O",
        "-n",
        "12,2",
        "-w",
        "31,28,31,30,31,30,31,31,30,31,30,31",
        "-o",
        fname_out,
        fname_in0,
    ]
    logger = logging.getLogger(__name__)
    logger.debug('cmd = "%s"', " ".join(cmd))
    subprocess.run(cmd, check=True)
