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

    it is okay for MM in fname_in0 to not be 01
    """

    dirname_in0 = os.path.dirname(fname_in0)
    basename_in0 = os.path.basename(fname_in0)
    (basename_in0_root, basename_in0_ext) = os.path.splitext(basename_in0)
    basename_in0_nodate = basename_in0_root[:-7]
    yr0 = int(basename_in0_root[-7:-3])
    mon0 = int(basename_in0_root[-2:])

    # construct averaging weights
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    days_all = [days_in_month[(mon0 - 1 + inc) % 12] for inc in range(12 * yr_cnt)]
    days_all_str = ",".join(["%d" % wval for wval in days_all])

    # generate filenames of input monthly means
    yr_vals = [yr0 + (mon0 - 1 + inc) // 12 for inc in range(12 * yr_cnt)]
    mon_vals = [(mon0 - 1 + inc) % 12 + 1 for inc in range(12 * yr_cnt)]
    fname_fmt = basename_in0_nodate + "%04d" + "-" + "%02d" + basename_in0_ext
    fnames = [fname_fmt % (yr_vals[inc], mon_vals[inc]) for inc in range(12 * yr_cnt)]

    cmd = [
        "ncra",
        "-O",
        "-w",
        days_all_str,
        "-o",
        fname_out,
        "-p",
        dirname_in0,
    ]
    cmd.extend(fnames)

    logger = logging.getLogger(__name__)
    logger.debug('cmd = "%s"', " ".join(cmd))

    subprocess.run(cmd, check=True)
