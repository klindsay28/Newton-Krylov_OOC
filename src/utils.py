"""general purpose utility functions"""

from datetime import datetime
import errno
import logging
import os
import subprocess

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


def create_dimension_exist_okay(fptr, dimname, dimlen):
    """
    Create a dimension in a netCDF4 file
    It is okay if it already exists, if the existing dimlen matches dimlen.
    Return dimension object
    """
    try:
        fptr.createDimension(dimname, dimlen)
    except RuntimeError as msg:
        if str(msg) != "NetCDF: String match to name in use":
            raise
        if fptr.dimensions[dimname].size != dimlen:
            raise
    return fptr.dimensions[dimname]


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


def fmt_vals(var, fmt):
    """apply format substitutions in fmt recursively to vals in var"""
    if isinstance(var, str):
        return var.format(**fmt)
    if isinstance(var, list):
        return [fmt_vals(item, fmt) for item in var]
    if isinstance(var, tuple):
        return tuple(fmt_vals(item, fmt) for item in var)
    if isinstance(var, set):
        return {fmt_vals(item, fmt) for item in var}
    if isinstance(var, dict):
        return {fmt_vals(key, fmt): fmt_vals(val, fmt) for key, val in var.items()}
    return var
