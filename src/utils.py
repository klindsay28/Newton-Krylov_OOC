"""general purpose utility functions"""

from datetime import datetime
import errno
import importlib
import inspect
import logging
import os
import subprocess

from netCDF4 import Dataset
import numpy as np

################################################################################
# utilities related to python built-in types


def attr_common(metadata_dict, attr_name):
    """
    If there is attribute named attr_name that has a common value for all dict entries,
    return this common value. Return None otherwise. Note that a return value of None
    can occur if the value is None for all dict entries.
    """
    attr_list = []
    for metadata in metadata_dict.values():
        if attr_name not in metadata.get("attrs", {}):
            return None
        attr = metadata["attrs"][attr_name]
        if attr_list == []:
            attr_list = [attr]
        else:
            if attr == attr_list[0]:
                continue
            return None
    return attr_list[0]


def class_name(obj):
    """return name of class and module that it is define in"""
    return obj.__module__ + "." + type(obj).__name__


def get_subclasses(mod_name, base_class):
    """return list of subclasses of base_class from mod, excluding base_class"""
    logger = logging.getLogger(__name__)
    try:
        mod = importlib.import_module(mod_name)
    except ModuleNotFoundError:
        logger.debug("module %s not found", mod_name)
        return []
    return [
        value
        for (name, value) in inspect.getmembers(mod, inspect.isclass)
        if issubclass(value, base_class) and value is not base_class
    ]


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


################################################################################
# utilities related to generic file/path manipulations


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


################################################################################
# utilities related to netCDF file operations


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
