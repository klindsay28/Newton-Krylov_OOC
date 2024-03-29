"""general purpose utility functions"""

import ast
import errno
import importlib
import inspect
import logging
import operator
import os
import subprocess
from datetime import datetime

import numpy as np
from netCDF4 import Dataset, default_fillvals
from pint import UnitRegistry
from scipy import interpolate

################################################################################
# utilities related to python built-in types


def attr_common(metadata_dict, attr_name):
    """
    If there is attribute named attr_name that has a common value for all dict entries,
    return this common value. Return None otherwise. Note that a return value of None
    can occur if the value is None for all dict entries.
    """
    if not isinstance(metadata_dict, dict):
        raise TypeError(f"metadata_dict must be a dict, not {type(metadata_dict)}")
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


def dict_sel(dict_obj, **kwargs):
    """
    return dict of (key, value) pairs from dict_obj where value is a dict
    that matches (sel_key, sel_value) pairs in kwargs
    """
    if not isinstance(dict_obj, dict):
        raise TypeError(f"dict_obj must be a dict, not {type(dict_obj)}")
    res = dict_obj
    for sel_key, sel_value in kwargs.items():
        res = {
            key: value
            for key, value in res.items()
            if isinstance(value, dict) and value.get(sel_key, None) == sel_value
        }
    return res


def dict_update_verify(dict_in, dict_add):
    """
    Add entries of dict_add to dict_in. If a key being added already exists, and the
    added value differs from the existing value, raise a RuntimeError. The updated
    dict_in is returned.
    """
    for key, value_add in dict_add.items():
        if key not in dict_in:
            dict_in[key] = value_add
        else:
            if isinstance(value_add, np.ndarray):
                if np.any(dict_in[key] != value_add):
                    raise RuntimeError(f"dict value mismatch for key={key}")
            elif dict_in[key] != value_add:
                raise RuntimeError(f"dict value mismatch for key={key}")
    return dict_in


def class_name(obj):
    """return name of class and module that it is define in"""
    return f"{obj.__module__}.{type(obj).__name__}"


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


def strtobool(val):
    """Convert a string representation of truth to True (1) or False (0).

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; False values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.

    copied from distutils/util.py in python 3.6 and run through black
    (code for strtobool in python 3.9 is the same)
    distutils is deprecated in python 3.10 and will be removed in 3.12
    https://www.python.org/dev/peps/pep-0632/
    """
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return 1
    if val in ("n", "no", "f", "false", "off", "0"):
        return 0
    raise ValueError("invalid truth value %r" % (val,))


################################################################################
# utilities related to arithmetic expression parsing/evaluation


def eval_expr(expr):
    """evaluate an arithmetic expression"""
    # based on https://stackoverflow.com/a/9558001/6298056
    return _eval(ast.parse(expr, mode="eval").body)


def _eval(node):
    # supported operators
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    if isinstance(node, ast.Num):  # <number>
        return node.n
    if isinstance(node, ast.Constant):  # <number>
        return node.value
    if isinstance(node, ast.BinOp):  # <left> <operator> <right>
        return operators[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp):  # <operator> <operand> e.g., -1
        return operators[type(node.op)](_eval(node.operand))
    raise TypeError(node)


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
# utilities related to unit strings


def units_str_format(units_str):
    """
    Return units string in canonical format
    parsing is very primitive
    """

    res = f"{UnitRegistry()(units_str).units:~}"
    # do some replacements
    term_repl = {"a": "years"}
    res = " ".join([term_repl.get(term, term) for term in res.split()])
    res = res.replace(" ** ", "^")
    res = res.replace(" * ", " ")
    # some reordering
    res_split = res.split(" / ")
    if len(res_split) == 3 and (res_split[1] in ["d", "s"]):
        res = " / ".join([res_split[0], res_split[2], res_split[1]])
    return res


################################################################################
# utilities related to netCDF file operations


def metadata_same(fname1, fname2):
    """Return True if metadata in fname1 and fname2 are the same."""
    logger = logging.getLogger(__name__)
    res = True
    with Dataset(fname1, mode="r") as fptr1, Dataset(fname2, mode="r") as fptr2:
        # verify that dimension names are the same
        if fptr1.dimensions.keys() != fptr2.dimensions.keys():
            logger.info("    dimension name mismatch in %s and %s", fname1, fname2)
            res = False
        # verify that common dimensions have the same lengths
        for dimname in fptr1.dimensions:
            if dimname in fptr2.dimensions:
                if len(fptr1.dimensions[dimname]) != len(fptr2.dimensions[dimname]):
                    logger.info(
                        "    %s length mismatch in %s and %s", dimname, fname1, fname2
                    )
                    res = False
        # verify that variable names are the same
        if fptr1.variables.keys() != fptr2.variables.keys():
            logger.info("    variable name mismatch in %s and %s", fname1, fname2)
            res = False
        # verify that common variables have the same dimensions and attributes
        for varname in fptr1.variables:
            if varname in fptr2.variables:
                if (
                    fptr1.variables[varname].dimensions
                    != fptr2.variables[varname].dimensions
                ):
                    logger.info(
                        "    %s dimension mismatch in %s and %s",
                        varname,
                        fname1,
                        fname2,
                    )
                    res = False
                if (
                    fptr1.variables[varname].__dict__
                    != fptr2.variables[varname].__dict__
                ):
                    logger.info(
                        "    %s attribute mismatch in %s and %s",
                        varname,
                        fname1,
                        fname2,
                    )
                    res = False
    return res


def isclose_all_vars(fname1, fname2, rtol, atol):
    """Return True if all vars in common to fname1 and fname2 are close."""
    res = True
    with Dataset(fname1, mode="r") as fptr1, Dataset(fname2, mode="r") as fptr2:
        fptr1.set_auto_mask(False)
        fptr2.set_auto_mask(False)
        for varname in fptr1.variables:
            if varname in fptr2.variables:
                var1 = fptr1.variables[varname]
                var2 = fptr2.variables[varname]
                if not _isclose_one_var(var1, var2, rtol=rtol, atol=atol):
                    res = False
    return res


def _isclose_one_var(var1, var2, rtol, atol):
    """Return True if netCDF vars var1 and var2 are close."""
    logger = logging.getLogger(__name__)

    # further comparisons do not make sense if shapes differ
    if var1.shape != var2.shape:
        logger.info(
            "    var1.shape %s != var2.shape %s for %s",
            var1.shape,
            var2.shape,
            var1.name,
        )
        return False

    res = True

    vals1 = var1[:]
    msv1 = getattr(var1, "_FillValue", None)
    vals2 = var2[:]
    msv2 = getattr(var2, "_FillValue", None)

    if ((vals1 == msv1) != (vals2 == msv2)).any():
        logger.info("    _FillValue pattern mismatch for %s", var1.name)
        res = False

    vals1 = np.where((vals1 == msv1) | (vals2 == msv2), np.nan, vals1)
    vals2 = np.where((vals1 == msv1) | (vals2 == msv2), np.nan, vals2)

    # if units are present and differ, convert vals1 to units of var2
    # this feature is not supported for udunits type time units
    if hasattr(var1, "units") and hasattr(var2, "units"):
        if "since" not in var1.units and "since" not in var2.units:
            ureg = UnitRegistry()
            if ureg(var1.units) != ureg(var2.units):
                vals1 = ureg.Quantity(vals1, var1.units).to(var2.units).magnitude
        else:
            if var1.units != var2.units:
                raise ValueError(
                    f"time-like units disagree '{var1.units}'!='{var2.units}'"
                )

    if not _isclose_one_var_core(vals1, vals2, rtol=rtol, atol=atol):
        logger.info("    %s vals not close", var1.name)
        res = False

    return res


def _isclose_one_var_core(vals1, vals2, rtol, atol):
    """core for comparing numpy arrays, and logging differences"""
    logger = logging.getLogger(__name__)
    res = np.isclose(vals1, vals2, rtol=rtol, atol=atol, equal_nan=True).all()
    if not res:
        for ind in range(vals1.size):
            val1 = vals1.reshape(-1)[ind]
            val2 = vals2.reshape(-1)[ind]
            if not np.isclose(val1, val2, rtol=rtol, atol=atol, equal_nan=True):
                atol_adj = abs(val1 - val2) - rtol * abs(val2)
                rtol_adj = (abs(val1 - val2) - atol) / abs(val2)
                logger.info(
                    "    %.10e %.10e not close, atol_adj=%e, rtol_adj=%e",
                    val1,
                    val2,
                    atol_adj,
                    rtol_adj,
                )
    return res


def extract_dimensions(fptr, names):
    """
    Return a dict of dimensions that names are defined on.
    names is a string containing a dimension or variable name, or a tuple or list of
    dimension or variable names.
    Raise a ValueError if a name from names is unknown.
    """
    if not isinstance(names, (str, tuple, list)):
        raise TypeError(f"names must be a str, tuple, or list, not {type(names)}")
    if isinstance(names, str):
        names = [names]
    res = {}
    for name in names:
        if name in fptr.dimensions:
            res[name] = len(fptr.dimensions[name])
        elif name in fptr.variables:
            res.update(extract_dimensions(fptr, fptr.variables[name].dimensions))
        else:
            raise ValueError(f"unknown name {name}")
    return res


def create_dimensions_verify(fptr, dimensions):
    """
    Create dimensions in a netCDF file. If a dimension with dimname already exists,
    and dimlen differs from the existing dimension's length, raise a RuntimeError.
    """
    if not isinstance(dimensions, dict):
        raise TypeError(f"dimensions must be a dict, not {type(dimensions)}")
    for dimname, dimlen in dimensions.items():
        try:
            fptr.createDimension(dimname, dimlen)
        except RuntimeError as msg:
            if str(msg) != "NetCDF: String match to name in use":
                raise
            if len(fptr.dimensions[dimname]) != dimlen:
                raise
        fptr.sync()


def datatype_sname(var):
    """
    return shortname of datatype of netCDF variable var
    useable in default_fillvals
    """
    datatype_replace = {"float64": "f8", "float32": "f4"}
    datatype = str(var.datatype)
    # drop leading endian specifying character if present
    if datatype[0] in [">", "<"]:
        datatype = datatype[1:]
    datatype = datatype_replace.get(datatype, datatype)
    if datatype not in default_fillvals:
        raise ValueError(
            f"unknown datatype {str(var.datatype)}->{datatype} for {var.name}"
        )
    return datatype


def create_vars(fptr, vars_metadata):
    """Create multiple netCDF variables, using metadata from vars_metadata."""
    for varname, metadata in vars_metadata.items():
        datatype = metadata.get("datatype", "f8")
        attrs = metadata.get("attrs", {})
        fill_value = attrs.get("_FillValue", None)
        var = fptr.createVariable(
            varname, datatype, metadata["dimensions"], fill_value=fill_value
        )
        attrs_nofill = {key: attrs[key] for key in attrs if key != "_FillValue"}
        var.setncatts(attrs_nofill)
        fptr.sync()


def ann_files_to_mean_file(dir_in, fname_fmt, year0, cnt, fname_out, caller):
    """
    average cnt number of files of annual means

    fname_fmt is a string format specifying the filenames,
    relative to dir_in, of the annual means, with year as a field
    e.g., fname_fmt = "casename.pop.h.{year:04}.nc"

    the mean is written to fname_out
    """

    cmd = ["ncra", "-O", "-o", fname_out, "-p", dir_in]

    fnames = [fname_fmt.format(year=year0 + inc) for inc in range(cnt)]

    cmd.extend(fnames)

    logger = logging.getLogger(__name__)
    logger.debug('cmd = "%s"', " ".join(cmd))

    subprocess.run(cmd, check=True)

    with Dataset(os.path.join(dir_in, fname_out), mode="a") as fptr:
        datestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        name = "nk_ooc.utils.ann_files_to_mean_file"
        msg = f"{datestamp}: ncra called from {name} called from {caller}"
        fptr.history = "\n".join([msg, fptr.history])


def mon_files_to_mean_file(dir_in, fname_fmt, year0, month0, cnt, fname_out, caller):
    """
    average cnt number of files of monthly means

    fname_fmt is a string format specifying the filenames,
    relative to dir_in, of the monthly means, with year and month as fields
    e.g., fname_fmt = "casename.pop.h.{year:04}-{month:02}.nc"

    the mean is written to fname_out

    it is okay for month0 to not be 1
    cnt does not need to be a multiple of 12
    noleap days in month weights are applied in the averaging
    """

    # construct averaging weights
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    days_all = [days_in_month[(month0 - 1 + inc) % 12] for inc in range(cnt)]
    days_all_str = ",".join([str(wval) for wval in days_all])

    # generate filenames of input monthly means
    yr_vals = [year0 + (month0 - 1 + inc) // 12 for inc in range(cnt)]
    month_vals = [(month0 - 1 + inc) % 12 + 1 for inc in range(cnt)]
    fnames = [
        fname_fmt.format(year=yr_vals[inc], month=month_vals[inc]) for inc in range(cnt)
    ]

    cmd = ["ncra", "-O", "-w", days_all_str, "-o", fname_out, "-p", dir_in]
    cmd.extend(fnames)

    logger = logging.getLogger(__name__)
    logger.debug('cmd = "%s"', " ".join(cmd))

    subprocess.run(cmd, check=True)

    with Dataset(os.path.join(dir_in, fname_out), mode="a") as fptr:
        datestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        name = "nk_ooc.utils.mon_files_to_mean_file"
        msg = f"{datestamp}: ncra called from {name} called from {caller}"
        fptr.history = "\n".join([msg, fptr.history])


def gen_forcing_fcn(fname, varname, additional_dims_out, scalef=1.0):
    """
    Return function for interpolating forcing field from a netCDF file.
    The returned function will interpolate along the field's 1st dimension.
    The typical use case is that this dimension is time.
    Can handle forcing data with 0, 1, or 2 additional dimensions.
    fname: name of file with forcing
    varname: name of file variable with forcing
    additional_dims_out: list of non-time axis values to interpolate data to.
    scalef: scaling factor that data is multiplied by
    """
    logger = logging.getLogger(__name__)
    logger.info("reading %s from %s", varname, fname)
    with Dataset(fname, mode="r") as fptr:
        fptr.set_auto_mask(False)
        var = fptr.variables[varname]

        # verify various assumptions of implementation
        if var.ndim not in [1, 2, 3]:
            raise ValueError(f"unexpected ndim={var.ndim}")
        if len(additional_dims_out) != var.ndim - 1:
            raise ValueError(
                f"len(additional_dims_out) = {len(additional_dims_out)} must be "
                f"{var.ndim - 1}"
            )
        dimnames = var.dimensions

        dim0_in = fptr.variables[dimnames[0]][:]
        data = scalef * var[:]

        # interpolate along additional dimensions,
        # if forcing axis differs from model axis
        for axis in range(1, var.ndim):
            dim_in = fptr.variables[dimnames[axis]][:]
            dim_out = additional_dims_out[axis - 1]
            if len(dim_in) != len(dim_out) or (dim_in != dim_out).any():
                fcn = interpolate.interp1d(
                    dim_in,
                    data,
                    axis=axis,
                    fill_value="extrapolate",
                    assume_sorted=True,
                )
                data = fcn(dim_out)

    fcn = interpolate.interp1d(
        dim0_in, data, axis=0, fill_value="extrapolate", assume_sorted=True
    )

    return fcn


################################################################################
# utilities related to numpy arrays


def min_by_region(region_cnt, region_mask, vals, out=None):
    """
    compute min of vals for each value of region_mask
    result is stored in out, if present
    out is assumed to be a 1d numpy array of shape (region_cnt,)
    """
    if out is None:
        out = np.empty(region_cnt)
    else:
        if out.shape != (region_cnt,):
            raise ValueError(f"unexpected out.shape={out.shape}")
    for region_ind in range(out.size):
        mask = region_mask == region_ind + 1
        out[region_ind] = np.amin(vals, initial=np.inf, where=mask)
    return out


def comp_scalef_lob(region_cnt, region_mask, base, increment, lob, out=None):
    """
    compute largest 0<scalef<1, by region, to ensure base + scalef * increment >= lob
    result is stored in out, if present
    out is assumed to be a 1d numpy array of shape (region_cnt,)
    """
    if out is None:
        out = np.empty(region_cnt)
    else:
        if out.shape != (region_cnt,):
            raise ValueError(f"unexpected out.shape={out.shape}")
    if lob is None or (base + increment >= lob).all():
        out[:] = 1.0
        return out
    if (base < lob).any():
        raise ValueError("base < lob")
    scalef_all = np.ones(base.shape)
    np.divide(lob - base, increment, out=scalef_all, where=base + increment < lob)
    return min_by_region(region_cnt, region_mask, scalef_all, out)


def comp_scalef_upb(region_cnt, region_mask, base, increment, upb, out=None):
    """
    compute largest 0<scalef<1, by region, to ensure base + scalef * increment <= upb
    result is stored in out, if present
    out is assumed to be a 1d numpy array of shape (region_cnt,)
    """
    if out is None:
        out = np.empty(region_cnt)
    else:
        if out.shape != (region_cnt,):
            raise ValueError(f"unexpected out.shape={out.shape}")
    if upb is None or (base + increment <= upb).all():
        out[:] = 1.0
        return out
    if (base > upb).any():
        raise ValueError("base > upb")
    scalef_all = np.ones(base.shape)
    np.divide(upb - base, increment, out=scalef_all, where=base + increment > upb)
    return min_by_region(region_cnt, region_mask, scalef_all, out)
