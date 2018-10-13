"""wrappers for writing and reading vars from and to a file"""

import netCDF4 as nc

def write_var(fname, varname, val):
    """write a value to var in a file"""
    with nc.Dataset(fname, 'w') as fptr:
        if (type(val) is int):
            varid = fptr.createVariable(varname, 'i4')
        else:
            varid = fptr.createVariable(varname, 'f8')
        varid[:] = val

def read_var(fname, varname):
    """read a value from var in a file"""
    with nc.Dataset(fname, 'r') as fptr:
        return fptr.variables[varname][:]
