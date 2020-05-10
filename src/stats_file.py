"""interface for stats file"""

from netCDF4 import Dataset

from .model_config import get_modelinfo, get_region_cnt

fill_value = -1.0e30


def stats_file_create(fname):
    """create the file for solver stats"""

    tracer_module_names = get_modelinfo("tracer_module_names").split(",")

    with Dataset(fname, mode="w") as fptr:
        # define dimensions
        fptr.createDimension("iteration", None)
        fptr.createDimension("region", get_region_cnt())

        # define coordinate variables
        fptr.createVariable("iteration", "i", dimensions=("iteration",))

        # define stats variables
        dimensions = ("iteration", "region")
        attrs_dict = {
            "%s_iterate_mean": {"long_name": "mean of %s iterate"},
            "%s_iterate_norm": {"long_name": "norm of %s iterate"},
            "%s_fcn_mean": {"long_name": "mean of fcn applied to %s iterate"},
            "%s_fcn_norm": {"long_name": "norm of fcn applied to %s iterate"},
            "%s_increment_mean": {"long_name": "mean of %s Newton increment"},
            "%s_increment_norm": {"long_name": "norm of %s Newton increment"},
            "%s_Armijo_Factor": {
                "long_name": "Armijo factor applied to %s Newton increment"
            },
        }
        for tracer_module_name in tracer_module_names:
            for varname, attrs in attrs_dict.items():
                var = fptr.createVariable(
                    varname % tracer_module_name,
                    "f8",
                    dimensions,
                    fill_value=fill_value,
                )
                for attr_name, attr_value in attrs.items():
                    setattr(var, attr_name, attr_value % tracer_module_name)


def stats_file_append_vals(fname, iteration, varname, vals):
    """
    append vals to varname with specified iteration index
    vals is a numpy array of RegionScalars objects
    the numpy array axis corresponds to tracer modules
    """

    tracer_module_names = get_modelinfo("tracer_module_names").split(",")

    with Dataset(fname, mode="a") as fptr:
        # If this is the first value being written for this iteration,
        # then set iteration, and set all other variables to fill_value.
        # Without doing the fill, some installs of ncview abort when viewing
        # the stats file.
        if iteration == len(fptr.variables["iteration"]):
            for full_varname in fptr.variables:
                if full_varname == "iteration":
                    fptr.variables[full_varname][iteration] = iteration
                else:
                    fptr.variables[full_varname][iteration, :] = fill_value

        for ind, tracer_module_name in enumerate(tracer_module_names):
            full_varname = tracer_module_name + "_" + varname
            fptr.variables[full_varname][iteration, :] = vals[ind].vals()
