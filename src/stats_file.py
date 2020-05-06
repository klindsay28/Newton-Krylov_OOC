"""interface for stats file"""

from netCDF4 import Dataset

from .model_config import get_modelinfo, get_region_cnt


def stats_file_create(fname):
    """create the file for solver stats"""

    with Dataset(fname, mode="w") as fptr:
        tracer_module_names = get_modelinfo("tracer_module_names").split(",")

        # define dimensions
        fptr.createDimension("iteration", None)
        fptr.createDimension("tracer_module", len(tracer_module_names))
        fptr.createDimension("region", get_region_cnt())

        # define coordinate variables
        fptr.createVariable("iteration", "i", dimensions=("iteration",))
        fptr.createVariable("tracer_module_names", str, dimensions=("tracer_module",))

        # define stats variables
        dimensions = ("iteration", "tracer_module", "region")
        attrs_dict = {
            "iterate_mean": {"long_name": "mean of iterate"},
            "iterate_norm": {"long_name": "norm of iterate"},
            "fcn_mean": {"long_name": "mean of fcn applied to iterate"},
            "fcn_norm": {"long_name": "norm of fcn applied to iterate"},
            "increment_mean": {"long_name": "mean of Newton increment"},
            "increment_norm": {"long_name": "norm of Newton increment"},
            "Armijo_Factor": {"long_name": "Armijo factor applied to increment"},
        }
        for varname, attrs in attrs_dict.items():
            var = fptr.createVariable(varname, "f8", dimensions)
            for attr_name, attr_value in attrs.items():
                setattr(var, attr_name, attr_value)

        # write coordinate variables
        for ind, name in enumerate(tracer_module_names):
            fptr.variables["tracer_module_names"][ind] = name


def stats_file_append_vals(fname, iteration, varname, vals):
    """
    append vals to varname with specified iteration index
    vals is a numpy array of RegionScalars objects
    """

    with Dataset(fname, mode="a") as fptr:
        fptr.variables["iteration"][iteration] = iteration
        for ind in range(vals.size):
            fptr.variables[varname][iteration, ind, :] = vals[ind].vals()
