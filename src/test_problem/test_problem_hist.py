"""history file support for Newton-Krylov solver test problem"""

from netCDF4 import Dataset


def write_hist(ms_in, depth, sol, hist_fname, po4_uptake):
    """write tracer values generated in comp_fcn to hist_fname"""
    with Dataset(hist_fname, mode="w") as fptr:
        _def_hist_dims(fptr, depth)
        _def_hist_coord_vars(fptr)

        tracer_names = ms_in.tracer_names()

        for tracer_name in tracer_names:
            var = fptr.createVariable(tracer_name, "f8", dimensions=("time", "depth"))
            tracer_metadata = ms_in.tracer_metadata(tracer_name)
            if "attrs" in tracer_metadata:
                for attr_name, attr_value in tracer_metadata["attrs"].items():
                    setattr(var, attr_name, attr_value)
            setattr(var, "cell_methods", "time: point")

        hist_vars_metadata = {
            "bldepth": {
                "dimensions": ("time"),
                "attrs": {"long_name": "boundary layer depth", "units": "m",},
            },
            "mixing_coeff": {
                "dimensions": ("time", "depth_edges"),
                "attrs": {
                    "long_name": "vertical mixing coefficient",
                    "units": "m2 s-1",
                },
            },
        }

        if "phosphorus" in ms_in.tracer_module_names:
            hist_vars_metadata["po4_uptake"] = {
                "dimensions": ("time", "depth"),
                "attrs": {"long_name": "uptake of po4", "units": "mmol m-3 s-1",},
            }

        for varname, metadata in hist_vars_metadata.items():
            var = fptr.createVariable(varname, "f8", dimensions=metadata["dimensions"])
            for attr_name, attr_value in metadata["attrs"].items():
                setattr(var, attr_name, attr_value)
            setattr(var, "cell_methods", "time: point")

        _write_hist_coord_vars(fptr, sol.t, depth)

        tracer_vals = sol.y.reshape((len(tracer_names), depth.axis.nlevs, -1))
        for tracer_ind, tracer_name in enumerate(tracer_names):
            fptr.variables[tracer_name][:] = tracer_vals[tracer_ind, :, :].transpose()

        days_per_sec = 1.0 / 86400.0

        for time_ind, time in enumerate(sol.t):
            fptr.variables["bldepth"][time_ind] = depth.bldepth(time)
            fptr.variables["mixing_coeff"][
                time_ind, :
            ] = days_per_sec * depth.mixing_coeff(time)

        if "phosphorus" in ms_in.tracer_module_names:
            po4_ind = tracer_names.index("po4")
            for time_ind, time in enumerate(sol.t):
                fptr.variables["po4_uptake"][time_ind, :] = days_per_sec * po4_uptake(
                    time, tracer_vals[po4_ind, :, time_ind]
                )


def _def_hist_dims(fptr, depth):
    """define netCDF4 dimensions relevant to test_problem"""
    fptr.createDimension("time", None)
    fptr.createDimension("depth", depth.axis.nlevs)
    fptr.createDimension("depth_edges", 1 + depth.axis.nlevs)


def _def_hist_coord_vars(fptr):
    """define netCDF4 coordinate vars relevant to test_problem"""
    fptr.createVariable("time", "f8", dimensions=("time",))
    fptr.variables["time"].long_name = "time"
    fptr.variables["time"].units = "days since 0001-01-01"

    fptr.createVariable("depth", "f8", dimensions=("depth",))
    fptr.variables["depth"].long_name = "depth"
    fptr.variables["depth"].units = "m"

    fptr.createVariable("depth_edges", "f8", dimensions=("depth_edges",))
    fptr.variables["depth_edges"].long_name = "depth_edges"
    fptr.variables["depth_edges"].units = "m"


def _write_hist_coord_vars(fptr, time, depth):
    """write netCDF4 coordinate vars relevant to test_problem"""
    fptr.variables["time"][:] = time
    fptr.variables["depth"][:] = depth.axis.mid
    fptr.variables["depth_edges"][:] = depth.axis.edges
