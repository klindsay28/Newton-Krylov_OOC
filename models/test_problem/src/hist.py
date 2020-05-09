"""history file support for Newton-Krylov solver test problem"""

from cf_units import Unit
from netCDF4 import Dataset


def hist_write(ms_in, sol, hist_fname, newton_fcn_obj):
    """write tracer values generated in comp_fcn to hist_fname"""
    with Dataset(hist_fname, mode="w") as fptr:
        tracer_names = ms_in.tracer_names()
        depth_units = newton_fcn_obj.depth.units

        _def_dims(fptr, newton_fcn_obj.depth)
        _def_coord_vars(fptr, depth_units)

        for tracer_name in tracer_names:
            varname = tracer_name
            var = fptr.createVariable(varname, "f8", dimensions=("time", "depth"))
            tracer_metadata = ms_in.tracer_metadata(tracer_name)
            if "attrs" in tracer_metadata:
                for attr_name, attr_value in tracer_metadata["attrs"].items():
                    setattr(var, attr_name, attr_value)
            setattr(var, "cell_methods", "time: point")

            varname = tracer_name + "_zint"
            var = fptr.createVariable(varname, "f8", dimensions=("time",))
            tracer_metadata = ms_in.tracer_metadata(tracer_name)
            if "attrs" in tracer_metadata:
                for attr_name, attr_value in tracer_metadata["attrs"].items():
                    if attr_name == "units":
                        setattr(var, attr_name, _zint_units(attr_value, depth_units))
                    elif attr_name == "long_name":
                        setattr(var, attr_name, attr_value + ", vertical integral")
                    else:
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
            po4_units = ms_in.tracer_metadata("po4")["attrs"]["units"]
            hist_var_units = po4_units + " s-1"
            hist_vars_metadata["po4_uptake"] = {
                "dimensions": ("time", "depth"),
                "attrs": {"long_name": "uptake of po4", "units": hist_var_units,},
            }
            hist_var_units = _zint_units(po4_units + " s-1", depth_units)
            hist_vars_metadata["po4_uptake_zint"] = {
                "dimensions": ("time"),
                "attrs": {"long_name": "uptake of po4", "units": hist_var_units,},
            }

        for varname, metadata in hist_vars_metadata.items():
            var = fptr.createVariable(varname, "f8", dimensions=metadata["dimensions"])
            for attr_name, attr_value in metadata["attrs"].items():
                setattr(var, attr_name, attr_value)
            setattr(var, "cell_methods", "time: point")

        _write_coord_vars(fptr, sol.t, newton_fcn_obj.depth)

        tracer_vals = sol.y.reshape((len(tracer_names), newton_fcn_obj.depth.nlevs, -1))
        for tracer_ind, tracer_name in enumerate(tracer_names):
            tracer_vals_time_depth = tracer_vals[tracer_ind, :, :].transpose()

            varname = tracer_name
            fptr.variables[tracer_name][:] = tracer_vals_time_depth

            varname = tracer_name + "_zint"
            fptr.variables[varname][:] = newton_fcn_obj.depth.int_vals_mid(
                tracer_vals_time_depth
            )

        days_per_sec = 1.0 / 86400.0

        for time_ind, time in enumerate(sol.t):
            fptr.variables["bldepth"][time_ind] = newton_fcn_obj.bldepth(time)
            fptr.variables["mixing_coeff"][
                time_ind, :
            ] = days_per_sec * newton_fcn_obj.mixing_coeff(time)

        if "phosphorus" in ms_in.tracer_module_names:
            po4_ind = tracer_names.index("po4")
            for time_ind, time in enumerate(sol.t):
                po4 = tracer_vals[po4_ind, :, time_ind]
                po4_uptake = days_per_sec * newton_fcn_obj.po4_uptake(time, po4)
                fptr.variables["po4_uptake"][time_ind, :] = po4_uptake
                fptr.variables["po4_uptake_zint"][
                    time_ind
                ] = newton_fcn_obj.depth.int_vals_mid(po4_uptake)


def _def_dims(fptr, depth):
    """define netCDF4 dimensions relevant to test_problem"""
    fptr.createDimension("time", None)
    fptr.createDimension("depth", depth.nlevs)
    fptr.createDimension("nbnds", 2)
    fptr.createDimension("depth_edges", 1 + depth.nlevs)


def _def_coord_vars(fptr, depth_units):
    """define netCDF4 coordinate vars relevant to test_problem"""
    fptr.createVariable("time", "f8", dimensions=("time",))
    fptr.variables["time"].long_name = "time"
    fptr.variables["time"].units = "days since 0001-01-01"

    fptr.createVariable("depth", "f8", dimensions=("depth",))
    fptr.variables["depth"].long_name = "depth layer midpoints"
    fptr.variables["depth"].units = depth_units
    fptr.variables["depth"].bounds = "depth_bounds"

    fptr.createVariable("depth_bounds", "f8", dimensions=("depth", "nbnds"))
    fptr.variables["depth_bounds"].long_name = "depth layer bounds"

    fptr.createVariable("depth_edges", "f8", dimensions=("depth_edges",))
    fptr.variables["depth_edges"].long_name = "depth layer edges"
    fptr.variables["depth_edges"].units = depth_units


def _write_coord_vars(fptr, time, depth):
    """write netCDF4 coordinate vars relevant to test_problem"""
    fptr.variables["time"][:] = time
    fptr.variables["depth"][:] = depth.mid
    fptr.variables["depth_bounds"][:] = depth.bounds
    fptr.variables["depth_edges"][:] = depth.edges


def _zint_units(units_in, depth_units):
    """vertical integral units"""
    units_out_list = units_in.split()
    for ind, term in enumerate(units_out_list):
        if _is_power_of(term, depth_units):
            units_out_list[ind] = Unit(term + " " + depth_units).format()
            return " ".join(units_out_list)
    units_out_list.append(depth_units)
    return " ".join(units_out_list)


def _is_power_of(term1, term2):
    """is term1 convertible from a power of term2"""
    for power in range(-6, 7):
        term2_raised = "(%s)%d" % (term2, power)
        if Unit(term1).is_convertible(term2_raised):
            return True
    return False
