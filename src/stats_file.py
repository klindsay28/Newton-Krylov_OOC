"""interface for stats file"""

from datetime import datetime
import os

from netCDF4 import Dataset

from .model_config import get_modelinfo, get_region_cnt
from .utils import class_name, create_dimension_exist_okay


class StatsFile:
    """class for stats for a solver"""

    def __init__(self, name, workdir, solver_state):
        self._fname = os.path.join(workdir, name + "_stats.nc")
        self._fill_value = -1.0e30  # file-wide default fill_value

        step = "stats file %s created" % self._fname
        if solver_state.step_logged(step, per_iteration=False):
            return

        with Dataset(self._fname, mode="w", format="NETCDF3_64BIT_OFFSET") as fptr:
            datestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            fcn_name = class_name(self) + ".__init__"
            msg = datestamp + ": created by " + fcn_name + " for " + name + " solver"
            setattr(fptr, "history", msg)

            # define dimensions
            fptr.createDimension("iteration", None)
            fptr.createDimension("region", get_region_cnt())

            # define coordinate variables
            fptr.createVariable("iteration", "i", dimensions=("iteration",))

        solver_state.log_step(step, per_iteration=False)

    def get_fill_value(self, varname):
        """return _FillValue for varname"""
        with Dataset(self._fname, mode="r") as fptr:
            fill_value = getattr(fptr.variables[varname], "_FillValue")
        return fill_value

    def def_vars_generic(self, vars_metadata):
        """define vars in stats file that generailize across all tracer modules"""
        dimensions = ("iteration", "region")
        with Dataset(self._fname, mode="a") as fptr:
            for tracer_module_name in get_modelinfo("tracer_module_names").split(","):
                fmt = {"tr_mod_name": tracer_module_name}
                for varname, attrs in vars_metadata.items():
                    fill_value = attrs.get("_FillValue", self._fill_value)
                    var = fptr.createVariable(
                        varname.format(**fmt), "f8", dimensions, fill_value=fill_value
                    )
                    for attr_name, attr_value in attrs.items():
                        if attr_name != "_FillValue":
                            setattr(var, attr_name, attr_value.format(**fmt))

    def def_vars_specific(self, coords_extra, vars_metadata, caller):
        """define vars in stats file that are specific to one tracer module"""

        dimensions = ("iteration", "region")
        with Dataset(self._fname, mode="a") as fptr:
            # define extra dimensions
            # if vals are provided, define and write coordinates variables
            # if vals are not provided, expect dimlen to be in metadata
            for dimname, metadata in coords_extra.items():
                if "vals" in metadata:
                    dimlen = len(metadata["vals"])
                elif "dimlen" in metadata:
                    dimlen = metadata["dimlen"]
                else:
                    msg = "dimlen for %s unknown" % dimname
                    raise ValueError(msg)
                create_dimension_exist_okay(fptr, dimname, dimlen)
                if "vals" in metadata:
                    var = fptr.createVariable(dimname, "f8", dimensions=(dimname,))
                    for attr_name, attr_value in metadata.get("attrs", {}).items():
                        setattr(var, attr_name, attr_value)
                    fptr.variables[dimname][:] = metadata["vals"]

            # define specific vars
            for varname, metadata in vars_metadata.items():
                if "dimensions_extra" in metadata:
                    dimensions_loc = dimensions + metadata["dimensions_extra"]
                else:
                    dimensions_loc = dimensions
                attrs = metadata.get("attrs", {})
                fill_value = attrs.get("_FillValue", self._fill_value)
                var = fptr.createVariable(
                    varname, "f8", dimensions_loc, fill_value=fill_value
                )
                for attr_name, attr_value in attrs.items():
                    if attr_name != "_FillValue":
                        setattr(var, attr_name, attr_value)

            datestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            varnames = ",".join(vars_metadata)
            fcn_name = class_name(self) + ".def_vars_specific"
            msg = datestamp + ": " + varnames + " appended by " + fcn_name
            msg = msg + " called by " + caller
            msg = msg + "\n" + getattr(fptr, "history")
            setattr(fptr, "history", msg)

    def put_vars_generic(self, iteration, varname, vals):
        """
        put vals to a generic varname with specified iteration index
        vals is a numpy array of RegionScalars objects
        the numpy array axis corresponds to tracer modules
        """

        tracer_module_names = get_modelinfo("tracer_module_names").split(",")

        with Dataset(self._fname, mode="a") as fptr:
            if iteration == len(fptr.variables["iteration"]):
                _grow_iteration(fptr)

            for ind, tracer_module_name in enumerate(tracer_module_names):
                full_varname = varname.format(tr_mod_name=tracer_module_name)
                fptr.variables[full_varname][iteration, :] = vals[ind].vals()

    def put_vars_specific(self, iteration, varname, vals):
        """
        put vals to a specific varname with specified iteration index
        vals is a numpy array of values for varname selected in the 1st dim
        """

        with Dataset(self._fname, mode="a") as fptr:
            if iteration == len(fptr.variables["iteration"]):
                _grow_iteration(fptr)

            fptr.variables[varname][iteration, :] = vals


def _grow_iteration(fptr):
    """grow iteration dimension"""
    # Set variables to fill_value. Without doing the fill, some installs of ncview
    # abort when viewing the stats file.
    iteration = len(fptr.variables["iteration"])
    for var in fptr.variables.values():
        if var.name == "iteration":
            var[iteration] = iteration
        elif var.dimensions[0] == "iteration":
            var[iteration, :] = getattr(var, "_FillValue")
