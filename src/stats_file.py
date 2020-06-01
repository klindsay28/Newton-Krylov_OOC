"""interface for stats file"""

from datetime import datetime
import os

from netCDF4 import Dataset

from .model_config import get_region_cnt
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

    def def_vars(self, coords_extra, vars_metadata, caller=None):
        """
        define vars in stats file
        vars are assumed to implicitly have dimensions ("iteration", "region")
        coords_extra are any coords required for additional dimensions
        """

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
                if "vals" in metadata and dimname not in fptr.variables:
                    var = fptr.createVariable(dimname, "f8", dimensions=(dimname,))
                    for attr_name, attr_value in metadata.get("attrs", {}).items():
                        setattr(var, attr_name, attr_value)
                    var[:] = metadata["vals"]

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

            if caller is not None:
                datestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                varnames = ",".join(vars_metadata)
                fcn_name = class_name(self) + ".def_vars"
                msg = datestamp + ": " + varnames + " appended by " + fcn_name
                msg = msg + " called by " + caller
                msg = msg + "\n" + getattr(fptr, "history")
                setattr(fptr, "history", msg)

    def put_vars(self, iteration, name_vals_dict):
        """
        write values to stats file for a particular iteration index
        name_vals_dict is a dict of (varname, vals) pairs
        where vals are specific to this iteration
        """

        # if there is nothing to write return immediately
        if name_vals_dict == {}:
            return

        with Dataset(self._fname, mode="a") as fptr:
            if iteration == len(fptr.variables["iteration"]):
                _grow_iteration(fptr)

            for name, vals in name_vals_dict.items():
                fptr.variables[name][iteration, :] = vals


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
