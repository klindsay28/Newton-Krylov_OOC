"""interface for stats file"""

from datetime import datetime
import os

from netCDF4 import Dataset

from .model_config import get_region_cnt
from .solver_state import action_step_log_wrap
from .utils import class_name, create_dimension_verify


class StatsFile:
    """class for stats for a solver"""

    def __init__(self, name, workdir, solver_state):
        self._fname = os.path.join(workdir, name + "_stats.nc")

        # file default _FillValue; needed for when iteration axis grows,
        # but variable values for new iteration are not yet available
        self._fill_value = -1.0e30

        self._create_stats_file(name=name, fname=self._fname, solver_state=solver_state)

    @action_step_log_wrap("_create_stats_file {fname}", per_iteration=False)
    # pylint: disable=unused-argument
    def _create_stats_file(self, name, fname, solver_state):
        """create the stats file, along with required dimensions"""

        with Dataset(fname, mode="w", format="NETCDF3_64BIT_OFFSET") as fptr:
            datestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            fcn_name = class_name(self) + "._create_stats_file"
            msg = datestamp + ": created by " + fcn_name + " for " + name + " solver"
            setattr(fptr, "history", msg)

            # define dimensions
            self._def_dimensions(fptr, {"iteration": None, "region": get_region_cnt()})

            # define coordinate variables
            self._def_vars(
                fptr,
                {
                    "iteration": {
                        "datatype": "i",
                        "dimensions": ("iteration",),
                        "attrs": {
                            "_FillValue": None,
                            "long_name": "%s solver iteration" % name,
                        },
                    }
                },
            )

    def def_dimensions(self, dimensions):
        """define dimensions in stats file"""
        with Dataset(self._fname, mode="a") as fptr:
            self._def_dimensions(fptr, dimensions)

    def _def_dimensions(self, fptr, dimensions):
        """define dimensions in an open stats file"""
        for dimname, dimlen in dimensions.items():
            create_dimension_verify(fptr, dimname, dimlen)

    def def_vars(self, vars_metadata, caller=None):
        """define vars in stats file"""
        with Dataset(self._fname, mode="a") as fptr:
            self._def_vars(fptr, vars_metadata)
            if caller is not None:
                datestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                varnames = ",".join(vars_metadata)
                fcn_name = class_name(self) + ".def_vars"
                msg = datestamp + ": " + varnames + " appended by " + fcn_name
                msg = msg + " called by " + caller
                msg = msg + "\n" + getattr(fptr, "history")
                setattr(fptr, "history", msg)

    def _def_vars(self, fptr, vars_metadata):
        """define vars in an open stats file"""
        # define specific vars
        for varname, metadata in vars_metadata.items():
            datatype = metadata.get("datatype", "f8")
            attrs = metadata.get("attrs", {})
            fill_value = attrs.get("_FillValue", self._fill_value)
            var = fptr.createVariable(
                varname, datatype, metadata["dimensions"], fill_value=fill_value
            )
            for attr_name, attr_value in attrs.items():
                if attr_name != "_FillValue":
                    setattr(var, attr_name, attr_value)

    def put_vars_iteration_invariant(self, name_vals_dict):
        """
        write iteration-invariant values to stats file
        name_vals_dict is a dict of (varname, vals) pairs
        """
        # if there is nothing to write return immediately
        if name_vals_dict == {}:
            return
        with Dataset(self._fname, mode="a") as fptr:
            for name, vals in name_vals_dict.items():
                if "iteration" in fptr.variables[name].dimensions:
                    msg = "iteration is a dimension for %s" % name
                    raise RuntimeError(msg)
                fptr.variables[name][:] = vals

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
                if "iteration" not in fptr.variables[name].dimensions:
                    msg = "iteration is not a dimension for %s" % name
                    raise RuntimeError(msg)
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
