"""interface for stats file"""

from datetime import datetime
import os

from netCDF4 import Dataset

from .model_config import get_modelinfo, get_region_cnt


class StatsFile:
    """class for stats for a solver"""

    def __init__(self, name, workdir, resume, fill_value=-1.0e30):
        self._fname = os.path.join(workdir, name + "_stats.nc")
        self._fill_value = fill_value

        if resume:
            # verify that stats file exists
            if not os.path.exists(self._fname):
                msg = "resume=True but stats file %s doesn't exist" % self._fname
                raise RuntimeError(msg)
            return

        with Dataset(self._fname, mode="w", format="NETCDF3_64BIT_OFFSET") as fptr:
            datestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            fcn_name = __name__ + ".stats_file_create"
            msg = datestamp + ": created by " + fcn_name + " for " + name + " solver"
            setattr(fptr, "history", msg)

            # define dimensions
            fptr.createDimension("iteration", None)
            fptr.createDimension("region", get_region_cnt())

            # define coordinate variables
            fptr.createVariable("iteration", "i", dimensions=("iteration",))

    def def_vars_gen(self, vars_metadata):
        """define vars in stats file that generailize across all tracer modules"""
        dimensions = ("iteration", "region")
        with Dataset(self._fname, mode="a") as fptr:
            for tracer_module_name in get_modelinfo("tracer_module_names").split(","):
                fmt = {"tr_mod_name": tracer_module_name}
                for varname, attrs in vars_metadata.items():
                    var = fptr.createVariable(
                        varname.format(**fmt),
                        "f8",
                        dimensions,
                        fill_value=self._fill_value,
                    )
                    for attr_name, attr_value in attrs.items():
                        setattr(var, attr_name, attr_value.format(**fmt))

    def def_vars_spec(self, vars_metadata):
        """define vars in stats file that are specific to one tracer module"""
        dimensions = ("iteration", "region")
        with Dataset(self._fname, mode="a") as fptr:
            for varname, attrs in vars_metadata.items():
                var = fptr.createVariable(
                    varname, "f8", dimensions, fill_value=self._fill_value
                )
                for attr_name, attr_value in attrs.items():
                    setattr(var, attr_name, attr_value)

    def put_vars_generic(self, iteration, varname, vals):
        """
        put vals to a generic varname with specified iteration index
        vals is a numpy array of RegionScalars objects
        the numpy array axis corresponds to tracer modules
        """

        tracer_module_names = get_modelinfo("tracer_module_names").split(",")

        with Dataset(self._fname, mode="a") as fptr:
            if iteration == len(fptr.variables["iteration"]):
                self._grow_iteration(fptr)

            for ind, tracer_module_name in enumerate(tracer_module_names):
                full_varname = varname.format(tr_mod_name=tracer_module_name)
                fptr.variables[full_varname][iteration, :] = vals[ind].vals()

    def put_vars_specific(self, iteration, varname, vals):
        """
        put vals to a specific varname with specified iteration index
        vals is a RegionScalars object
        """

        with Dataset(self._fname, mode="a") as fptr:
            if iteration == len(fptr.variables["iteration"]):
                self._grow_iteration(fptr)

            fptr.variables[varname][:] = vals.vals()

    def _grow_iteration(self, fptr):
        """grow iteration dimension"""
        # Set variables to fill_value. Without doing the fill, some installs of ncview
        # abort when viewing the stats file.
        iteration = len(fptr.variables["iteration"])
        for varname in fptr.variables:
            if varname == "iteration":
                fptr.variables[varname][iteration] = iteration
            else:
                fptr.variables[varname][iteration, :] = self._fill_value
