"""SpatialAxis class"""

import numpy as np

from netCDF4 import Dataset


class SpatialAxis:
    """class for spatial axis related quantities"""

    def __init__(self, axisname, fname):
        """
        initialize class object

        edges are the fundamental quantity defining a SpatialAxis
        all other quantities are derived from edges

        for file input, assume edges variable is named axis_name+"_edges"
        other fields in the input file are ignored
        """

        self.name = axisname
        with Dataset(fname, mode="r") as fptr:
            fptr.set_auto_mask(False)
            self.units = fptr.variables[axisname + "_edges"].units
            self.edges = fptr.variables[axisname + "_edges"][:]
        self.nlevs = len(self.edges) - 1
        self.bounds = np.empty((self.nlevs, 2))
        self.bounds[:, 0] = self.edges[:-1]
        self.bounds[:, 1] = self.edges[1:]
        self.mid = self.bounds.mean(axis=1)
        self.delta = self.bounds[:, 1] - self.bounds[:, 0]
        self.delta_r = 1.0 / self.delta
        self.delta_mid_r = 1.0 / np.ediff1d(self.mid)

    def dump(self, fname):
        """write axis information to a netCDF4 file"""

        bounds_name = self.name + "_bounds"
        edges_name = self.name + "_edges"
        delta_name = self.name + "_delta"

        with Dataset(fname, mode="w") as fptr:
            # define dimensions
            fptr.createDimension(self.name, self.nlevs)
            fptr.createDimension("nbnds", 2)
            fptr.createDimension(edges_name, 1 + self.nlevs)

            # define variables

            fptr.createVariable(self.name, "f8", dimensions=(self.name,))
            fptr.variables[self.name].long_name = self.name + " layer midpoints"
            fptr.variables[self.name].units = self.units
            fptr.variables[self.name].bounds = bounds_name

            fptr.createVariable(bounds_name, "f8", dimensions=(self.name, "nbnds"))
            fptr.variables[bounds_name].long_name = self.name + " layer bounds"

            fptr.createVariable(edges_name, "f8", dimensions=(edges_name,))
            fptr.variables[edges_name].long_name = self.name + " layer edges"
            fptr.variables[edges_name].units = self.units

            fptr.createVariable(delta_name, "f8", dimensions=(self.name,))
            fptr.variables[delta_name].long_name = self.name + " layer thickness"
            fptr.variables[delta_name].units = self.units

            # write variables
            fptr.variables[self.name][:] = self.mid
            fptr.variables[bounds_name][:] = self.bounds
            fptr.variables[edges_name][:] = self.edges
            fptr.variables[delta_name][:] = self.delta

    def grad_vals_mid(self, vals):
        """
        gradient at layer edges of vals at layer midpoints
        only works for a single tracer
        """
        grad = np.zeros(1 + self.nlevs)
        grad[1:-1] = np.ediff1d(vals) * self.delta_mid_r
        return grad

    def grad_vals_edges(self, vals):
        """
        gradient at layer midpoints of vals at layer edges
        only works for a single tracer
        """
        return np.ediff1d(vals) * self.delta_r

    def int_vals_mid(self, vals):
        """
        integral of vals at layer midpoints
        works for multiple tracer values, assuming vertical axis is last
        """
        return (self.delta * vals).sum(axis=-1)
