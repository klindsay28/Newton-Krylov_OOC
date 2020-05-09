"""SpatialAxis class"""

import numpy as np

from netCDF4 import Dataset


class SpatialAxis:
    """class for spatial axis related quantities"""

    def __init__(self, axisname, fname):
        """
        initialize class object
        for file input, assume edges have name axis_name+"_edges"
        """

        self.name = axisname
        with Dataset(fname) as fptr:
            fptr.set_auto_mask(False)
            edges = fptr.variables[axisname + "_edges"]
            self.units = edges.units
            self.edges = edges[:]
        self.mid = 0.5 * (self.edges[:-1] + self.edges[1:])
        self.delta = np.ediff1d(self.edges)
        self.delta_r = 1.0 / self.delta
        self.delta_mid_r = 1.0 / np.ediff1d(self.mid)
        self.nlevs = len(self.mid)

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
