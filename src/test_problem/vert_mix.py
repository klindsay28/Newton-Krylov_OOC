"""functions related to vertical mixing"""

import numpy as np

from . import constants


class VertMix:
    """class related to vertical mixing"""

    def __init__(self, depth):

        self._depth = depth

        self._mixing_coeff_time = None
        self._mixing_coeff_vals = np.zeros(len(self._depth) - 1)

        self._tend_work = np.zeros(len(self._depth) + 1)

    def tend(self, time, tracer_vals, surf_flux=0.0):
        """single tracer tendency from mixing, with surface flux"""
        self._tend_work[0] = -surf_flux
        self._tend_work[1:-1] = self.mixing_coeff(time) * (
            tracer_vals[1:] - tracer_vals[:-1]
        )
        return (self._tend_work[1:] - self._tend_work[:-1]) * self._depth.delta_r

    def mixing_coeff(self, time):
        """
        vertical mixing coefficient at interior edges, divided by distance
        between layer midpoints, m s-1
        store computed vals, so their computation can be skipped on subsequent calls
        """

        # if vals have already been computed for this time, skip computation
        if time == self._mixing_coeff_time:
            return self._mixing_coeff_vals

        bldepth = self.bldepth(time)
        res_log10_shallow = 0.0
        res_log10_deep = -5.0
        res_log10 = np.interp(
            self._depth.edges[1:-1],
            [bldepth - 20.0, bldepth + 20.0],
            [res_log10_shallow, res_log10_deep],
        )
        self._mixing_coeff_time = time
        self._mixing_coeff_vals[:] = 10.0 ** res_log10 * self._depth.delta_mid_r
        return self._mixing_coeff_vals

    @staticmethod
    def bldepth(time):
        """time varying boundary layer depth"""

        bldepth_min = 50.0
        bldepth_max = 150.0
        frac = 0.5 + 0.5 * np.cos((2 * np.pi) * (constants.year_per_sec * time - 0.25))
        return bldepth_min + (bldepth_max - bldepth_min) * frac
