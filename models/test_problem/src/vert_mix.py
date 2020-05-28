"""functions related to vertical mixing"""

import numpy as np


class VertMix:
    """class related to vertical mixing"""

    def __init__(self, depth):

        self._depth = depth

        self._bldepth_time = None
        self._bldepth_val = self.bldepth(0.0)

        self._mixing_coeff_time = None
        self._mixing_coeff_vals = self.mixing_coeff(0.0)

        self._tend_work = np.zeros(1 + self._depth.nlevs)

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
        between layer midpoints, m d-1
        store computed vals, so their computation can be skipped on subsequent calls
        """

        # if vals have already been computed for this time, skip computation
        if time == self._mixing_coeff_time:
            return self._mixing_coeff_vals

        # z_lin ranges from 0.0 to 1.0 over span of 40.0 m, is 0.5 at bldepth
        z_lin = 0.5 + (self._depth.edges[1:-1] - self.bldepth(time)) * (1.0 / 40.0)
        z_lin = np.maximum(0.0, np.minimum(1.0, z_lin))
        res_log10_shallow = 0.0
        res_log10_deep = -5.0
        res_log10_del = res_log10_deep - res_log10_shallow
        res_log10 = res_log10_shallow + res_log10_del * z_lin
        self._mixing_coeff_time = time
        self._mixing_coeff_vals = 86400.0 * 10.0 ** res_log10 * self._depth.delta_mid_r
        return self._mixing_coeff_vals

    def bldepth(self, time):
        """time varying boundary layer depth"""

        # if vals have already been computed for this time, skip computation
        if time == self._bldepth_time:
            return self._bldepth_val

        bldepth_min = 50.0
        bldepth_max = 150.0
        frac = 0.5 + 0.5 * np.cos((2 * np.pi) * ((time / 365.0) - 0.25))
        self._bldepth_val = bldepth_min + (bldepth_max - bldepth_min) * frac
        return self._bldepth_val
