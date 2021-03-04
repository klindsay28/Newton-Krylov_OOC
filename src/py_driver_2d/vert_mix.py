"""functions related to vertical mixing"""

import numpy as np

from .advection import Advection
from .model_process import ModelProcess


class VertMix(ModelProcess):
    """class related to vertical mixing"""

    def __init__(self, depth, ypos):

        VertMix.depth = depth
        VertMix.ypos = ypos

        self._mixing_coeff_time = None
        self._mixing_coeff_vals = np.zeros((len(self.depth) - 1, len(self.ypos)))

        self._tend_work = np.zeros((len(self.depth) + 1, len(self.ypos)))

        super().__init__(depth, ypos)

    def comp_tend(self, time, tracer_vals):
        """single tracer tendency from mixing, with zero surface and bottom fluxes"""
        self._tend_work[1:-1, :] = self.mixing_coeff(time) * (
            tracer_vals[1:, :] - tracer_vals[:-1, :]
        )
        return (self._tend_work[1:, :] - self._tend_work[:-1, :]) * self.depth.delta_r[
            :, np.newaxis
        ]

    def mixing_coeff(self, time):
        """
        vertical mixing coefficient at interior edges, divided by distance
        between layer midpoints, m / s
        store computed vals, so their computation can be skipped on subsequent calls
        """

        # if vals have already been computed for this time, skip computation
        if time == self._mixing_coeff_time:
            return self._mixing_coeff_vals

        self._mixing_coeff_time = time

        bldepth_vals = self.bldepth(time)
        res_log_shallow = np.log(1.0e1)
        res_log_deep = np.log(5.0e-4)
        for j in range(len(self.ypos)):
            self._mixing_coeff_vals[:, j] = np.interp(
                self.depth.edges[1:-1],
                [bldepth_vals[j] - 20.0, bldepth_vals[j] + 20.0],
                [res_log_shallow, res_log_deep],
            )
        self._mixing_coeff_vals[:] = np.exp(self._mixing_coeff_vals[:])

        # increase mixing_coeff to keep grid Peclet number <= 2.0
        peclet_p5 = (
            0.5
            * self.depth.delta_mid[:, np.newaxis]
            * abs(Advection.wvel[1:-1, :])
            / self._mixing_coeff_vals[:]
        )
        self._mixing_coeff_vals[:] *= np.where(peclet_p5 > 1.0, peclet_p5, 1.0)

        self._mixing_coeff_vals[:] = (
            self._mixing_coeff_vals[:] * self.depth.delta_mid_r[:, np.newaxis]
        )
        return self._mixing_coeff_vals

    def bldepth(self, time):
        """time varying boundary layer depth"""

        bldepth_min = 35.0
        bldepth_max = np.interp(self.ypos.mid, [5.0e5, 10.0e5], [1500.0, bldepth_min])
        tvals = 365.0 * 86400.0 * np.array([0.3, 0.4, 0.6, 0.7])
        frac = np.interp(time, tvals, [0.0, 1.0, 1.0, 0.0])
        frac = 1.0
        return bldepth_min + (bldepth_max - bldepth_min) * frac

    def get_hist_vars_metadata(self):
        """return dict of process-specific history variable metadata"""

        depth_edges_name = self.depth.dump_names["edges"]
        ypos_name = self.ypos.axisname

        hist_vars_metadata = {}
        hist_vars_metadata["bldepth"] = {
            "dimensions": ("time", ypos_name),
            "attrs": {"long_name": "boundary layer depth", "units": "m"},
        }
        hist_vars_metadata["vert_mixing_coeff"] = {
            "dimensions": ("time", depth_edges_name, ypos_name),
            "attrs": {"long_name": "vertical mixing coefficient", "units": "m^2 / s"},
        }

        return hist_vars_metadata

    def hist_write(self, sol, fptr_hist):
        """write processs-specific history variables"""

        # (re-)compute and write tracer module independent vars
        for time_ind, time in enumerate(sol.t):
            fptr_hist.variables["bldepth"][time_ind, :] = self.bldepth(time)
            fptr_hist.variables["vert_mixing_coeff"][time_ind, 1:-1, :] = (
                self.mixing_coeff(time) * self.depth.delta_mid[:, np.newaxis]
            )
            # kludge to avoid missing values
            fptr_hist.variables["vert_mixing_coeff"][
                time_ind, 0, :
            ] = fptr_hist.variables["vert_mixing_coeff"][time_ind, 1, :]
            fptr_hist.variables["vert_mixing_coeff"][
                time_ind, -1, :
            ] = fptr_hist.variables["vert_mixing_coeff"][time_ind, -2, :]

        fptr_hist.sync()
