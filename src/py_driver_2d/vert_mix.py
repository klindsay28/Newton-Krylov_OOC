"""functions related to vertical mixing"""

import numpy as np
from scipy.sparse import csr_matrix

from ..spatial_axis import SpatialAxis
from .advection import Advection
from .model_process import ModelProcess


class VertMix(ModelProcess):
    """class related to vertical mixing"""

    def __init__(self, depth, ypos):

        VertMix.depth = depth
        VertMix.ypos = ypos

        VertMix.depth_edges_axis = SpatialAxis("depth_edges", depth.mid)

        self._mixing_coeff_time = None
        self._mixing_coeff_vals = np.zeros((len(self.depth) - 1, len(self.ypos)))

        self._tend_work = np.zeros((len(self.depth) + 1, len(self.ypos)))

        super().__init__(depth, ypos)

    def comp_tend(self, time, tracer_vals):
        """
        single tracer tendency from vertical mixing
        assume zero flux surface and bottom boundary conditions
        """
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

        bldepth_cache = None
        for j in range(len(self.ypos)):
            if bldepth_vals[j] != bldepth_cache:
                self._mixing_coeff_vals[
                    :, j
                ] = VertMix.depth_edges_axis.remap_linear_interpolant(
                    [bldepth_vals[j] - 20.0, bldepth_vals[j] + 20.0],
                    [res_log_shallow, res_log_deep],
                )
                bldepth_cache = bldepth_vals[j]
                j_cache = j
            else:
                self._mixing_coeff_vals[:, j] = self._mixing_coeff_vals[:, j_cache]

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
        bldepth_max = np.interp(self.ypos.mid, [3.0e5, 6.0e5], [3000.0, bldepth_min])
        tvals = 365.0 * 86400.0 * np.array([0.3, 0.4, 0.6, 0.7])
        frac = np.interp(time, tvals, [0.0, 1.0, 1.0, 0.0])
        # frac = 1.0
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

    def comp_jacobian(self, time):
        """
        compute jacobian of tracer tendencies from vertical mixing
        jacobian units are 1 / s
        """
        mixing_coeff_vals = self.mixing_coeff(time)
        depth_n = len(self.depth)
        ypos_n = len(self.ypos)
        nnz = (3 * (depth_n - 2) + 2 * 2) * ypos_n
        data = np.empty(nnz)
        row_ind = np.empty(nnz, int)
        col_ind = np.empty(nnz, int)
        mat_ind = 0
        for depth_i in range(depth_n):
            for ypos_i in range(ypos_n):
                cell_ind = ypos_i + ypos_n * depth_i
                tmp_sum = 0.0
                # cell shallower
                if depth_i > 0:
                    tmp = mixing_coeff_vals[depth_i - 1, ypos_i]
                    tmp *= self.depth.delta_r[depth_i]
                    tmp_sum += tmp
                    data[mat_ind] = tmp
                    row_ind[mat_ind] = cell_ind
                    col_ind[mat_ind] = cell_ind - ypos_n
                    mat_ind += 1
                # cell deeper
                if depth_i < depth_n - 1:
                    tmp = mixing_coeff_vals[depth_i, ypos_i]
                    tmp *= self.depth.delta_r[depth_i]
                    tmp_sum += tmp
                    data[mat_ind] = tmp
                    row_ind[mat_ind] = cell_ind
                    col_ind[mat_ind] = cell_ind + ypos_n
                    mat_ind += 1
                # cell itself
                data[mat_ind] = -tmp_sum
                row_ind[mat_ind] = cell_ind
                col_ind[mat_ind] = cell_ind
                mat_ind += 1
        if mat_ind != nnz:
            msg = "mat_ind = %d, nnz = %d" % (mat_ind, nnz)
            raise RuntimeError(msg)
        dof = ypos_n * depth_n
        return csr_matrix((data, (row_ind, col_ind)), shape=(dof, dof))
