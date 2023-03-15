"""functions related to advection"""

import numpy as np
from scipy import sparse

from .model_process import ModelProcess


class Advection(ModelProcess):
    """class related to advection"""

    def __init__(self, depth, ypos, modelinfo):
        super().__init__(depth, ypos)

        self.gen_vel_field(depth, ypos, float(modelinfo["max_abs_vvel"]))

        self._tend_work_y = np.zeros((len(self.depth), len(self.ypos) + 1))
        self._tend_work_z = np.zeros((len(self.depth) + 1, len(self.ypos)))

        Advection.jacobian_cache = None

    @staticmethod
    def gen_vel_field(depth, ypos, max_abs_vvel):
        """generate streamfunction and velocity field"""

        depth_norm = (depth.edges - depth.edges.min()) / (
            depth.edges.max() - depth.edges.min()
        )
        stretch = 2.0
        depth_norm = stretch * depth_norm / (1 + (stretch - 1) * depth_norm)
        depth_fcn = (27.0 / 4.0) * depth_norm * (1.0 - depth_norm) ** 2

        ypos_norm = (ypos.edges - ypos.edges.min()) / (
            ypos.edges.max() - ypos.edges.min()
        )
        ypos_fcn = 4.0 * ypos_norm * (1.0 - ypos_norm)

        stream = np.outer(depth_fcn, ypos_fcn)

        # normalize so that max vvel = max_abs_vvel
        vvel = (stream[1:, :] - stream[:-1, :]) * depth.delta_r[:, np.newaxis]
        stream = stream * max_abs_vvel / abs(vvel).max()

        vvel = (stream[1:, :] - stream[:-1, :]) * depth.delta_r[:, np.newaxis]
        wvel = (stream[:, 1:] - stream[:, :-1]) * ypos.delta_r

        Advection.stream = stream
        Advection.vvel = vvel
        Advection.wvel = wvel

    def comp_tend(self, time, tracer_vals):
        """single tracer tendency from advection"""

        shape = tracer_vals.shape
        res = np.empty(shape)

        for tracer_ind in range(shape[0]):
            self._tend_work_y[:, 1:-1] = 0.5 * (
                tracer_vals[tracer_ind, :, 1:] + tracer_vals[tracer_ind, :, :-1]
            )
            self._tend_work_y *= self.vvel

            res[tracer_ind, :] = self.ypos.delta_r * (
                self._tend_work_y[:, :-1] - self._tend_work_y[:, 1:]
            )

            self._tend_work_z[1:-1, :] = 0.5 * (
                tracer_vals[tracer_ind, 1:, :] + tracer_vals[tracer_ind, :-1, :]
            )
            self._tend_work_z *= self.wvel

            res[tracer_ind, :] += self.depth.delta_r[:, np.newaxis] * (
                self._tend_work_z[1:, :] - self._tend_work_z[:-1, :]
            )

        return res

    def get_hist_vars_metadata(self):
        """return dict of process-specific history variable metadata"""

        depth_name = self.depth.axisname
        depth_edges_name = self.depth.dump_names["edges"]
        ypos_name = self.ypos.axisname
        ypos_edges_name = self.ypos.dump_names["edges"]

        hist_vars_metadata = {}
        hist_vars_metadata["stream"] = {
            "dimensions": (depth_edges_name, ypos_edges_name),
            "attrs": {"long_name": "velocity streamfunction", "units": "m^2 / s"},
        }
        hist_vars_metadata["vvel"] = {
            "dimensions": (depth_name, ypos_edges_name),
            "attrs": {"long_name": "velocity in ypos direction", "units": "m / s"},
        }
        hist_vars_metadata["wvel"] = {
            "dimensions": (depth_edges_name, ypos_name),
            "attrs": {"long_name": "velocity in depth direction", "units": "m / s"},
        }

        return hist_vars_metadata

    def hist_write(self, sol, fptr_hist):
        """write processs-specific history variables"""

        fptr_hist.variables["stream"][:] = self.stream
        fptr_hist.variables["vvel"][:] = self.vvel
        fptr_hist.variables["wvel"][:] = self.wvel

        fptr_hist.sync()

    def comp_jacobian(self, time, tracer_cnt):
        """
        compute jacobian of tracer tendencies from advection
        jacobian units are 1 / s
        """
        if Advection.jacobian_cache is None:
            # First construct matrix with sparsity pattern for a single tracer.
            # Then concatenate using sparse.block_diag.
            depth_n = len(self.depth)
            ypos_n = len(self.ypos)
            nnz = 5 * (depth_n - 2) * (ypos_n - 2)
            nnz += 4 * 2 * (depth_n - 2) + 4 * 2 * (ypos_n - 2) + 3 * 4
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
                        tmp = -0.5 * self.wvel[depth_i, ypos_i]
                        tmp *= self.depth.delta_r[depth_i]
                        tmp_sum += tmp
                        data[mat_ind] = tmp
                        row_ind[mat_ind] = cell_ind
                        col_ind[mat_ind] = cell_ind - ypos_n
                        mat_ind += 1
                    # cell to the south
                    if ypos_i > 0:
                        tmp = 0.5 * self.vvel[depth_i, ypos_i]
                        tmp *= self.ypos.delta_r[ypos_i]
                        tmp_sum += tmp
                        data[mat_ind] = tmp
                        row_ind[mat_ind] = cell_ind
                        col_ind[mat_ind] = cell_ind - 1
                        mat_ind += 1
                    # cell to the north
                    if ypos_i < ypos_n - 1:
                        tmp = -0.5 * self.vvel[depth_i, ypos_i + 1]
                        tmp *= self.ypos.delta_r[ypos_i]
                        tmp_sum += tmp
                        data[mat_ind] = tmp
                        row_ind[mat_ind] = cell_ind
                        col_ind[mat_ind] = cell_ind + 1
                        mat_ind += 1
                    # cell deeper
                    if depth_i < depth_n - 1:
                        tmp = 0.5 * self.wvel[depth_i + 1, ypos_i]
                        tmp *= self.depth.delta_r[depth_i]
                        tmp_sum += tmp
                        data[mat_ind] = tmp
                        row_ind[mat_ind] = cell_ind
                        col_ind[mat_ind] = cell_ind + ypos_n
                        mat_ind += 1
                    # cell itself
                    data[mat_ind] = tmp_sum
                    row_ind[mat_ind] = cell_ind
                    col_ind[mat_ind] = cell_ind
                    mat_ind += 1
            if mat_ind != nnz:
                raise RuntimeError(f"mat_ind={mat_ind}, nnz={nnz}")
            dof = ypos_n * depth_n
            Advection.jacobian_cache = sparse.csr_matrix(
                (data, (row_ind, col_ind)), shape=(dof, dof)
            )

        return sparse.block_diag(tracer_cnt * [Advection.jacobian_cache], "csr")
