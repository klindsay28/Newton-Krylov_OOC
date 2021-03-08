"""functions related to horizontal mixing"""

import numpy as np
from scipy.sparse import csr_matrix

from .advection import Advection
from .model_process import ModelProcess


class HorizMix(ModelProcess):
    """class related to horizontal mixing"""

    def __init__(self, depth, ypos):

        HorizMix.depth = depth
        HorizMix.ypos = ypos

        self._tend_work = np.zeros((len(depth), len(ypos) + 1))

        # compute (time-invariant) mixing coefficients
        self._mixing_coeff = self._comp_mixing_coeff(mixing_coeff_const=1000.0)

        HorizMix.jacobian_cache = None

        super().__init__(depth, ypos)

    def _comp_mixing_coeff(self, mixing_coeff_const):
        """
        compute mixing_coeff values
        includes 1/dypos term
        """

        res = np.full((len(self.depth), len(self.ypos) - 1), mixing_coeff_const)

        # increase mixing_coeff to keep grid Peclet number <= 2.0
        peclet_p5 = (
            (0.5 / mixing_coeff_const)
            * self.ypos.delta_mid[:]
            * abs(Advection.vvel[:, 1:-1])
        )
        res *= np.where(peclet_p5 > 1.0, peclet_p5, 1.0)

        return res * self.ypos.delta_mid_r

    def comp_tend(self, time, tracer_vals):
        """
        single tracer tendency from horizontal mixing
        assume zero flux lateral boundary conditions
        """

        # compute horiz_mix fluxes
        self._tend_work[:, 1:-1] = self._mixing_coeff * (
            tracer_vals[:, 1:] - tracer_vals[:, :-1]
        )

        return (self._tend_work[:, 1:] - self._tend_work[:, :-1]) * self.ypos.delta_r

    def get_hist_vars_metadata(self):
        """return dict of process-specific history variable metadata"""

        depth_name = self.depth.axisname
        ypos_edges_name = self.ypos.dump_names["edges"]

        hist_vars_metadata = {}
        hist_vars_metadata["horiz_mixing_coeff"] = {
            "dimensions": (depth_name, ypos_edges_name),
            "attrs": {"long_name": "horizontal mixing coefficient", "units": "m^2 / s"},
        }

        return hist_vars_metadata

    def hist_write(self, sol, fptr_hist):
        """write processs-specific history variables"""

        # write tracer module independent vars
        fptr_hist.variables["horiz_mixing_coeff"][:, 1:-1] = (
            self._mixing_coeff * self.ypos.delta_mid
        )
        # kludge to avoid missing values
        fptr_hist.variables["horiz_mixing_coeff"][:, 0] = fptr_hist.variables[
            "horiz_mixing_coeff"
        ][:, 1]
        fptr_hist.variables["horiz_mixing_coeff"][:, -1] = fptr_hist.variables[
            "horiz_mixing_coeff"
        ][:, -2]

        fptr_hist.sync()

    def comp_jacobian(self, time):
        """
        compute jacobian of tracer tendencies from horizontal mixing
        jacobian units are 1 / s
        """
        if HorizMix.jacobian_cache is None:
            depth_n = len(self.depth)
            ypos_n = len(self.ypos)
            nnz = (3 * (ypos_n - 2) + 2 * 2) * depth_n
            data = np.empty(nnz)
            row_ind = np.empty(nnz, int)
            col_ind = np.empty(nnz, int)
            mat_ind = 0
            for depth_i in range(depth_n):
                for ypos_i in range(ypos_n):
                    cell_ind = ypos_i + ypos_n * depth_i
                    tmp_sum = 0.0
                    # cell to the south
                    if ypos_i > 0:
                        tmp = self._mixing_coeff[depth_i, ypos_i - 1]
                        tmp *= self.ypos.delta_r[ypos_i]
                        tmp_sum += tmp
                        data[mat_ind] = tmp
                        row_ind[mat_ind] = cell_ind
                        col_ind[mat_ind] = cell_ind - 1
                        mat_ind += 1
                    # cell to the north
                    if ypos_i < ypos_n - 1:
                        tmp = self._mixing_coeff[depth_i, ypos_i]
                        tmp *= self.ypos.delta_r[ypos_i]
                        tmp_sum += tmp
                        data[mat_ind] = tmp
                        row_ind[mat_ind] = cell_ind
                        col_ind[mat_ind] = cell_ind + 1
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
            HorizMix.jacobian_cache = csr_matrix(
                (data, (row_ind, col_ind)), shape=(dof, dof)
            )

        return HorizMix.jacobian_cache
