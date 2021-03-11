"""phosphorus subclass of py_driver_2d's TracerModuleState"""

import copy
import logging

import numpy as np
from scipy import sparse

from .tracer_module_state import TracerModuleState


class phosphorus(TracerModuleState):  # pylint: disable=invalid-name
    """phosphorus tracer module specifics for TracerModuleState"""

    def __init__(self, tracer_module_name, fname, model_config_obj, depth, ypos):

        super().__init__(tracer_module_name, fname, model_config_obj, depth, ypos)

        # light has e-folding decay of 25m
        self.light_lim = np.outer(
            np.exp((-1.0 / 25.0) * depth.mid),
            np.exp(-1.0 * ((self.ypos.mid - 2.5e6) / 1.5e6) ** 2),
        )

        self.po4_ind = 0
        self.dop_ind = 1
        self.pop_ind = 2

        self.po4_halfsat = 0.5
        self.max_uptake_rate = 0.1 / 86400.0
        self.sigma = 0.67
        self.dop_remin_rate = 0.5 / (365.0 * 86400.0)
        self.pop_remin_rate = 0.5 / (365.0 * 86400.0)
        self.pop_sink_vel = 1.0 / 86400.0

        self.pop_sink_work = np.zeros((len(self.depth) + 1, len(self.ypos)))

    def comp_tend(self, time, tracer_vals, processes):
        """
        compute tendency of phosphorus tracers
        tendency units are tr_units / s
        """
        tracer_tend_vals = super().comp_tend(time, tracer_vals, processes)

        shape = (self.tracer_cnt, len(self.depth), len(self.ypos))
        tracer_tend_vals_3d = tracer_tend_vals.reshape(shape)
        tracer_vals_3d = tracer_vals.reshape(shape)

        po4_uptake = self.comp_po4_uptake(tracer_vals_3d[self.po4_ind, :])
        tracer_tend_vals_3d[self.po4_ind, :] -= po4_uptake
        tracer_tend_vals_3d[self.dop_ind, :] += self.sigma * po4_uptake
        tracer_tend_vals_3d[self.pop_ind, :] += (1.0 - self.sigma) * po4_uptake

        dop_remin = self.dop_remin_rate * tracer_vals_3d[self.dop_ind, :]
        pop_remin = self.pop_remin_rate * tracer_vals_3d[self.pop_ind, :]
        tracer_tend_vals_3d[self.po4_ind, :] += dop_remin + pop_remin
        tracer_tend_vals_3d[self.dop_ind, :] -= dop_remin
        tracer_tend_vals_3d[self.pop_ind, :] -= pop_remin

        self.pop_sink_work[1:-1, :] = self.pop_sink_vel * (
            tracer_vals_3d[self.pop_ind, :-1]
        )
        tracer_tend_vals_3d[self.pop_ind, :] += self.depth.delta_r[:, np.newaxis] * (
            self.pop_sink_work[:-1, :] - self.pop_sink_work[1:, :]
        )

        return tracer_tend_vals_3d.reshape(-1)

    def comp_po4_uptake(self, po4):
        """return po4_uptake, [mmol/m^3/s]"""

        po4_lim = po4 / (po4 + self.po4_halfsat)
        return self.max_uptake_rate * self.light_lim * po4_lim

    def comp_po4_uptake_jacobian(self, po4):
        """return deriv of po4_uptake wrt po4, [1/s]"""

        po4_lim_jacobian = self.po4_halfsat / (po4 + self.po4_halfsat) ** 2
        return self.max_uptake_rate * self.light_lim * po4_lim_jacobian

    def comp_jacobian(self, time, tracer_vals, processes):
        """
        compute jacobian of phosphorus tracer tendencies
        jacobian units are 1 / s
        note that solve_ivp requires comp_jacobian to have same signature as comp_tend
            tracer values
        """
        jacobian = super().comp_jacobian(time, tracer_vals, processes)

        shape = (self.tracer_cnt, len(self.depth), len(self.ypos))
        tracer_vals_3d = tracer_vals.reshape(shape)

        # block structure of jacobian
        if (self.po4_ind, self.dop_ind, self.pop_ind) != (0, 1, 2):
            raise RuntimeError("tracer indices out of assumed order")

        block_size = len(self.depth) * len(self.ypos)
        zeros_block = sparse.csr_matrix((block_size, block_size))
        id_block = sparse.identity(block_size)

        po4_uptake_jacobian_2d = self.comp_po4_uptake_jacobian(
            tracer_vals_3d[self.po4_ind, :]
        )
        po4_uptake_block = sparse.diags(po4_uptake_jacobian_2d.reshape(-1))
        jacobian += sparse.bmat(
            [
                [-po4_uptake_block, zeros_block, zeros_block],
                [self.sigma * po4_uptake_block, zeros_block, zeros_block],
                [(1.0 - self.sigma) * po4_uptake_block, zeros_block, zeros_block],
            ]
        )

        dop_remin_block = self.dop_remin_rate * id_block
        pop_remin_block = self.pop_remin_rate * id_block
        jacobian += sparse.bmat(
            [
                [zeros_block, dop_remin_block, pop_remin_block],
                [zeros_block, -dop_remin_block, zeros_block],
                [zeros_block, zeros_block, -pop_remin_block],
            ]
        )

        pop_sink_diag_0_2d = np.empty((len(self.depth), len(self.ypos)))
        pop_sink_diag_0_2d[:] = -self.pop_sink_vel * self.depth.delta_r[:, np.newaxis]
        pop_sink_diag_0_2d[-1, :] = 0.0
        pop_sink_diag_m1_2d = np.empty((len(self.depth) - 1, len(self.ypos)))
        pop_sink_diag_m1_2d[:] = self.pop_sink_vel * self.depth.delta_r[1:, np.newaxis]
        pop_sink_block = sparse.diags(
            (pop_sink_diag_0_2d.reshape(-1), pop_sink_diag_m1_2d.reshape(-1)),
            (0, -len(self.ypos)),
        )
        jacobian += sparse.bmat(
            [
                [zeros_block, zeros_block, zeros_block],
                [zeros_block, zeros_block, zeros_block],
                [zeros_block, zeros_block, pop_sink_block],
            ]
        )

        return jacobian

    def hist_vars_metadata_tracer_like(self):
        """return dict of metadata for tracer-like vars to appear in the hist file"""
        res = super().hist_vars_metadata_tracer_like()
        po4_units = res["po4"]["attrs"]["units"]
        res["po4_uptake"] = {
            "attrs": {"long_name": "uptake of po4", "units": po4_units + " / s"}
        }
        return res

    def write_hist_vars(self, fptr, tracer_vals_all):
        """write hist vars"""

        # compute po4_uptake
        po4_uptake_vals = np.empty((1,) + tracer_vals_all.shape[1:])
        for time_ind in range(tracer_vals_all.shape[-1]):
            po4 = tracer_vals_all[self.po4_ind, :, :, time_ind]
            po4_uptake_vals[0, :, :, time_ind] = self.comp_po4_uptake(po4)

        # append po4_uptake to tracer_vals_all and pass up the class chain
        super().write_hist_vars(
            fptr, np.concatenate((tracer_vals_all, po4_uptake_vals)),
        )

    def apply_precond_jacobian(self, time_range, res_tms, processes, time, po4):
        """
        apply preconditioner of jacobian of phosphorus fcn

        time_range: length-2 sequence with start and end times of model
        res_tms: TracerModuleState object where results are stored
        """
        logger = logging.getLogger(__name__)

        self_vals_3d = self.get_tracer_vals_all()
        shape = self_vals_3d.shape
        self_vals = self_vals_3d.reshape(-1)

        time_n = 1
        time_delta = (time_range[1] - time_range[0]) / time_n

        # argument to comp_jacobian
        # only po4 values are used
        tracer_vals_3d = np.zeros(self_vals_3d.shape)
        tracer_vals = tracer_vals_3d.reshape(-1)

        mat_id = sparse.identity(self_vals.size)
        mat = sparse.identity(self_vals.size)
        for time_ind in range(time_n):
            time_end = time_range[0] + (time_ind + 1.0) * time_delta
            tracer_vals_3d[self.po4_ind, :] = po4[np.argmin(abs(time_end - time)), :]
            time_mid = time_range[0] + (time_ind + 0.5) * time_delta
            mat_tmp = time_delta * self.comp_jacobian(time_mid, tracer_vals, processes)
            mat *= mat_id - mat_tmp
        mat = mat_id - mat

        e_cnt = 5
        e_vals, e_vects = sparse.linalg.eigs(mat, k=e_cnt)
        for k in range(e_cnt):
            logger.info(
                "large e_val[%d] = %e + %e j", k, e_vals[k].real, e_vals[k].imag
            )
        e_vals, e_vects = sparse.linalg.eigs(mat, k=e_cnt, sigma=0.0)
        for k in range(e_cnt):
            logger.info(
                "small e_val[%d] = %e + %e j", k, e_vals[k].real, e_vals[k].imag
            )

        # confirm that imaginary part of null vector is small before dropping it
        null_vect_comp = e_vects[:, 0]
        if max(abs(null_vect_comp.imag)) > 1.0e-10 * max(abs(null_vect_comp.real)):
            raise RuntimeError("1st eigenvector has non-trivial imaginary part")
        null_vect = null_vect_comp.real

        # Regularize system of equations by shifting matrix.
        # Extrapolate to zero-shift to generate results.
        shift = 0.5 * e_vals[1].real
        solve_vals_tmp = sparse.linalg.spsolve(mat - shift * mat_id, self_vals)
        solve_vals = sparse.linalg.spsolve(mat - (0.5 * shift) * mat_id, self_vals)
        solve_vals[:] = 2.0 * solve_vals[:] - solve_vals_tmp[:]

        # Subtract from solve_vals a multiple of e_vect corresponding to smallest e_val.
        # The multiple is compute to ensure that the mean of the result is 0.
        # This ensures that total P is conserved.
        e_vect_tms = copy.copy(self)
        e_vect_tms.set_tracer_vals_all(null_vect.reshape(shape), reseat_vals=True)
        e_vect_tms /= e_vect_tms.mean()
        solve_vals_tms = copy.copy(self)
        solve_vals_tms.set_tracer_vals_all(solve_vals.reshape(shape), reseat_vals=True)
        solve_vals_tms -= solve_vals_tms.mean() * e_vect_tms

        res_tms.set_tracer_vals_all((solve_vals - self_vals).reshape(shape))
