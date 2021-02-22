"""phosphorus subclass of test_problem's TracerModuleState"""

import numpy as np
from scipy.linalg import svd
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve

from .tracer_module_state import TracerModuleState


class phosphorus(TracerModuleState):  # pylint: disable=invalid-name
    """phosphorus tracer module specifics for TracerModuleState"""

    def __init__(self, tracer_module_name, fname, model_config_obj, depth):
        super().__init__(tracer_module_name, fname, model_config_obj, depth)

        # light has e-folding decay of 25m
        self.light_lim = np.exp((-1.0 / 25.0) * depth.mid)

        self._sinking_tend_work = np.zeros(1 + len(depth))

        # 0: surface layer only, 1 / day
        # 1: d po4_uptake / d po4
        self.po4_s_restoring_opt = int(
            model_config_obj.modelinfo["po4_s_restoring_opt"]
        )

    def comp_tend(self, time, tracer_vals_flat, vert_mix):
        """
        compute tendency for phosphorus tracers
        tendency units are tr_units / day
        """

        tracer_vals = tracer_vals_flat.reshape((6, -1))
        dtracer_vals_dt = np.empty(tracer_vals.shape)

        po4_uptake = self.po4_uptake(tracer_vals[0, :])

        self._comp_tend_phosphorus_core(
            time, po4_uptake, tracer_vals[0:3, :], dtracer_vals_dt[0:3, :], vert_mix
        )
        self._comp_tend_phosphorus_core(
            time, po4_uptake, tracer_vals[3:6, :], dtracer_vals_dt[3:6, :], vert_mix
        )

        # restore po4_s to po4, at a rate of 1 / day
        # compensate equally from and dop and pop,
        # so that total shadow phosphorus is conserved
        rest_term = self.po4_s_restore_tau_r(tracer_vals[0, :], po4_uptake) * (
            tracer_vals[0, :] - tracer_vals[3, :]
        )
        dtracer_vals_dt[3, :] += rest_term
        dtracer_vals_dt[4, :] -= 0.67 * rest_term
        dtracer_vals_dt[5, :] -= 0.33 * rest_term

        return dtracer_vals_dt.reshape(-1)

    def po4_s_restore_tau_r(self, po4, po4_uptake):
        """inverse timescale for po4_s restoring"""

        if self.po4_s_restoring_opt == 0:
            # 1 / day in top layer
            res = np.zeros(po4.shape)
            res[0] = 1.0
        else:
            # finite-difference approximation to d po4_uptake / d po4
            po4_delta = 1.0e-3 * abs(po4)
            po4_delta[po4_delta < 1.0e-8] = 1.0e-8
            res = (self.po4_uptake(po4 + po4_delta) - po4_uptake) / po4_delta

        return res

    def po4_uptake(self, po4):
        """return po4_uptake, [mmol m-3 d-1]"""

        # po4 half-saturation = 0.5
        # maximum uptake rate = 1 d-1
        po4_lim = po4 / (po4 + 0.5)
        return self.light_lim * po4_lim

    def _comp_tend_phosphorus_core(
        self, time, po4_uptake, tracer_vals, dtracer_vals_dt, vert_mix
    ):
        """
        core fuction for computing tendency for phosphorus tracers
        tendency units are tr_units / day
        """

        po4 = tracer_vals[0, :]
        dop = tracer_vals[1, :]
        pop = tracer_vals[2, :]

        # dop remin rate is 1% / day
        dop_remin = 0.01 * dop
        # pop remin rate is 1% / day
        pop_remin = 0.01 * pop

        sigma = 0.67

        dtracer_vals_dt[0, :] = (
            -po4_uptake + dop_remin + pop_remin + vert_mix.tend(time, po4)
        )
        dtracer_vals_dt[1, :] = (
            sigma * po4_uptake - dop_remin + vert_mix.tend(time, dop)
        )
        dtracer_vals_dt[2, :] = (
            (1.0 - sigma) * po4_uptake
            - pop_remin
            + vert_mix.tend(time, pop)
            + self._sinking_tend(pop)
        )

    def _sinking_tend(self, tracer_vals):
        """tracer tendency from sinking"""
        self._sinking_tend_work[1:-1] = -tracer_vals[
            :-1
        ]  # assume velocity is 1 m / day
        return (
            self._sinking_tend_work[1:] - self._sinking_tend_work[:-1]
        ) * self.depth.delta_r

    def hist_vars_metadata_tracer_like(self):
        """return dict of metadata for tracer-like vars to appear in the hist file"""
        res = super().hist_vars_metadata_tracer_like()
        po4_units = res["po4"]["attrs"]["units"]
        res["po4_uptake"] = {
            "attrs": {"long_name": "uptake of po4", "units": po4_units + " / d"}
        }
        res["po4_s_restore_tau_r"] = {
            "attrs": {
                "long_name": "inverse timescale for po4_s restoring",
                "units": "1 / d",
            }
        }
        return res

    def write_hist_vars(self, fptr, tracer_vals_all):
        """write hist vars"""

        # compute po4_uptake, po4_s_restore_tau_r
        po4_ind = 1
        po4_uptake_vals = np.empty((1, len(self.depth), tracer_vals_all.shape[-1]))
        po4_s_restore_tau_r_vals = np.empty(
            (1, len(self.depth), tracer_vals_all.shape[-1])
        )
        for time_ind in range(tracer_vals_all.shape[-1]):
            po4 = tracer_vals_all[po4_ind, :, time_ind]
            po4_uptake_vals[0, :, time_ind] = self.po4_uptake(po4)
            po4_s_restore_tau_r_vals[0, :, time_ind] = self.po4_s_restore_tau_r(
                po4, po4_uptake_vals[0, :, time_ind]
            )

        # append po4_uptake to tracer_vals_all and pass up the class chain
        super().write_hist_vars(
            fptr,
            np.concatenate(
                (tracer_vals_all, po4_uptake_vals, po4_s_restore_tau_r_vals)
            ),
        )

    def stats_vars_tracer_like(self):
        """
        return list of tracer-like vars in hist file to be processed for the stats file
        """
        res = super().stats_vars_tracer_like()
        res.append("po4_uptake")
        return res

    def apply_precond_jacobian(self, time_range, res_tms, mca, po4_s_restore_tau_r):
        """
        apply preconditioner of jacobian of phosphorus fcn
        it is only applied to shadow phosphorus tracers [3:6]
        """

        nlevs = len(self.depth)

        self_vals = self.get_tracer_vals_all()[3:6, :].reshape(-1)
        rhs_vals = (1.0 / (time_range[1] - time_range[0])) * self_vals

        matrix = diags(
            [
                self._diag_0_phosphorus(mca, po4_s_restore_tau_r),
                self._diag_p_1_phosphorus(mca),
                self._diag_m_1_phosphorus(mca),
                self._diag_p_nlevs_phosphorus(),
                self._diag_m_nlevs_phosphorus(po4_s_restore_tau_r),
                self._diag_p_2nlevs_phosphorus(),
                self._diag_m_2nlevs_phosphorus(po4_s_restore_tau_r),
            ],
            [0, 1, -1, nlevs, -nlevs, 2 * nlevs, -2 * nlevs],
            format="csr",
        )

        # regularize system of equations and use Richardson extrapolation to results
        matrix_adj = matrix - 1.0e-11 * eye(3 * nlevs)
        res_vals_a = spsolve(matrix_adj, rhs_vals)
        matrix_adj = matrix - 0.5e-11 * eye(3 * nlevs)
        res_vals_b = spsolve(matrix_adj, rhs_vals)
        res_vals = 2.0 * res_vals_b - res_vals_a

        _, sing_vals, r_sing_vects = svd(matrix.todense())
        min_ind = sing_vals.argmin()
        dz3 = np.concatenate((self.depth.delta, self.depth.delta, self.depth.delta))
        numer = (res_vals * dz3).sum()
        denom = (r_sing_vects[min_ind, :] * dz3).sum()
        res_vals[:] -= numer / denom * r_sing_vects[min_ind, :]

        res_vals[:] = res_vals - self_vals
        res_tms.set_tracer_vals("po4_s", res_vals[0:nlevs])
        res_tms.set_tracer_vals("dop_s", res_vals[nlevs : 2 * nlevs])
        res_tms.set_tracer_vals("pop_s", res_vals[2 * nlevs : 3 * nlevs])

    def _diag_0_phosphorus(self, mca, po4_s_restore_tau_r):
        """return main diagonal of preconditioner of jacobian of phosphorus fcn"""
        diag_0_single_tracer = np.zeros(len(self.depth))
        diag_0_single_tracer[:-1] -= (
            mca * self.depth.delta_mid_r * self.depth.delta_r[:-1]
        )
        diag_0_single_tracer[1:] -= (
            mca * self.depth.delta_mid_r * self.depth.delta_r[1:]
        )
        diag_0_po4_s = diag_0_single_tracer.copy()
        diag_0_po4_s -= po4_s_restore_tau_r  # po4_s restoring
        diag_0_dop_s = diag_0_single_tracer.copy()
        diag_0_dop_s -= 0.01  # dop_s remin
        diag_0_pop_s = diag_0_single_tracer.copy()
        diag_0_pop_s -= 0.01  # pop_s remin
        # pop_s sinking loss to layer below
        diag_0_pop_s[:-1] -= 1.0 * self.depth.delta_r[:-1]
        return np.concatenate((diag_0_po4_s, diag_0_dop_s, diag_0_pop_s))

    def _diag_p_1_phosphorus(self, mca):
        """return +1 upper diagonal of preconditioner of jacobian of phosphorus fcn"""
        diag_p_1_single_tracer = mca * self.depth.delta_mid_r * self.depth.delta_r[:-1]
        diag_p_1_po4_s = diag_p_1_single_tracer.copy()
        zero = np.zeros(1)
        diag_p_1_dop_s = diag_p_1_single_tracer.copy()
        diag_p_1_pop_s = diag_p_1_single_tracer.copy()
        return np.concatenate(
            (diag_p_1_po4_s, zero, diag_p_1_dop_s, zero, diag_p_1_pop_s)
        )

    def _diag_m_1_phosphorus(self, mca):
        """return +1 upper diagonal of preconditioner of jacobian of phosphorus fcn"""
        diag_m_1_single_tracer = mca * self.depth.delta_mid_r * self.depth.delta_r[1:]
        diag_m_1_po4_s = diag_m_1_single_tracer.copy()
        zero = np.zeros(1)
        diag_m_1_dop_s = diag_m_1_single_tracer.copy()
        diag_m_1_pop_s = diag_m_1_single_tracer.copy()
        # pop_s sinking gain from layer above
        diag_m_1_pop_s += 1.0 * self.depth.delta_r[1:]
        return np.concatenate(
            (diag_m_1_po4_s, zero, diag_m_1_dop_s, zero, diag_m_1_pop_s)
        )

    def _diag_p_nlevs_phosphorus(self):
        """
        return +nlevs upper diagonal of preconditioner of jacobian of phosphorus fcn
        """
        diag_p_1_dop_po4 = 0.01 * np.ones(len(self.depth))  # dop_s remin
        diag_p_1_pop_dop = np.zeros(len(self.depth))
        return np.concatenate((diag_p_1_dop_po4, diag_p_1_pop_dop))

    def _diag_m_nlevs_phosphorus(self, po4_s_restore_tau_r):
        """
        return -nlevs lower diagonal of preconditioner of jacobian of phosphorus fcn
        """
        diag_p_1_po4_dop = (
            0.67 * po4_s_restore_tau_r
        )  # po4_s restoring conservation balance
        diag_p_1_dop_pop = np.zeros(len(self.depth))
        return np.concatenate((diag_p_1_po4_dop, diag_p_1_dop_pop))

    def _diag_p_2nlevs_phosphorus(self):
        """
        return +2nlevs upper diagonal of preconditioner of jacobian of phosphorus fcn
        """
        return 0.01 * np.ones(len(self.depth))  # pop_s remin

    @staticmethod
    def _diag_m_2nlevs_phosphorus(po4_s_restore_tau_r):
        """
        return -2nlevs lower diagonal of preconditioner of jacobian of phosphorus fcn
        """
        diag_p_1_po4_pop = (
            0.33 * po4_s_restore_tau_r
        )  # po4_s restoring conservation balance
        return diag_p_1_po4_pop
