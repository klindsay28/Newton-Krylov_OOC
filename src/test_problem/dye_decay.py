"""dye_decay subclass of test_problem's TracerModuleState"""

import numpy as np
from scipy.linalg import solve_banded

from .tracer_module_state import TracerModuleState


class dye_decay(TracerModuleState):  # pylint: disable=invalid-name
    """dye_decay tracer module specifics for TracerModuleState"""

    def __init__(self, tracer_module_name, fname, depth):
        super().__init__(tracer_module_name, fname, depth)

        # integral of surface flux over year is 1 mol m-2
        self._dye_decay_surf_flux_times = 365.0 * np.array([0.1, 0.2, 0.6, 0.7])
        self._dye_decay_surf_flux_vals = np.array([0.0, 2.0, 2.0, 0.0]) / 365.0
        self._dye_decay_surf_flux_time = None
        self._dye_decay_surf_flux_val = 0.0

    def comp_tend(self, time, tracer_vals_flat, vert_mix):
        """
        compute tendency for dye_decay tracer
        tendency units are tr_units / day
        """
        surf_flux = self._dye_decay_surf_flux(time)
        dtracer_vals_dt_flat = vert_mix.tend(time, tracer_vals_flat[:], surf_flux)
        # decay (suff / 1000) / y
        suff = self.name[10:]
        dtracer_vals_dt_flat[:] -= int(suff) * 0.001 / 365.0 * tracer_vals_flat[:]
        return dtracer_vals_dt_flat

    def _dye_decay_surf_flux(self, time):
        """return surf flux applied to dye_decay tracers"""
        if time != self._dye_decay_surf_flux_time:
            self._dye_decay_surf_flux_val = np.interp(
                time, self._dye_decay_surf_flux_times, self._dye_decay_surf_flux_vals
            )
            time = self._dye_decay_surf_flux_time
        return self._dye_decay_surf_flux_val

    def apply_precond_jacobian(self, time_range, res_tms, mca):
        """apply preconditioner of jacobian of dye_decay fcn"""

        self_vals = self.get_tracer_vals_all()[0, :]
        rhs_vals = (1.0 / (time_range[1] - time_range[0])) * self_vals

        l_and_u = (1, 1)
        matrix_diagonals = np.zeros((3, len(self.depth)))
        # d tend[k] / d tracer[k-1]
        matrix_diagonals[0, 1:] = mca * self.depth.delta_mid_r * self.depth.delta_r[:-1]
        # d tend[k] / d tracer[k]
        matrix_diagonals[1, :-1] -= (
            mca * self.depth.delta_mid_r * self.depth.delta_r[:-1]
        )
        matrix_diagonals[1, 1:] -= mca * self.depth.delta_mid_r * self.depth.delta_r[1:]
        # d tend[k] / d tracer[k+1]
        matrix_diagonals[2, :-1] = mca * self.depth.delta_mid_r * self.depth.delta_r[1:]

        # decay (suff / 1000) / y
        suff = self.name[10:]
        matrix_diagonals[1, :] -= int(suff) * 0.001 / 365.0

        res_vals = solve_banded(l_and_u, matrix_diagonals, rhs_vals)

        res_tms.set_tracer_vals_all(res_vals - self_vals)
