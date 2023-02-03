"""iage subclass of test_problem's TracerModuleState"""

import numpy as np
from scipy import linalg

from . import constants
from .tracer_module_state import TracerModuleState


class iage(TracerModuleState):  # pylint: disable=invalid-name
    """iage tracer module specifics for TracerModuleState"""

    def __init__(self, tracer_module_name, fname, model_config_obj, depth):
        super().__init__(tracer_module_name, fname, model_config_obj, depth)

        # surface_flux equivalent to restoring 24 / day over 10 m
        # leads to piston velocity = 240 m / day
        self.pist_vel = 24.0 * constants.day_per_sec * 10.0

    def comp_tend(self, time, tracer_vals_flat, vert_mix):
        """
        compute tendency for iage
        tendency units are tr_units / s
        """
        surf_flux = -self.pist_vel * tracer_vals_flat[0]
        dtracer_vals_dt_flat = vert_mix.tend(time, tracer_vals_flat[:], surf_flux)
        # age 1/year
        dtracer_vals_dt_flat += constants.year_per_sec
        return dtracer_vals_dt_flat

    def apply_precond_jacobian(self, time_range, res_tms, mca):
        """apply preconditioner of jacobian of iage fcn"""

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
        matrix_diagonals[1, 0] -= self.pist_vel * self.depth.delta_r[0]
        # d tend[k] / d tracer[k+1]
        matrix_diagonals[2, :-1] = mca * self.depth.delta_mid_r * self.depth.delta_r[1:]

        res_vals = linalg.solve_banded(l_and_u, matrix_diagonals, rhs_vals)

        res_tms.set_tracer_vals_all(res_vals - self_vals)
