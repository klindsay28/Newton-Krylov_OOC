"""iage subclass of test_problem's TracerModuleState"""

import numpy as np
from scipy.linalg import solve_banded

from .tracer_module_state import TracerModuleState


class iage(TracerModuleState):  # pylint: disable=invalid-name
    """iage tracer module specifics for TracerModuleState"""

    def comp_tend(self, time, tracer_vals, dtracer_vals_dt, vert_mix):
        """
        compute tendency for iage
        tendency units are tr_units / day
        tracer_vals and dtracer_vals_dt have a leading dim of length 1
            (for the single iage tracer)
        """
        # surface_flux piston velocity = 240 m / day
        # same as restoring 24 / day over 10 m
        surf_flux = -240.0 * tracer_vals[0, 0]
        dtracer_vals_dt[0, :] = vert_mix.tend(time, tracer_vals[0, :], surf_flux)
        # age 1/year
        dtracer_vals_dt[0, :] += 1.0 / 365.0

    def apply_precond_jacobian(self, time_range, mca, res_tms):
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
        matrix_diagonals[1, 0] -= 240.0 * self.depth.delta_r[0]
        # d tend[k] / d tracer[k+1]
        matrix_diagonals[2, :-1] = mca * self.depth.delta_mid_r * self.depth.delta_r[1:]

        res_vals = solve_banded(l_and_u, matrix_diagonals, rhs_vals)

        res_tms.set_tracer_vals_all(res_vals - self_vals)
