"""iage subclass of py_driver_2d's TracerModuleState"""

import numpy as np
from scipy.linalg import solve_banded

from .tracer_module_state import TracerModuleState


class iage(TracerModuleState):  # pylint: disable=invalid-name
    """iage tracer module specifics for TracerModuleState"""

    def comp_tend(self, time, tracer_vals_flat, processes):
        """
        compute tendency for iage
        tendency units are tr_units / s
        """
        print(time / (365.0 * 86400.0))
        shape = (len(self.depth), len(self.ypos))
        dtracer_vals_dt = super().comp_tend(time, tracer_vals_flat, processes)

        # restore surface layer to zero at rate of 24 / day over 10 m
        rate = 24.0 / 86400.0 * self.depth.delta[0] / 10.0
        tracer_vals_2d = tracer_vals_flat.reshape(shape)
        dtracer_vals_dt_2d = dtracer_vals_dt.reshape(shape)
        dtracer_vals_dt_2d[0, :] -= rate * tracer_vals_2d[0, :]

        # age 1/year
        dtracer_vals_dt[:] += 1.0 / (365.0 * 86400.0)

        return dtracer_vals_dt

    def apply_precond_jacobian(self, time_range, res_tms, mca):
        """
        apply preconditioner of jacobian of iage fcn

        time_range: length-2 sequence with start and end times of model
        res_tms: TracerModuleState object where results are stored
        mca: (vertical) mixing coefficient, time average
        """

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
