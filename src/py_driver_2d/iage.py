"""iage subclass of py_driver_2d's TracerModuleState"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from .tracer_module_state import TracerModuleState


class iage(TracerModuleState):  # pylint: disable=invalid-name
    """iage tracer module specifics for TracerModuleState"""

    def comp_tend(self, time, tracer_vals, processes):
        """
        compute tendency of iage tracers
        tendency units are tr_units / s
        """
        shape = (len(self.depth), len(self.ypos))
        tracer_tend_vals = super().comp_tend(time, tracer_vals, processes)

        # restore surface layer to zero at rate of 24 / day over 10 m
        rate = 24.0 / 86400.0 * self.depth.delta[0] / 10.0
        tracer_vals_2d = tracer_vals.reshape(shape)
        tracer_tend_vals_2d = tracer_tend_vals.reshape(shape)
        tracer_tend_vals_2d[0, :] -= rate * tracer_vals_2d[0, :]

        # age 1/year
        tracer_tend_vals[:] += 1.0 / (365.0 * 86400.0)

        return tracer_tend_vals

    def comp_jacobian(self, time, tracer_vals, processes):
        """
        compute jacobian of iage tracer tendencies
        tracer_vals: not used, only present because solve_ivp requires
            comp_jacobian to have same signature as comp_tend, and comp_tend requires
            tracer values
        jacobian units are 1 / s
        """
        jacobian = super().comp_jacobian(time, tracer_vals, processes)
        jacobian += self.comp_jacobian_surf_restore()
        return jacobian

    def comp_jacobian_surf_restore(self):
        """
        compute jacobian of iage restoring term
        jacobian units are 1 / s
        """
        row_ind = np.arange(len(self.ypos))
        rate = 24.0 / 86400.0 * self.depth.delta[0] / 10.0
        data = np.full(len(row_ind), -rate)
        dof = len(self.ypos) * len(self.depth)
        return csr_matrix((data, (row_ind, row_ind)), shape=(dof, dof))

    def apply_precond_jacobian(self, time_range, res_tms, processes):
        """
        apply preconditioner of jacobian of iage fcn

        time_range: length-2 sequence with start and end times of model
        res_tms: TracerModuleState object where results are stored
        mca: (vertical) mixing coefficient, time average
        """

        self_vals = self.get_tracer_vals_all()[0, :]
        shape = self_vals.shape
        self_vals = self_vals.reshape(-1)
        rhs_vals = (1.0 / (time_range[1] - time_range[0])) * self_vals

        time_n = 100
        time_delta = (time_range[1] - time_range[0]) / time_n
        jacobian_mean = self.full_jacobian_sparsity(0.0)
        for time_ind in range(time_n):
            jacobian_mean += (1.0 / time_n) * self.comp_jacobian(
                (time_ind + 0.5) * time_delta, tracer_vals=None, processes=processes
            )

        res_vals = spsolve(jacobian_mean, rhs_vals)

        res_tms.set_tracer_vals_all((res_vals - self_vals).reshape(shape))
