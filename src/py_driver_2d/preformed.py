"""preformed subclass of py_driver_2d's TracerModuleState"""

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sp_linalg

from .. import utils
from .tracer_module_state import TracerModuleState


class preformed(TracerModuleState):  # pylint: disable=invalid-name
    """preformed tracer module specifics for TracerModuleState"""

    def __init__(self, tracer_module_name, fname, model_config_obj, depth, ypos):

        super().__init__(tracer_module_name, fname, model_config_obj, depth, ypos)

        # restore surface layer at rate of 24 / day over 10 m
        self.surf_restore_rate = 24.0 / 86400.0 * 10.0 / self.depth.delta[0]

        self.surf_restore_fcn = utils.gen_forcing_fcn(
            model_config_obj.modelinfo["surf_restore_fname"],
            model_config_obj.modelinfo["surf_restore_varname"],
            [self.ypos.mid],
        )

    def comp_tend(self, time, tracer_vals, processes):
        """
        compute tendency of preformed tracers
        tendency units are tr_units / s
        """
        shape = (self.tracer_cnt, len(self.depth), len(self.ypos))
        tracer_tend_vals = super().comp_tend(time, tracer_vals, processes)

        tracer_vals_3d = tracer_vals.reshape(shape)
        tracer_tend_vals_3d = tracer_tend_vals.reshape(shape)

        tracer_tend_vals_3d[0, 0, :] += self.surf_restore_rate * (
            self.surf_restore_fcn(time) - tracer_vals_3d[0, 0, :]
        )

        return tracer_tend_vals

    def comp_jacobian(self, time, tracer_vals, processes):
        """
        compute jacobian of preformed tracer tendencies
        jacobian units are 1 / s
        tracer_vals: not used, only present because solve_ivp requires
            comp_jacobian to have same signature as comp_tend, and comp_tend requires
            tracer values
        """
        jacobian = super().comp_jacobian(time, tracer_vals, processes)
        jacobian += self.comp_jacobian_surf_restore()
        return jacobian

    def comp_jacobian_surf_restore(self):
        """
        compute jacobian of preformed restoring term
        jacobian units are 1 / s
        """
        row_ind = np.arange(len(self.ypos))
        dof = len(self.ypos) * len(self.depth)
        data = np.full(len(row_ind), -self.surf_restore_rate)
        return sparse.csr_matrix((data, (row_ind, row_ind)), shape=(dof, dof))

    def apply_precond_jacobian(self, time_range, res_tms, processes):
        """
        apply preconditioner of jacobian of preformed fcn

        time_range: length-2 sequence with start and end times of model
        res_tms: TracerModuleState object where results are stored
        """

        self_vals_3d = self.get_tracer_vals_all()
        shape = self_vals_3d.shape
        self_vals = self_vals_3d.reshape(-1)

        time_n = 3
        time_delta = (time_range[1] - time_range[0]) / time_n

        mat_id = sparse.identity(self_vals.size)
        mat = sparse.identity(self_vals.size)
        for time_ind in range(time_n):
            time = time_range[0] + (time_ind + 0.5) * time_delta
            mat *= mat_id - time_delta * self.comp_jacobian(
                time, self_vals_3d, processes
            )
        mat = mat_id - mat

        res_vals = sp_linalg.spsolve(mat, self_vals)

        res_tms.set_tracer_vals_all((res_vals - self_vals).reshape(shape))
