"""forced subclass of py_driver_2d's TracerModuleState"""

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sp_linalg

from ..utils import eval_expr, gen_forcing_fcn
from .tracer_module_state import TracerModuleState


class forced(TracerModuleState):  # pylint: disable=invalid-name
    """forced tracer module specifics for TracerModuleState"""

    forced_class_vars_set = False

    def __init__(self, tracer_module_name, fname, model_config_obj, depth, ypos):
        super().__init__(tracer_module_name, fname, model_config_obj, depth, ypos)

        self._set_forced_class_vars(model_config_obj.modelinfo)

    def _set_forced_class_vars(self, modelinfo):
        """set (time-invariant) class variables"""

        if forced.forced_class_vars_set:
            return

        forced.params = self.gen_surf_restore_params(modelinfo)

        forced.params.update(self.gen_sms_params(modelinfo))

        # check for param compatibility
        if (
            forced.params["surf_restore_opt"] == "none"
            and forced.params["sms_opt"] != "decay"
        ):
            raise ValueError(
                "forced_sms_opt must be decay if forced_surf_restore_opt == none"
            )

        # generate forcing functions

        if forced.params["surf_restore_opt"] == "file":
            forced.surf_restore_fcn = gen_forcing_fcn(
                modelinfo["forced_surf_restore_fname"],
                modelinfo["forced_surf_restore_varname"],
                [self.ypos.mid],
            )

        if forced.params["sms_opt"] == "file":
            forced.sms_fcn = gen_forcing_fcn(
                modelinfo["forced_sms_fname"],
                modelinfo["forced_sms_varname"],
                [self.depth.mid, self.ypos.mid],
                scalef=forced.params["sms_scalef"],
            )

        forced.forced_class_vars_set = True

    def gen_surf_restore_params(self, modelinfo):
        """Generate dict of surf_restore related parameters."""

        params = {}

        params["surf_restore_opt"] = modelinfo["forced_surf_restore_opt"]
        if params["surf_restore_opt"] not in ["none", "const", "file"]:
            raise ValueError(
                f'unknown forced_surf_restore_opt={params["surf_restore_opt"]}'
            )

        if params["surf_restore_opt"] == "none":
            return params

        surf_restore_rate_10m = 24.0 / 86400.0
        if "forced_surf_restore_rate_10m" in modelinfo:
            surf_restore_rate_10m = eval_expr(modelinfo["forced_surf_restore_rate_10m"])
        # convert 10m restoring rate to rate for surface layer
        params["surf_restore_rate"] = 10.0 / self.depth.delta[0] * surf_restore_rate_10m

        if params["surf_restore_opt"] == "const":
            params["surf_restore_const"] = eval_expr(
                modelinfo["forced_surf_restore_const"]
            )

        return params

    @staticmethod
    def gen_sms_params(modelinfo):
        """Generate dict of sms related parameters."""

        params = {}

        params["sms_opt"] = modelinfo["forced_sms_opt"]
        if params["sms_opt"] not in ["none", "const", "decay", "file"]:
            raise ValueError(f'unknown forced_sms_opt={params["sms_opt"]}')

        if params["sms_opt"] == "none":
            return params

        if params["sms_opt"] == "const":
            params["sms_const"] = eval_expr(modelinfo["forced_sms_const"])

        if params["sms_opt"] == "decay":
            params["sms_decay_rate"] = eval_expr(modelinfo["forced_sms_decay_rate"])

        if params["sms_opt"] == "file":
            params["sms_scalef"] = 1.0
            if "forced_sms_scalef" in modelinfo:
                params["sms_scalef"] = eval_expr(modelinfo["forced_sms_scalef"])
            if "forced_sink_thres" in modelinfo:
                params["sink_thres"] = eval_expr(modelinfo["forced_sink_thres"])

        return params

    def comp_tend(self, time, tracer_vals, processes):
        """
        compute tendency of forced tracers
        tendency units are tr_units / s
        """
        shape = (self.tracer_cnt, len(self.depth), len(self.ypos))
        tracer_tend_vals = super().comp_tend(time, tracer_vals, processes)

        tracer_vals_3d = tracer_vals.reshape(shape)
        tracer_tend_vals_3d = tracer_tend_vals.reshape(shape)

        if self.params["surf_restore_opt"] != "none":
            if self.params["surf_restore_opt"] == "const":
                restore_to_vals = self.params["surf_restore_const"]
            if self.params["surf_restore_opt"] == "file":
                restore_to_vals = self.surf_restore_fcn(time)
            tracer_tend_vals_3d[0, 0, :] += self.params["surf_restore_rate"] * (
                restore_to_vals - tracer_vals_3d[0, 0, :]
            )

        if self.params["sms_opt"] == "const":
            tracer_tend_vals_3d[0, :] += self.params["sms_const"]
        if self.params["sms_opt"] == "decay":
            tracer_tend_vals_3d[0, :] += (
                -self.params["sms_decay_rate"] * tracer_vals_3d[0, :]
            )
        if self.params["sms_opt"] == "file":
            sms = self.sms_fcn(time)
            if "sink_thres" in self.params:
                sink_thres_r = 1.0 / self.params["sink_thres"]
                tmp = sink_thres_r * tracer_vals_3d[0, :]
                sms_scalef = np.where(
                    (sms < 0.0) & (tmp > 0.0) & (tmp < 1.0),
                    # tmp * tmp * (3 - 2 * tmp),
                    tmp,
                    1.0,
                )
                sms *= sms_scalef
            tracer_tend_vals_3d[0, :] += sms

        return tracer_tend_vals

    def comp_jacobian(self, time, tracer_vals, processes):
        """
        compute jacobian of forced tracer tendencies
        jacobian units are 1 / s
        """
        jacobian = super().comp_jacobian(time, tracer_vals, processes)
        if self.params["surf_restore_opt"] != "none":
            jacobian += self.comp_jacobian_surf_restore()
        if self.params["sms_opt"] == "decay":
            jacobian += self.comp_jacobian_sms_decay()
        if self.params["sms_opt"] == "file" and "sink_thres" in self.params:
            jacobian += self.comp_jacobian_sms_file(time, tracer_vals)
        return jacobian

    def comp_jacobian_surf_restore(self):
        """
        compute jacobian of surf restoring term
        jacobian units are 1 / s
        """
        row_ind = np.arange(len(self.ypos))
        dof = len(self.ypos) * len(self.depth)
        data = np.full(len(row_ind), -self.params["surf_restore_rate"])
        return sparse.csr_matrix((data, (row_ind, row_ind)), shape=(dof, dof))

    def comp_jacobian_sms_decay(self):
        """
        compute jacobian of sms decay term
        jacobian units are 1 / s
        """
        block_size = len(self.ypos) * len(self.depth)
        return -self.params["sms_decay_rate"] * sparse.identity(block_size)

    def comp_jacobian_sms_file(self, time, tracer_vals):
        """
        compute jacobian of sms file term
        jacobian units are 1 / s
        """
        sms = self.sms_fcn(time)
        sink_thres_r = 1.0 / self.params["sink_thres"]
        tmp = sink_thres_r * tracer_vals.reshape((len(self.depth), len(self.ypos)))
        d_sms_d_tracer = np.where(
            (sms < 0.0) & (tmp > 0.0) & (tmp < 1.0),
            # sink_thres_r * 6.0 * tmp * (1.0 - tmp) * sms,
            sink_thres_r * sms,
            0.0,
        )
        return sparse.diags(d_sms_d_tracer.reshape(-1))

    def apply_precond_jacobian(self, time_range, res_tms, processes, fptr_precond):
        """
        apply preconditioner of jacobian of comp_fcn

        time_range: length-2 sequence with start and end times of model
        res_tms: TracerModuleState object where results are stored
        """

        self_vals_3d = self.get_tracer_vals_all()
        shape = self_vals_3d.shape
        self_vals = self_vals_3d.reshape(-1)

        time_n = 3
        time_delta = (time_range[1] - time_range[0]) / time_n

        # argument to comp_jacobian
        tracer_vals_3d = np.zeros(self_vals_3d.shape)
        tracer_vals = tracer_vals_3d.reshape(-1)

        precond_time_vals = fptr_precond.variables["time"][:]
        tracer_name = list(self._tracer_module_def["tracers"])[0]
        precond_tracer = fptr_precond.variables[tracer_name]

        mat_id = sparse.identity(self_vals.size)
        mat = sparse.identity(self_vals.size)
        for time_ind in range(time_n):
            time_end = time_range[0] + (time_ind + 1.0) * time_delta
            precond_time_ind = np.argmin(abs(time_end - precond_time_vals))
            tracer_vals_3d[0, :] = precond_tracer[precond_time_ind, :]
            time_mid = time_range[0] + (time_ind + 0.5) * time_delta
            mat_tmp = time_delta * self.comp_jacobian(time_mid, tracer_vals, processes)
            mat *= mat_id - mat_tmp

        mat = mat_id - mat

        res_vals = sp_linalg.spsolve(mat, self_vals)

        res_tms.set_tracer_vals_all((res_vals - self_vals).reshape(shape))
