"""functions related to advection"""

import numpy as np

from .model_process import ModelProcess


class Advection(ModelProcess):
    """class related to advection"""

    def __init__(self, depth, ypos):

        Advection.depth = depth
        Advection.ypos = ypos

        self.gen_vel_field(depth, ypos)

        self._tend_work_y = np.zeros((len(self.depth), len(self.ypos) + 1))
        self._tend_work_z = np.zeros((len(self.depth) + 1, len(self.ypos)))

        super().__init__(depth, ypos)

    @staticmethod
    def gen_vel_field(depth, ypos):
        """generate streamfunction and velocity field"""

        depth_norm = (depth.edges - depth.edges.min()) / (
            depth.edges.max() - depth.edges.min()
        )
        stretch = 2.0
        depth_norm = stretch * depth_norm / (1 + (stretch - 1) * depth_norm)
        depth_fcn = (27.0 / 4.0) * depth_norm * (1.0 - depth_norm) ** 2

        ypos_norm = (ypos.edges - ypos.edges.min()) / (
            ypos.edges.max() - ypos.edges.min()
        )
        ypos_fcn = 4.0 * ypos_norm * (1.0 - ypos_norm)

        stream = np.outer(depth_fcn, ypos_fcn)

        # normalize so that max vvel ~ 0.1 m/s
        vvel = (stream[1:, :] - stream[:-1, :]) * depth.delta_r[:, np.newaxis]
        stream = stream * 0.1 / abs(vvel).max()

        vvel = (stream[1:, :] - stream[:-1, :]) * depth.delta_r[:, np.newaxis]
        wvel = (stream[:, 1:] - stream[:, :-1]) * ypos.delta_r

        Advection.stream = stream
        Advection.vvel = vvel
        Advection.wvel = wvel

    def comp_tend(self, time, tracer_vals):
        """single tracer tendency from advection"""
        self._tend_work_y[:, 1:-1] = 0.5 * (tracer_vals[:, 1:] + tracer_vals[:, :-1])
        self._tend_work_y *= self.vvel

        res = (self._tend_work_y[:, :-1] - self._tend_work_y[:, 1:]) * self.ypos.delta_r

        self._tend_work_z[1:-1, :] = 0.5 * (tracer_vals[1:, :] + tracer_vals[:-1, :])
        self._tend_work_z *= self.wvel

        res += (
            self._tend_work_z[1:, :] - self._tend_work_z[:-1, :]
        ) * self.depth.delta_r[:, np.newaxis]

        return res

    def get_hist_vars_metadata(self):
        """return dict of process-specific history variable metadata"""

        depth_name = self.depth.axisname
        depth_edges_name = self.depth.dump_names["edges"]
        ypos_name = self.ypos.axisname
        ypos_edges_name = self.ypos.dump_names["edges"]

        hist_vars_metadata = {}
        hist_vars_metadata["stream"] = {
            "dimensions": (depth_edges_name, ypos_edges_name),
            "attrs": {"long_name": "velocity streamfunction", "units": "m^2 / s"},
        }
        hist_vars_metadata["vvel"] = {
            "dimensions": (depth_name, ypos_edges_name),
            "attrs": {"long_name": "velocity in ypos direction", "units": "m / s"},
        }
        hist_vars_metadata["wvel"] = {
            "dimensions": (depth_edges_name, ypos_name),
            "attrs": {"long_name": "velocity in depth direction", "units": "m / s"},
        }

        return hist_vars_metadata

    def hist_write(self, sol, fptr_hist):
        """write processs-specific history variables"""

        fptr_hist.variables["stream"][:] = self.stream
        fptr_hist.variables["vvel"][:] = self.vvel
        fptr_hist.variables["wvel"][:] = self.wvel

        fptr_hist.sync()
