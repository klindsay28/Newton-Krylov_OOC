"""functions related to horizontal mixing"""

import numpy as np

from .advection import Advection
from .model_process import ModelProcess


class HorizMix(ModelProcess):
    """class related to horizontal mixing"""

    def __init__(self, depth, ypos):

        HorizMix.depth = depth
        HorizMix.ypos = ypos

        self._tend_work = np.zeros((len(depth), len(ypos) + 1))

        # compute (time-invariant) mixing coefficients
        self._mixing_coeff = self._comp_mixing_coeff(mixing_coeff_const=1000.0)

        super().__init__(depth, ypos)

    def _comp_mixing_coeff(self, mixing_coeff_const):
        """
        compute mixing_coeff values
        includes 1/dypos term
        """

        res = np.full((len(self.depth), len(self.ypos) - 1), mixing_coeff_const)

        # increase mixing_coeff to keep grid Peclet number <= 2.0
        peclet_p5 = (
            (0.5 / mixing_coeff_const)
            * self.ypos.delta_mid[:]
            * abs(Advection.vvel[:, 1:-1])
        )
        res *= np.where(peclet_p5 > 1.0, peclet_p5, 1.0)

        return res * self.ypos.delta_mid_r

    def comp_tend(self, time, tracer_vals):
        """single tracer tendency from mixing, with zero flux boundary conditions"""

        # compute horiz_mix fluxes
        self._tend_work[:, 1:-1] = self._mixing_coeff * (
            tracer_vals[:, 1:] - tracer_vals[:, :-1]
        )

        return (self._tend_work[:, 1:] - self._tend_work[:, :-1]) * self.ypos.delta_r

    def get_hist_vars_metadata(self):
        """return dict of process-specific history variable metadata"""

        depth_name = self.depth.axisname
        ypos_edges_name = self.ypos.dump_names["edges"]

        hist_vars_metadata = {}
        hist_vars_metadata["horiz_mixing_coeff"] = {
            "dimensions": (depth_name, ypos_edges_name),
            "attrs": {"long_name": "horizontal mixing coefficient", "units": "m^2 / s"},
        }

        return hist_vars_metadata

    def hist_write(self, sol, fptr_hist):
        """write processs-specific history variables"""

        # write tracer module independent vars
        fptr_hist.variables["horiz_mixing_coeff"][:, 1:-1] = (
            self._mixing_coeff * self.ypos.delta_mid
        )
        # kludge to avoid missing values
        fptr_hist.variables["horiz_mixing_coeff"][:, 0] = fptr_hist.variables[
            "horiz_mixing_coeff"
        ][:, 1]
        fptr_hist.variables["horiz_mixing_coeff"][:, -1] = fptr_hist.variables[
            "horiz_mixing_coeff"
        ][:, -2]

        fptr_hist.sync()
