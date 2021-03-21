"""generic iterative solver infrastructure"""

import logging
import os

from .solver_state import SolverState
from .stats_file import StatsFile
from .utils import mkdir_exist_okay


class SolverBase:
    """
    generic iterative solver class
    base class for NewtonSolver and KrylovSolver
    """

    def __init__(self, solver_name, solverinfo, resume, rewind):
        """initialize Krylov solver"""
        logger = logging.getLogger(__name__)

        logger.debug(
            'solver_name=""%s", resume="%r", rewind="%r"', solver_name, resume, rewind
        )

        self._solver_name = solver_name
        self._solverinfo = solverinfo

        workdir = self._get_workdir()
        logger.debug('%s solver workdir="%s"', solver_name, workdir)
        mkdir_exist_okay(workdir)

        self._solver_state = SolverState(self._solver_name, workdir, resume, rewind)

        self._stats_file = StatsFile(self._solver_name, workdir, self._solver_state)

    def get_iteration(self):
        """get current iteration"""
        return self._solver_state.get_iteration()

    def _get_workdir(self):
        """get name of workdir from solverinfo"""
        key = "_".join([self._solver_name, "workdir"])
        if key not in self._solverinfo:
            key = "workdir"
        return self._solverinfo[key]

    def _fname(self, quantity, iteration=None):
        """construct fname corresponding to particular quantity"""
        if iteration is None:
            iteration = self.get_iteration()
        return os.path.join(self._get_workdir(), "%s_%02d.nc" % (quantity, iteration))

    def _get_rel_tol(self):
        """get solver's relative tolerance from solverinfo"""
        key = "_".join([self._solver_name, "rel_tol"])
        return float(self._solverinfo[key])
