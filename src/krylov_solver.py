"""Krylov iterator solver"""

import logging
import os

import numpy as np

from .model_config import get_model_config_attr
from .model_state_base import lin_comb
from .region_scalars import to_ndarray, to_region_scalar_ndarray
from .solver_state import SolverState, action_step_log_wrap
from .utils import class_name, mkdir_exist_okay


class KrylovSolver:
    """
    class for applying a Krylov method to approximate the solution of a system of linear
    equations

    The specific Krylov method used is Left-Preconditioned GMRES, algorithm 9.4 of
    'Iterative Methods for Sparse Linear Systems, 2nd Edition', Yousef Saad, available
    at https://www-users.cs.umn.edu/~saad/books.html.

    The solver is applied to A x = -fcn, where A is
    comp_jacobian_fcn_state_prod evaluated at iterate.

    Assumes x0 = 0.
    """

    def __init__(self, iterate, workdir, resume, rewind, hist_fname):
        """initialize Krylov solver"""
        logger = logging.getLogger(__name__)
        logger.debug(
            'KrylovSolver, workdir="%s", resume="%r", rewind="%r", hist_fname="%s"',
            workdir,
            resume,
            rewind,
            hist_fname,
        )

        # ensure workdir exists
        mkdir_exist_okay(workdir)

        self._workdir = workdir
        self._solver_state = SolverState("Krylov", workdir, resume, rewind)

        iterate.gen_precond_jacobian(
            hist_fname,
            precond_fname=self._fname("precond", iteration=0),
            solver_state=self._solver_state,
        )

    def _fname(self, quantity, iteration=None):
        """construct fname corresponding to particular quantity"""
        if iteration is None:
            iteration = self._solver_state.get_iteration()
        return os.path.join(self._workdir, "%s_%02d.nc" % (quantity, iteration))

    def converged(self):
        """is solver converged"""
        return self._solver_state.get_iteration() >= 3

    @action_step_log_wrap(step="KrylovSolver._solve0", per_iteration=False)
    # pylint: disable=unused-argument
    def _solve0(self, fcn, solver_state):
        """
        steps of solve that are only performed for iteration 0
        This is step 1 of Saad's alogrithm 9.4.
        """
        # assume x0 = 0, so r0 = M.inv*(rhs - A*x0) = M.inv*rhs = -M.inv*fcn
        precond_fcn = fcn.apply_precond_jacobian(
            self._fname("precond", 0), self._fname("precond_fcn"), self._solver_state
        )
        beta = precond_fcn.norm()
        caller = class_name(self) + "._solve0"
        (-precond_fcn / beta).dump(self._fname("basis"), caller)
        self._solver_state.set_value_saved_state("beta_ndarray", to_ndarray(beta))

    def solve(self, res_fname, iterate, fcn):
        """apply Krylov method"""
        logger = logging.getLogger(__name__)
        logger.debug('res_fname="%s"', res_fname)

        self._solve0(fcn, solver_state=self._solver_state)

        caller = class_name(self) + ".solve"

        while True:
            j_val = self._solver_state.get_iteration()
            h_mat = to_region_scalar_ndarray(
                np.zeros(
                    (
                        len(iterate.tracer_modules),
                        j_val + 2,
                        j_val + 1,
                        get_model_config_attr("region_cnt"),
                    )
                )
            )
            if j_val > 0:
                h_mat[:, :-1, :-1] = to_region_scalar_ndarray(
                    self._solver_state.get_value_saved_state("h_mat_ndarray")
                )
            basis_j = type(iterate)(self._fname("basis"))
            w_raw = iterate.comp_jacobian_fcn_state_prod(
                fcn, basis_j, self._fname("w_raw"), self._solver_state
            )
            w_j = w_raw.apply_precond_jacobian(
                self._fname("precond", 0), self._fname("w"), self._solver_state
            )
            h_mat[:, :-1, -1] = w_j.mod_gram_schmidt(j_val + 1, self._fname, "basis")
            h_mat[:, -1, -1] = w_j.norm()
            w_j /= h_mat[:, -1, -1]
            h_mat_ndarray = to_ndarray(h_mat)
            self._solver_state.set_value_saved_state("h_mat_ndarray", h_mat_ndarray)

            # solve least-squares minimization problem for each tracer module
            coeff_ndarray = self.comp_krylov_basis_coeffs(h_mat_ndarray)
            iterate.log_vals("KrylovCoeff", coeff_ndarray)

            # construct approximate solution
            res = lin_comb(
                type(iterate),
                to_region_scalar_ndarray(coeff_ndarray),
                self._fname,
                "basis",
            )
            res.dump(self._fname("krylov_res", j_val), caller)

            if self.converged():
                break

            self._solver_state.inc_iteration()
            w_j.dump(self._fname("basis"), caller)

        return res.dump(res_fname, caller)

    def comp_krylov_basis_coeffs(self, h_mat_ndarray):
        """solve least-squares minimization problem for each tracer module"""
        h_shape = h_mat_ndarray.shape
        coeff_ndarray = np.zeros((h_shape[0], h_shape[2], h_shape[3]))
        lstsq_rhs = np.zeros(h_shape[1])
        beta_ndarray = self._solver_state.get_value_saved_state("beta_ndarray")
        for tracer_module_ind in range(h_shape[0]):
            for region_ind in range(h_shape[3]):
                lstsq_rhs[0] = beta_ndarray[tracer_module_ind, region_ind]
                coeff_ndarray[tracer_module_ind, :, region_ind] = np.linalg.lstsq(
                    h_mat_ndarray[tracer_module_ind, :, :, region_ind],
                    lstsq_rhs,
                    rcond=None,
                )[0]
        return coeff_ndarray
