"""Krylov iterative solver"""

import logging

import numpy as np

from . import model_state_base
from .region_scalars import RegionScalars, to_ndarray, to_region_scalar_ndarray
from .solver_base import SolverBase
from .solver_state import action_step_log_wrap
from .utils import class_name


class KrylovSolver(SolverBase):
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

    def __init__(self, iterate, solverinfo, resume, rewind, hist_fname):
        """initialize Krylov solver"""
        logger = logging.getLogger(__name__)

        super().__init__("Krylov", solverinfo, resume, rewind)

        logger.debug('hist_fname="%s"', hist_fname)

        self._iterate = iterate

        iterate.gen_precond_jacobian(
            hist_fname,
            precond_fname=self._fname("precond", iteration=0),
            solver_state=self._solver_state,
        )

    def converged_flat(self, beta_ndarray, precond_resid_norm_ndarray):
        """
        is solver converged
        precond_resid: preconditioned residuals
        """
        rel_tol = self._get_rel_tol()
        return (self.get_iteration() >= self._get_min_iter()) & (
            precond_resid_norm_ndarray < rel_tol * beta_ndarray
        )

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
        fcn.log_vals("beta", beta)
        caller = class_name(self) + "._solve0"
        (-precond_fcn / beta).dump(self._fname("basis"), caller)
        self._solver_state.set_value_saved_state("beta_ndarray", to_ndarray(beta))

    def solve(self, res_fname, fcn):
        """apply Krylov method"""
        logger = logging.getLogger(__name__)
        logger.debug('res_fname="%s"', res_fname)

        self._solve0(fcn, solver_state=self._solver_state)

        caller = class_name(self) + ".solve"

        while True:
            j_val = self.get_iteration()
            region_cnt = RegionScalars.region_cnt
            h_mat = to_region_scalar_ndarray(
                np.zeros((len(fcn.tracer_modules), j_val + 2, j_val + 1, region_cnt))
            )
            if j_val > 0:
                h_mat[:, :-1, :-1] = to_region_scalar_ndarray(
                    self._solver_state.get_value_saved_state("h_mat_ndarray")
                )
            basis_j = type(self._iterate)(self._fname("basis"))
            w_raw = self._iterate.comp_jacobian_fcn_state_prod(
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
            beta_ndarray = self._solver_state.get_value_saved_state("beta_ndarray")
            coeff_ndarray = _comp_krylov_basis_coeffs(beta_ndarray, h_mat_ndarray)
            self._iterate.log_vals("KrylovCoeff", coeff_ndarray)

            # construct approximate solution
            res = model_state_base.lin_comb(
                type(self._iterate),
                to_region_scalar_ndarray(coeff_ndarray),
                self._fname,
                "basis",
            )
            res.dump(self._fname("krylov_res", j_val), caller)

            precond_resid = model_state_base.lin_comb(
                type(self._iterate),
                to_region_scalar_ndarray(coeff_ndarray),
                self._fname,
                "w",
            )
            precond_resid += type(self._iterate)(self._fname("precond_fcn", 0))
            precond_resid_norm = precond_resid.norm()
            self._iterate.log_vals("precond_resid", precond_resid_norm)

            self._solver_state.inc_iteration()

            if self.converged_flat(beta_ndarray, to_ndarray(precond_resid_norm)).all():
                logger.info("Krylov convergence criterion satisfied")
                break

            w_j.dump(self._fname("basis"), caller)

        return res.dump(res_fname, caller)


def _comp_krylov_basis_coeffs(beta_ndarray, h_mat_ndarray):
    """solve least-squares minimization problem for each tracer module"""
    h_shape = h_mat_ndarray.shape
    coeff_ndarray = np.zeros((h_shape[0], h_shape[2], h_shape[3]))
    lstsq_rhs = np.zeros(h_shape[1])
    for tracer_module_ind in range(h_shape[0]):
        for region_ind in range(h_shape[3]):
            lstsq_rhs[0] = beta_ndarray[tracer_module_ind, region_ind]
            coeff_ndarray[tracer_module_ind, :, region_ind] = np.linalg.lstsq(
                h_mat_ndarray[tracer_module_ind, :, :, region_ind],
                lstsq_rhs,
                rcond=None,
            )[0]
    return coeff_ndarray
