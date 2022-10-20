"""Krylov iterative solver"""

import logging

import numpy as np

from . import model_state_base
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

        super().__init__(
            "Krylov", solverinfo, iterate.model_config_obj.region_cnt, resume, rewind
        )

        logger.debug('hist_fname="%s"', hist_fname)

        self._iterate = iterate

        self._def_solver_stats_vars(
            self.gen_stats_vars_metadata(), self._iterate.tracer_modules
        )

        iterate.gen_precond_jacobian(
            hist_fname,
            precond_fname=self._fname("precond", iteration=0),
            solver_state=self._solver_state,
        )

    @staticmethod
    def gen_stats_vars_metadata():
        """generate metadata for stats vars from Krylov solver"""
        vars_metadata = {}

        vars_metadata["precond_rhs_norm"] = {
            "category": "per_tracer_module",
            "dimensions": ("region",),
            "attrs": {
                "long_name": ("norm of {tracer_module_name} preconditioned rhs"),
                "units": "{tracer_module_units}",
            },
        }

        vars_metadata["precond_resid_norm"] = {
            "category": "per_tracer_module",
            "dimensions": ("iteration", "region"),
            "attrs": {
                "long_name": ("norm of {tracer_module_name} preconditioned residual"),
                "units": "{tracer_module_units}",
            },
        }

        return vars_metadata

    def converged(self, beta, precond_resid_norm):
        """
        is solver converged
        precond_resid: preconditioned residuals
        """
        rel_tol = self._get_rel_tol()
        return (self.get_iteration() >= self._get_min_iter()) & (
            precond_resid_norm < rel_tol * beta
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
        self._put_solver_stats_vars_iteration_independent(precond_rhs_norm=beta)
        caller = class_name(self) + "._solve0"
        (-precond_fcn / beta).dump(self._fname("basis"), caller)
        self._solver_state.set_value_saved_state("beta", beta)

    def solve(self, res_fname, fcn):
        """apply Krylov method"""
        logger = logging.getLogger(__name__)
        logger.debug('res_fname="%s"', res_fname)

        self._solve0(fcn, solver_state=self._solver_state)

        caller = class_name(self) + ".solve"

        while True:
            j_val = self.get_iteration()
            h_mat = np.zeros(
                (
                    len(fcn.tracer_modules),
                    j_val + 2,
                    j_val + 1,
                    fcn.model_config_obj.region_cnt,
                )
            )
            if j_val > 0:
                h_mat[:, :-1, :-1, :] = self._solver_state.get_value_saved_state(
                    "h_mat"
                )
            basis_j = type(self._iterate)(self._fname("basis"))
            w_raw = self._iterate.comp_jacobian_fcn_state_prod(
                fcn, basis_j, self._fname("w_raw"), self._solver_state
            )
            w_j = w_raw.apply_precond_jacobian(
                self._fname("precond", 0), self._fname("w"), self._solver_state
            )
            h_mat[:, :-1, -1, :] = w_j.mod_gram_schmidt(j_val + 1, self._fname, "basis")
            h_mat[:, -1, -1, :] = w_j.norm()
            w_j /= h_mat[:, -1, -1, :]
            self._solver_state.set_value_saved_state("h_mat", h_mat)

            # solve least-squares minimization problem for each tracer module
            beta = self._solver_state.get_value_saved_state("beta")
            coeff = _comp_krylov_basis_coeffs(beta, h_mat)
            self._iterate.log_vals("KrylovCoeff", coeff)

            # construct approximate solution
            res = model_state_base.lin_comb(
                type(self._iterate), coeff, self._fname, "basis"
            )
            res.dump(self._fname("krylov_res", j_val), caller)

            precond_resid = model_state_base.lin_comb(
                type(self._iterate), coeff, self._fname, "w"
            )
            precond_resid += type(self._iterate)(self._fname("precond_fcn", 0))
            precond_resid_norm = precond_resid.norm()
            self._iterate.log_vals("precond_resid", precond_resid_norm)
            self._put_solver_stats_vars(precond_resid_norm=precond_resid_norm)

            self._solver_state.inc_iteration()

            if self.converged(beta, precond_resid_norm).all():
                logger.info("Krylov convergence criterion satisfied")
                break

            w_j.dump(self._fname("basis"), caller)

        return res.dump(res_fname, caller)


def _comp_krylov_basis_coeffs(beta, h_mat):
    """solve least-squares minimization problem for each tracer module"""
    h_shape = h_mat.shape
    coeff = np.zeros((h_shape[0], h_shape[2], h_shape[3]))
    lstsq_rhs = np.zeros(h_shape[1])
    for tracer_module_ind in range(h_shape[0]):
        for region_ind in range(h_shape[3]):
            lstsq_rhs[0] = beta[tracer_module_ind, region_ind]
            coeff[tracer_module_ind, :, region_ind] = np.linalg.lstsq(
                h_mat[tracer_module_ind, :, :, region_ind],
                lstsq_rhs,
                rcond=None,
            )[0]
    return coeff
