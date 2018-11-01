"""Krylov iterator solver"""

import logging
import os
import numpy as np
import util
from solver import SolverState
from model import ModelState, lin_comb

class KrylovSolver:
    """
    class for applying Krylov method to approximate the solution of system of linear equations

    The specific Krylov method used is Left-Preconditioned GMRES, algorithm 9.4 of 'Iterative
    Methods for Sparse Linear Systems, 2nd Edition', Yousef Saad, available at
    https://www-users.cs.umn.edu/~saad/books.html.

    The solver is applied to A x = -fcn, where A is
    ModelState.comp_jacobian_fcn_state_prod evaluated at iterate.

    Assumes x0 = 0.
    """

    def __init__(self, workdir, tracer_module_names, resume):
        """initialize Krylov solver"""
        logger = logging.getLogger(__name__)
        logger.debug('entering, resume=%r', str(resume))

        # ensure workdir exists
        util.mkdir_exist_okay(workdir)

        self._workdir = workdir
        self._tracer_module_names = tracer_module_names
        self._tracer_module_cnt = len(tracer_module_names)
        self._solver_state = SolverState('Krylov', workdir, 'krylov_state.json', resume)
        self._solver_state.log_saved_state()

        logger.debug('returning')

    def _fname(self, quantity, iteration=None):
        """construct fname corresponding to particular quantity"""
        if iteration is None:
            iteration = self._solver_state.get_iteration()
        return os.path.join(self._workdir, '%s_%02d.nc'%(quantity, iteration))

    def converged(self):
        """is solver converged"""
        return self._solver_state.get_iteration() >= 3

    def solve(self, krylov_solve_res_fname, iterate, fcn):
        """apply Krylov method"""
        logger = logging.getLogger(__name__)
        logger.debug('entering')

        if self._solver_state.get_iteration() == 0:
            # assume x0 = 0, so r0 = M.inv*(rhs - A*x0) = M.inv*rhs = -M.inv*fcn
            precond_fcn = fcn.run_ext_cmd('./comp_precond_jacobian_fcn_state_prod.sh',
                                          self._fname('precond_fcn'), self._solver_state)
            self._solver_state.set_currstep('computing beta and basis_00')
            if not self._solver_state.currstep_logged():
                beta = precond_fcn.norm()
                self._solver_state.set_value_saved_state('beta', beta)
                (-precond_fcn / beta).dump(self._fname('basis'))

        while True:
            j_val = self._solver_state.get_iteration()
            h_mat = np.zeros((self._tracer_module_cnt, j_val+2, j_val+1))
            if j_val > 0:
                h_mat[:, :-1, :-1] = self._solver_state.get_value_saved_state('h_mat')
            basis_j = ModelState(self._tracer_module_names, self._fname('basis'))
            w_raw = iterate.comp_jacobian_fcn_state_prod(fcn, basis_j, self._solver_state)
            w_j = w_raw.run_ext_cmd('./comp_precond_jacobian_fcn_state_prod.sh',
                                    self._fname('w'), self._solver_state)
            h_mat[:, :-1, -1] = w_j.mod_gram_schmidt(j_val+1, self._fname, 'basis')
            h_mat[:, -1, -1] = w_j.norm()
            self._solver_state.set_value_saved_state('h_mat', h_mat)
            w_j /= h_mat[:, -1, -1]

            # solve least-squares minimization problem for each tracer module
            coeff = self.comp_krylov_basis_coeffs(h_mat)

            # construct approximate solution
            res = lin_comb(self._tracer_module_names, coeff, self._fname, 'basis')
            res.dump(self._fname('krylov_res', j_val))

            if self.converged():
                break

            self._solver_state.inc_iteration()
            w_j.dump(self._fname('basis'))

        logger.debug('returning')
        return res.dump(krylov_solve_res_fname)

    def comp_krylov_basis_coeffs(self, h_mat):
        """solve least-squares minimization problem for each tracer module"""
        logger = logging.getLogger(__name__)
        logger.debug('entering')
        coeff = np.zeros((self._tracer_module_cnt, h_mat.shape[-1]))
        lstsq_rhs = np.zeros(h_mat.shape[-2])
        beta = self._solver_state.get_value_saved_state('beta')
        for ind in range(self._tracer_module_cnt):
            lstsq_rhs[0] = beta[ind]
            coeff[ind, :] = np.linalg.lstsq(h_mat[ind, :, :], lstsq_rhs, rcond=None)[0]
            for j_val in range(coeff.shape[-1]):
                logger.info('coeff[%s,%d]=%e', self._tracer_module_names[ind], j_val,
                            coeff[ind, j_val])
        logger.debug('returning')
        return coeff
