"""Newton's method iterator solver"""

import logging
import os

import numpy as np

import util

from krylov import KrylovSolver
from model import ModelState, log_vals, to_ndarray, to_region_scalar_ndarray, tracer_module_cnt
from solver import SolverState

class NewtonSolver:
    """class for applying Newton's method to approximate the solution of system of equations"""

    def __init__(self, workdir, modelinfo, resume, rewind):
        """initialize Newton solver"""
        logger = logging.getLogger(__name__)
        logger.debug('entering, resume=%r, rewind=%r', resume, rewind)

        # ensure workdir exists
        util.mkdir_exist_okay(workdir)

        self._workdir = workdir
        self._solver_state = SolverState('Newton', workdir, resume, rewind)

        # get solver started on an initial run
        if not resume:
            ModelState(modelinfo['init_iterate_fname']).dump(self._fname('iterate'))

        self._iterate = ModelState(self._fname('iterate'))
        self._fcn = self._iterate.run_ext_cmd('./comp_fcn.sh', self._fname('fcn'),
                                              self._solver_state)

        logger.debug('returning')

    def _fname(self, quantity, iteration=None):
        """construct fname corresponding to particular quantity"""
        if iteration is None:
            iteration = self._solver_state.get_iteration()
        return os.path.join(self._workdir, '%s_%02d.nc'%(quantity, iteration))

    def log(self):
        """write the state of the instance to the log"""
        iteration = self._solver_state.get_iteration()
        for ind in range(tracer_module_cnt()):
            self._iterate.log('iteration=%02d,iterate'%iteration, ind)
            self._fcn.log('iteration=%02d,fcn'%iteration, ind)

    def converged_flat(self):
        """is residual small"""
        return to_ndarray(self._fcn.norm()) < 1.0e-7 * to_ndarray(self._iterate.norm())

    def _comp_increment(self, iterate, fcn):
        """
        compute Newton's method increment
        (d(fcn) / d(iterate)) (increment) = -fcn
        """
        logger = logging.getLogger(__name__)
        logger.debug('entering')

        complete_step = '_comp_increment complete'

        if self._solver_state.step_logged(complete_step):
            logger.debug('"%s" logged, returning result', complete_step)
            return ModelState(self._fname('increment'))

        logger.debug('"%s" not logged, computing increment', complete_step)

        krylov_dir = os.path.join(self._workdir, 'krylov_%02d'%self._solver_state.get_iteration())
        self._solver_state.set_currstep('instantiating KrylovSolver')
        rewind = self._solver_state.currstep_was_rewound()
        resume = True if rewind else self._solver_state.currstep_logged()
        if not resume:
            self.log()
        krylov_solver = KrylovSolver(krylov_dir, resume, rewind)
        try:
            increment = krylov_solver.solve(self._fname('increment'), iterate, fcn)
        except SystemExit:
            logger.debug('flushing self._solver_state')
            self._solver_state.flush()
            raise
        self._solver_state.set_currstep(complete_step)
        logger.debug('returning')
        return increment

    def step(self):
        """perform a step of Newton's method"""
        logger = logging.getLogger(__name__)
        logger.debug('entering')

        increment = self._comp_increment(self._iterate, self._fcn)

        self._solver_state.set_currstep('computing Armijo factor')
        if not self._solver_state.currstep_logged():
            armijo_ind = 0
            self._solver_state.set_value_saved_state('armijo_ind', armijo_ind)
            armijo_factor_flat = np.where(self.converged_flat(), 0.0, 1.0)
            self._solver_state.set_value_saved_state('armijo_factor_flat', armijo_factor_flat)
        else:
            armijo_ind = self._solver_state.get_value_saved_state('armijo_ind')
            armijo_factor_flat = self._solver_state.get_value_saved_state('armijo_factor_flat')

        while True:
            # compute provisional candidate for next iterate
            armijo_factor = to_region_scalar_ndarray(armijo_factor_flat)
            prov = self._iterate + armijo_factor * increment
            prov_fcn = prov.run_ext_cmd('./comp_fcn.sh', self._fname('prov_fcn', armijo_ind),
                                        self._solver_state)

            logger.info('Armijo_ind=%d', armijo_ind)

            # Determine if Armijo condition is satisfied. Based on Eq. (A.1) of
            # Kelley, C. T., Solving nonlinear equations with Newton's method, 2003.
            fcn_norm = self._fcn.norm()
            prov_fcn_norm = prov_fcn.norm()
            for ind in range(tracer_module_cnt()):
                log_vals('ArmijoFactor', armijo_factor, ind)
                log_vals('fcn_norm', fcn_norm, ind)
                log_vals('prov_fcn_norm', prov_fcn_norm, ind)
            alpha = 1.0e-4
            armijo_cond_flat = to_ndarray(prov_fcn_norm) \
                <= (1.0 - alpha * armijo_factor_flat) * to_ndarray(fcn_norm)

            if armijo_cond_flat.all():
                logger.info("Armijo condition satisfied")
                break

            logger.info("Armijo condition not satisfied")
            armijo_factor_flat = np.where(armijo_cond_flat,
                                          armijo_factor_flat, 0.5*armijo_factor_flat)
            armijo_ind += 1
            self._solver_state.set_value_saved_state('armijo_ind', armijo_ind)
            self._solver_state.set_value_saved_state('armijo_factor_flat', armijo_factor_flat)

        self._solver_state.inc_iteration()

        self._iterate = prov
        self._iterate.dump(self._fname('iterate'))
        self._fcn = prov_fcn
        self._fcn.dump(self._fname('fcn'))

        logger.debug('returning')
