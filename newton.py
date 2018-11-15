"""Newton's method iterator solver"""

import logging
import os

import numpy as np

import util

from krylov import KrylovSolver
from model import ModelState, log_vals, to_ndarray, to_region_scalar_ndarray, tracer_module_cnt
from model import shadow_tracers_on
from solver import SolverState

class NewtonSolver:
    """class for applying Newton's method to approximate the solution of system of equations"""

    def __init__(self, solverinfo, modelinfo, resume, rewind):
        """initialize Newton solver"""
        logger = logging.getLogger(__name__)
        logger.debug('entering, resume=%r, rewind=%r', resume, rewind)

        # ensure workdir exists
        workdir = solverinfo['workdir']
        util.mkdir_exist_okay(workdir)

        self._solverinfo = solverinfo
        self._modelinfo = modelinfo
        self._solver_state = SolverState('Newton', workdir, resume, rewind)

        # get solver started on an initial run
        if not resume:
            iterate = ModelState(modelinfo['init_iterate_fname'])
            iterate.copy_real_tracers_to_shadow_tracers().dump(self._fname('iterate'))

        self._iterate = ModelState(self._fname('iterate'))
        if self._solver_state.get_iteration() == 0:
            self._fcn = self._iterate.run_ext_cmd(self._modelinfo['newton_fcn_script'],
                                                  self._fname('fcn'), self._solver_state)
        else:
            self._fcn = ModelState(self._fname('fcn'))

        logger.debug('returning')

    def _fname(self, quantity, iteration=None):
        """construct fname corresponding to particular quantity"""
        if iteration is None:
            iteration = self._solver_state.get_iteration()
        return os.path.join(self._solverinfo['workdir'], '%s_%02d.nc' % (quantity, iteration))

    def log(self, iterate, fcn, msg=None):
        """write the state of the instance to the log"""
        iteration = self._solver_state.get_iteration()
        if msg is None:
            iteration_p_msg = 'iteration=%02d' % iteration
        else:
            iteration_p_msg = 'iteration=%02d,%s' % (iteration, msg)
        for ind in range(tracer_module_cnt()):
            iterate.log('%s,iterate' % iteration_p_msg, ind)
            fcn.log('%s,fcn' % iteration_p_msg, ind)

    def converged_flat(self):
        """is residual small"""
        rel_tol = self._solverinfo.getfloat('newton_rel_tol')
        return to_ndarray(self._fcn.norm()) < rel_tol * to_ndarray(self._iterate.norm())

    def _comp_increment(self):
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

        krylov_dir = os.path.join(self._solverinfo['workdir'],
                                  'krylov_%02d' % self._solver_state.get_iteration())
        self._solver_state.set_currstep('instantiating KrylovSolver')
        rewind = self._solver_state.currstep_was_rewound()
        resume = True if rewind else self._solver_state.currstep_logged()
        if not resume:
            self.log(self._iterate, self._fcn)
        krylov_solver = KrylovSolver(self._modelinfo, krylov_dir, resume, rewind)
        try:
            increment = krylov_solver.solve(self._fname('increment'), self._iterate, self._fcn)
        except SystemExit:
            logger.debug('flushing self._solver_state')
            self._solver_state.flush()
            raise
        self._solver_state.set_currstep(complete_step)
        logger.debug('returning')
        return increment

    def _comp_next_iterate(self, increment):
        """compute next Newton iterate"""
        logger = logging.getLogger(__name__)
        logger.debug('entering')

        self._solver_state.set_currstep('computing Armijo factor')
        if not self._solver_state.currstep_logged():
            armijo_ind = 0
            self._solver_state.set_value_saved_state('armijo_ind', armijo_ind)
            armijo_factor_flat = np.where(self.converged_flat(), 0.0, 1.0)
            self._solver_state.set_value_saved_state('armijo_factor_flat', armijo_factor_flat)
        else:
            armijo_ind = self._solver_state.get_value_saved_state('armijo_ind')
            armijo_factor_flat = self._solver_state.get_value_saved_state('armijo_factor_flat')

        complete_step = '_comp_next_iterate complete'

        if self._solver_state.step_logged(complete_step):
            logger.debug('"%s" logged, returning result', complete_step)
            return ModelState(self._fname('prov_Armijo_%02d' % armijo_ind))

        logger.debug('"%s" not logged, computing next iterate', complete_step)

        while True:
            # compute provisional candidate for next iterate
            armijo_factor = to_region_scalar_ndarray(armijo_factor_flat)
            prov = self._iterate + armijo_factor * increment
            prov.dump(self._fname('prov_Armijo_%02d' % armijo_ind))
            prov_fcn = prov.run_ext_cmd(self._modelinfo['newton_fcn_script'],
                                        self._fname('prov_fcn_Armijo_%02d' % armijo_ind),
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
            armijo_cond_flat = (armijo_factor_flat == 0.0) | (to_ndarray(prov_fcn_norm) \
                <= (1.0 - alpha * armijo_factor_flat) * to_ndarray(fcn_norm))

            if armijo_cond_flat.all():
                logger.info("Armijo condition satisfied")
                self._solver_state.set_currstep(complete_step)
                logger.debug('returning')
                return prov

            logger.info("Armijo condition not satisfied")
            armijo_factor_flat = np.where(armijo_cond_flat,
                                          armijo_factor_flat, 0.5*armijo_factor_flat)
            armijo_ind += 1
            self._solver_state.set_value_saved_state('armijo_ind', armijo_ind)
            self._solver_state.set_value_saved_state('armijo_factor_flat', armijo_factor_flat)

            if armijo_ind > 10:
                raise RuntimeError('Armijo_ind exceeds limit')

    def step(self):
        """perform a step of Newton's method"""
        logger = logging.getLogger(__name__)
        logger.debug('entering')

        increment = self._comp_increment()

        prov = self._comp_next_iterate(increment)

        self._solver_state.set_currstep('running fp iterations')
        if not self._solver_state.currstep_logged():
            fp_iter = 0
            self._solver_state.set_value_saved_state('fp_iter', fp_iter)
            prov.copy_shadow_tracers_to_real_tracers()
            prov.dump(self._fname('prov_fp_%02d' % fp_iter))
            # Evaluate comp_fcn after copying shadow tracers to their real counterparts. If no
            # shadow tracers are on, then this is the same as the final comp_fcn result from Armijo
            # iterations.
            if shadow_tracers_on():
                prov.run_ext_cmd(self._modelinfo['newton_fcn_script'],
                                 self._fname('prov_fcn_fp_%02d' % fp_iter), self._solver_state)
            else:
                armijo_ind = self._solver_state.get_value_saved_state('armijo_ind')
                prov_fcn = ModelState(self._fname('prov_fcn_Armijo_%02d' % armijo_ind))
                prov_fcn.dump(self._fname('prov_fcn_fp_%02d' % fp_iter))
        else:
            fp_iter = self._solver_state.get_value_saved_state('fp_iter')
            prov = ModelState(self._fname('prov_fp_%02d' % fp_iter))
            prov_fcn = ModelState(self._fname('prov_fcn_fp_%02d' % fp_iter))

        while fp_iter < self._solverinfo.getint('post_newton_fp_iter'):
            self._solver_state.set_currstep('performing fp iteration %02d' % fp_iter)
            if not self._solver_state.currstep_logged():
                if fp_iter == 0:
                    self.log(prov, prov_fcn, 'pre-fp_iter')
                prov += prov_fcn
                prov.copy_shadow_tracers_to_real_tracers()
                prov.dump(self._fname('prov_fp_%02d' % (fp_iter+1)))
            else:
                prov = ModelState(self._fname('prov_fp_%02d' % (fp_iter+1)))
            prov_fcn = prov.run_ext_cmd(self._modelinfo['newton_fcn_script'],
                                        self._fname('prov_fcn_fp_%02d'% (fp_iter+1)),
                                        self._solver_state)
            fp_iter += 1
            self._solver_state.set_value_saved_state('fp_iter', fp_iter)
            self.log(prov, prov_fcn, 'fp_iter=%02d' % fp_iter)

        self._solver_state.inc_iteration()

        self._iterate = prov
        self._iterate.dump(self._fname('iterate'))
        self._fcn = prov_fcn
        self._fcn.dump(self._fname('fcn'))

        if self._solver_state.get_iteration() >= self._solverinfo.getint('newton_max_iter'):
            raise RuntimeError('number of maximum Newton iterations exceeded')

        logger.debug('returning')
