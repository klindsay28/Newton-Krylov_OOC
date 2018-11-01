#!/usr/bin/env python
"""Newton's method example"""

import argparse
import configparser
import logging
import os
import sys
import numpy as np
import util
from solver import SolverState
from model import ModelState
from krylov import KrylovSolver

class NewtonSolver:
    """class for applying Newton's method to approximate the solution of system of equations"""

    def __init__(self, workdir, iterate_fname, modelinfo, resume):
        """initialize Newton solver"""
        logger = logging.getLogger(__name__)
        logger.debug('entering, resume=%r', str(resume))

        # ensure workdir exists
        util.mkdir_exist_okay(workdir)

        self._workdir = workdir
        self._solver_state = SolverState(workdir, 'newton_state.json', resume)
        self._solver_state.log_saved_state()
        self._tracer_module_names = modelinfo['tracer_module_names'].split(',')

        # get solver started on an initial run
        if not resume:
            iterate = ModelState(self._tracer_module_names, iterate_fname)
            iterate.dump(self._fname('iterate'))
            iterate.run_ext_cmd('./comp_fcn.sh', self._fname('fcn'), self._solver_state)

        self._iterate = ModelState(self._tracer_module_names, self._fname('iterate'))
        self._fcn = ModelState(self._tracer_module_names, self._fname('fcn'))

        logger.debug('returning')

    def _fname(self, quantity, iteration=None):
        """construct fname corresponding to particular quantity"""
        if iteration is None:
            iteration = self._solver_state.get_iteration()
        return os.path.join(self._workdir, '%s_%02d.nc'%(quantity, iteration))

    def log(self):
        """write the state of the instance to the log"""
        iteration = self._solver_state.get_iteration()
        msg = 'iteration=%02d,iterate'%iteration
        self._iterate.log(msg)
        msg = 'iteration=%02d,fcn'%iteration
        self._fcn.log(msg)

    def converged(self):
        """is residual small"""
        return self._fcn.norm() < 1.0e-7 * self._iterate.norm()

    def _comp_increment(self, iterate, fcn):
        """
        compute Newton's method increment
        (d(fcn) / d(iterate)) (increment) = -fcn
        """
        logger = logging.getLogger(__name__)
        logger.debug('entering')

        complete_step = '_comp_increment complete'

        if self._solver_state.step_logged(complete_step):
            logger.info('"%s" logged, returning result', complete_step)
            return ModelState(self._tracer_module_names, self._fname('increment'))

        logger.info('"%s" not logged, computing increment', complete_step)

        krylov_dir = os.path.join(self._workdir, 'krylov_%02d'%self._solver_state.get_iteration())
        self._solver_state.set_currstep('instantiating KrylovSolver')
        resume = self._solver_state.currstep_logged()
        krylov_solver = KrylovSolver(krylov_dir, self._tracer_module_names, resume)
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

        fcn_norm = self._fcn.norm()

        self._solver_state.set_currstep('computing Armijo factor')
        if not self._solver_state.currstep_logged():
            armijo_ind = 0
            self._solver_state.set_value_saved_state('armijo_ind', armijo_ind)
            armijo_factor = np.where(self.converged(), 0.0, 1.0)
            self._solver_state.set_value_saved_state('armijo_factor', armijo_factor)
        else:
            armijo_ind = self._solver_state.get_value_saved_state('armijo_ind')
            armijo_factor = self._solver_state.get_value_saved_state('armijo_factor')

        while True:

            # compute provisional candidate for next iterate
            prov = (self._iterate + armijo_factor * increment)
            prov_fcn = prov.run_ext_cmd('./comp_fcn.sh', self._fname('prov_fcn', armijo_ind),
                                        self._solver_state)
            prov_fcn_norm = prov_fcn.norm()

            logger.info('Armijo_ind=%d', armijo_ind)
            armijo_cond = True
            alpha = 1.0e-4
            for ind in range(self._iterate._tracer_module_cnt):
                logger.info('"%s":Armijo_factor=%e,fcn_norm=%e,prov_fcn_norm=%e',
                            self._iterate._tracer_module_names[ind], armijo_factor[ind],
                            fcn_norm[ind], prov_fcn_norm[ind])
                if prov_fcn_norm[ind] > (1.0 - alpha * armijo_factor[ind]) * fcn_norm[ind]:
                    armijo_factor[ind] *= 0.5
                    armijo_cond = False

            if armijo_cond:
                logger.info("Armijo condition satisfied")
                break

            logger.info("Armijo condition not satisfied")
            armijo_ind += 1
            self._solver_state.set_value_saved_state('armijo_ind', armijo_ind)
            self._solver_state.set_value_saved_state('armijo_factor', armijo_factor)

        self._solver_state.inc_iteration()

        self._iterate = prov
        self._iterate.dump(self._fname('iterate'))
        self._fcn = prov_fcn
        self._fcn.dump(self._fname('fcn'))

        logger.debug('returning')

def _parse_args():
    """parse command line arguments"""

    parser = argparse.ArgumentParser(description="Newton's method example")

    parser.add_argument('--cfg_fname', help='name of configuration file',
                        default='newton.cfg')

    parser.add_argument('--resume', help="resume Newton's method from solver's saved state",
                        action='store_true', default=False)
    parser.add_argument('--rewind', help="rewind last step to recover from error (not implemented)",
                        action='store_true', default=False)

    return parser.parse_args()

def main(args):
    """Newton's method example"""

    config = configparser.ConfigParser()
    config.read(args.cfg_fname)
    solverinfo = config['solverinfo']

    logging.basicConfig(filename=solverinfo['logging_fname'],
                        filemode='a' if args.resume else 'w',
                        format='%(asctime)s:%(process)s:%(filename)s:%(funcName)s:%(message)s',
                        level=solverinfo['logging_level'])
    logger = logging.getLogger(__name__)

    if os.path.exists('KILL'):
        logger.warning('KILL file detected, exiting')
        sys.exit()

    newton_solver = NewtonSolver(workdir=solverinfo['workdir'],
                                 iterate_fname=solverinfo['init_iterate_fname'],
                                 modelinfo=config['modelinfo'],
                                 resume=args.resume)

    while True:
        if all(newton_solver.converged()):
            logger.info('convergence criterion satisfied')
            break
        newton_solver.step()

if __name__ == '__main__':
    main(_parse_args())
