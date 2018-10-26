#!/usr/bin/env python
"""Newton's method example"""

import argparse
import logging
import os
import sys
import util
from solver import SolverState
from model import ModelState

class NewtonSolver:
    """class for applying Newton's method to approximate the solution of system of equations"""

    def __init__(self, workdir, iterate_fname, resume):
        """initialize Newton solver"""

        # ensure workdir exists
        util.mkdir_exist_okay(workdir)

        self._workdir = workdir
        self.solver_state = SolverState(workdir, 'newton_state.json', resume)
        self.solver_state.log_saved_state()

        # get solver started on an initial run
        if not resume:
            iterate = ModelState(iterate_fname).dump(self._fname('iterate'))
            self.solver_state.set_currstep('init_comp_fcn')
            iterate.comp_fcn(self._fname('fcn'), self.solver_state)

        self._iterate = ModelState(self._fname('iterate'))
        self._fcn = ModelState(self._fname('fcn'))

    def _fname(self, quantity):
        """construct fname corresponding to particular quantity"""
        iteration = self.solver_state.get_iteration()
        return os.path.join(self._workdir, '%s_%02d.nc'%(quantity, iteration))

    def log(self):
        """write the state of the instance to the log"""
        iteration = self.solver_state.get_iteration()
        msg = 'iteration=%02d,iterate'%iteration
        self._iterate.log(msg)
        msg = 'iteration=%02d,fcn'%iteration
        self._fcn.log(msg)

    def converged(self):
        """is solver converged"""
        return self._fcn.converged()

    def _comp_increment(self, iterate, fcn):
        """
        compute Newton's method increment
        (d(fcn) / d(iterate)) (increment) = -fcn
        """
        logger = logging.getLogger(__name__)
        logger.debug('entering')

        logger.info('computing increment')

        self.solver_state.set_currstep('comp_increment')

        delta = 1.0e-6
        iterate_p_delta = iterate + delta
        fcn_p_delta = iterate_p_delta.comp_fcn(self._fname('fcn_p_delta'), self.solver_state)
        dfcn_darg = (fcn_p_delta - fcn) / delta

        logger.debug('returning')
        return (-1.0 / dfcn_darg) * fcn

    def step(self):
        """perform a step of Newton's method"""
        logger = logging.getLogger(__name__)
        logger.debug('entering')

        self.solver_state.set_currstep('calling _comp_increment')
        increment = self._comp_increment(self._iterate, self._fcn)
        self.solver_state.inc_iteration()
        provisional = (self._iterate + increment).dump(self._fname('iterate'))
        self.solver_state.set_currstep('step_comp_fcn')
        provisional.comp_fcn(self._fname('fcn'), self.solver_state)

        logger.debug('returning')

def _parse_args():
    """parse command line arguments"""

    parser = argparse.ArgumentParser(description="Newton's method example")

    parser.add_argument('--workdir', help='directory where files of state vectors are stored',
                        default='work')
    parser.add_argument('--log_fname', help='name of file logging entries are written',
                        default='newton.log')
    parser.add_argument('--iterate_fname', help='name of file with initial iterate',
                        default=None)
    parser.add_argument('--resume', help="resume Newton's method from solver's saved state",
                        action='store_true', default=False)
    parser.add_argument('--rewind', help="rewind last step to recover from error (not implemented)",
                        action='store_true', default=False)

    return parser.parse_args()

def main(args):
    """Newton's method example"""

    workdir = args.workdir

    util.mkdir_exist_okay(workdir)

    filemode = 'a' if args.resume else 'w'
    logging.basicConfig(filename=os.path.join(workdir, args.log_fname),
                        filemode=filemode,
                        format='%(asctime)s:%(process)s:%(funcName)s:%(message)s',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)

    if os.path.exists(os.path.join(workdir, 'KILL')):
        logger.warning('KILL file detected, exiting')
        sys.exit()

    solver = NewtonSolver(workdir=workdir,
                          iterate_fname=args.iterate_fname,
                          resume=args.resume)

    solver.log()

    if solver.converged():
        logger.info('convergence criterion satisfied')
    else:
        solver.step()

if __name__ == '__main__':
    main(_parse_args())
