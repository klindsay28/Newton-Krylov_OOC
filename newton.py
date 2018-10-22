#!/usr/bin/env python
"""Newton's method example"""

import argparse
import errno
import logging
import os
import sys
from solver import SolverState
from model import ModelState

class NewtonSolver:
    """class for applying Newton's method to approximate the solution of system of equations"""

    def __init__(self, workdir, solver_state_fname, resume):
        """initialize Newton solver"""

        self._workdir = workdir
        self.solver_state = SolverState(workdir, solver_state_fname, resume)
        self.solver_state.log_saved_state()

        # get solver started on an initial run
        if not resume:
            iterate = ModelState(val=0.0)
            iterate.dump(self._fname('iterate'))
            iterate.comp_fcn(self._workdir, self._fname('fcn'), self.solver_state, 'comp_fcn')

    def _fname(self, quantity):
        """construct fname corresponding to particular quantity"""
        iteration = self.solver_state.get_iteration()
        return os.path.join(self._workdir, '%s_%02d.nc'%(quantity, iteration))

    def log(self):
        """write the state of the instance to the log"""
        iteration = self.solver_state.get_iteration()
        iterate_val = ModelState(self._fname('iterate')).get_val('x1')
        fcn_val = ModelState(self._fname('fcn')).get_val('x1')
        logger = logging.getLogger(__name__)
        logger.info('iteration=%d, iterate=%e, fcn=%e', iteration, iterate_val, fcn_val)

    def converged(self):
        """is solver converged"""
        return ModelState(self._fname('fcn')).converged()

    def _comp_increment(self, iterate, fcn):
        """
        compute Newton's method increment
        """
        logger = logging.getLogger(__name__)

        logger.info('computing increment')

        step = 'div_diff'

        delta = 1.0e-6
        iterate_p_delta = iterate + delta
        fcn_p_delta = iterate_p_delta.comp_fcn(self._workdir, self._fname('fcn_p_delta'),
                                               self.solver_state, step)
        dfcn_darg = (fcn_p_delta - fcn) / delta
        return (-1.0) * fcn / dfcn_darg

    def step(self):
        """perform a step of Newton's method"""
        iterate = ModelState(self._fname('iterate'))
        fcn = ModelState(self._fname('fcn'))
        increment = self._comp_increment(iterate, fcn)
        self.log()
        self.solver_state.inc_iteration()
        provisional = iterate + increment
        provisional.dump(self._fname('iterate'))
        provisional.comp_fcn(self._workdir, self._fname('fcn'), self.solver_state, 'comp_fcn')

def _parse_args():
    """parse command line arguments"""

    parser = argparse.ArgumentParser(description="Newton's method example")

    parser.add_argument('--workdir', help='directory where files of state vectors are stored',
                        default='work')
    parser.add_argument('--log_fname', help='name of file logging entries are written',
                        default='newton.log')
    parser.add_argument('--solver_state_fname', help='name of file where solver state is stored',
                        default='newton_state.json')
    parser.add_argument('--resume', help="resume Newton's method from solver's saved state",
                        action='store_true', default=False)
    parser.add_argument('--rewind', help="rewind last step to recover from error (not implemented)",
                        action='store_true', default=False)

    return parser.parse_args()

def main(args):
    """Newton's method example"""

    try:
        os.mkdir(args.workdir)
    except OSError as err:
        if err.errno == errno.EEXIST:
            pass
        else:
            raise

    filemode = 'a' if args.resume else 'w'
    logging.basicConfig(filename=os.path.join(args.workdir, args.log_fname),
                        filemode=filemode,
                        format='%(asctime)s:%(process)s:%(funcName)s:%(message)s',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)

    if os.path.exists(os.path.join(args.workdir, 'KILL')):
        logger.warning('KILL file detected, exiting')
        sys.exit()

    solver = NewtonSolver(workdir=args.workdir,
                          solver_state_fname=args.solver_state_fname,
                          resume=args.resume)

    if solver.converged():
        solver.log()
        logger.info('convergence criterion satisfied')
    else:
        solver.step()

if __name__ == '__main__':
    main(_parse_args())
