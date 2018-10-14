#!/usr/bin/env python
"""Newton's method example"""

import argparse
import json
import logging
import os
import subprocess
import sys
import numpy as np
import file_wrap

def comp_fcn(solver_state):
    """
    compute function whose root is being found

    skip computation if result has already been set in solver_state
    re-invoke top-level script and exit, after storing computed result in solver_state
    """
    logger = logging.getLogger(__name__)
    if solver_state.is_set('fcn'):
        logger.info('fcn already computed, skipping computation and returning')
        return
    logger.info('computing fcn')
    iterate = solver_state.get_val('iterate')
    fcn = np.cos(iterate)-0.7*iterate
    solver_state.set_val('fcn', fcn)
    logger.info('re-invoking %s', __file__)
    subprocess.Popen([sys.executable, __file__, '--resume'])
    sys.exit()

def comp_increment(solver_state):
    """
    compute Newton's method increment

    skip computation if result has already been set in solver_state
    re-invoke top-level script and exit, after storing computed result in solver_state
    """
    logger = logging.getLogger(__name__)
    if solver_state.is_set('increment'):
        logger.info('increment already computed, skipping computation and returning')
        return
    logger.info('computing increment')
    iterate = solver_state.get_val('iterate')
    fcn = solver_state.get_val('fcn')
    dfcn_darg = -np.sin(iterate)-0.7
    solver_state.set_val('increment', -fcn/dfcn_darg)
    logger.info('re-invoking %s', __file__)
    subprocess.Popen([sys.executable, __file__, '--resume'])
    sys.exit()

class NewtonState:
    """
    class for representing the state of the Newton's method solver

    There are no public members.

    Private members are:
    _workdir            directory where files of values are located
    _state_fname        name of file where solver state is stored
    _iter               current iteration
    _steps_completed    steps of solver that have been completed in the current iteration
    """

    def __init__(self, workdir, state_fname, resume):
        """initialize solver state"""
        self._workdir = workdir
        self._state_fname = os.path.join(self._workdir, state_fname)
        self._steps_completed = []
        if resume:
            self._read()
        else:
            self._iter = 0

    def inc_iter(self):
        """increment iter, reset steps_completed"""
        self._iter += 1
        self._steps_completed = []
        self._write()

    def get_iter(self):
        """return value of iter"""
        return self._iter

    def is_set(self, val_name):
        """has val_name been set in the current iteration"""
        return val_name+'_set' in self._steps_completed

    def set_val(self, val_name, val):
        """
        set a value in Newton's method

        in current usage, values are: iterate, fcn, increment
        value is written to a value-specific file
        store in steps_completed that the value has been set
        """
        fname = os.path.join(self._workdir, val_name+'_%02d.nc'%self._iter)
        file_wrap.write_var(fname, val_name, val)
        self._steps_completed.append(val_name+'_set')
        self._write()

    def get_val(self, val_name):
        """
        get a parameter in Newton's method

        in current usage, values are: iterate, fcn, increment
        value is read and returned from a value-specific file
        it is an error to get a value that has not been set
        """
        if not self.is_set(val_name):
            raise Exception(val_name+' not set')
        fname = os.path.join(self._workdir, val_name+'_%02d.nc'%self._iter)
        return file_wrap.read_var(fname, val_name)

    def log(self):
        """write solver state to log"""
        logger = logging.getLogger(__name__)
        logger.info('iter=%d', self._iter)
        for step_name in self._steps_completed:
            logger.info('%s completed', step_name)

    def _write(self):
        """
        write solver state to a file

        store the state in a dictionary and write, using JSON
        """
        obj = {'iter':self._iter, 'steps_completed':self._steps_completed}
        with open(self._state_fname, mode='w') as fptr:
            json.dump(obj, fptr, indent=1)

    def _read(self):
        """
        read solver state from a file

        read, using JSON, dictionary form of state, and extract vals
        """
        with open(self._state_fname, mode='r') as fptr:
            obj = json.load(fptr)
        self._iter = obj['iter']
        self._steps_completed = obj['steps_completed']

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

    return parser.parse_args()

def main(args):
    """Newton's method example"""

    try:
        os.mkdir(args.workdir)
    except FileExistsError:
        pass

    filemode = 'a' if args.resume else 'w'
    logging.basicConfig(filename=os.path.join(args.workdir, args.log_fname),
                        filemode=filemode,
                        format='%(asctime)s:%(process)s:%(funcName)s:%(message)s',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)


    solver_state = NewtonState(workdir=args.workdir,
                               state_fname=args.solver_state_fname,
                               resume=args.resume)
    if args.resume:
        iterate = solver_state.get_val('iterate')
    else:
        iterate = 0.0
        solver_state.set_val('iterate', iterate)
    solver_state.log()
    comp_fcn(solver_state)
    fcn_val = solver_state.get_val('fcn')
    logger.info('iter=%d, iterate=%e, y=%e', solver_state.get_iter(), iterate, fcn_val)

    while np.abs(fcn_val) > 1.0e-10:
        comp_increment(solver_state)
        increment = solver_state.get_val('increment')
        solver_state.inc_iter()
        iterate = iterate + increment
        solver_state.set_val('iterate', iterate)
        comp_fcn(solver_state)
        fcn_val = solver_state.get_val('fcn')
        logger.info('iter=%d, iterate=%e, y=%e', solver_state.get_iter(), iterate, fcn_val)

if __name__ == '__main__':
    main(_parse_args())
