#!/usr/bin/env python
"""Newton's method example"""

import argparse
import logging
import os
import numpy as np
import file_wrap

def comp_fcn(solver_state):
    """compute function whose root is being found"""
    iterate = solver_state.get_val('iterate')
    fcn = np.cos(iterate)-0.7*iterate
    solver_state.set_val('fcn', fcn)

def comp_increment(solver_state):
    """compute Newton's method increment"""
    iterate = solver_state.get_val('iterate')
    fcn = solver_state.get_val('fcn')
    dfcn_darg = -np.sin(iterate)-0.7
    solver_state.set_val('increment', -fcn/dfcn_darg)

class NewtonState:
    """class for representing the state of the Newton's method solver"""

    def __init__(self, workdir, state_fname):
        """initialize solver state"""
        self.workdir = workdir
        try:
            os.mkdir(self.workdir)
        except FileExistsError:
            pass
        self.state_fname = self.workdir+'/'+state_fname
        self.iter = 0
        self.step_completed = {
            'iterate_set': False,
            'fcn_set': False,
            'increment_set': False
        }

    def inc_iter(self):
        """increment iter"""
        self.iter += 1
        for step_name in self.step_completed:
            self.step_completed[step_name] = False
        self.write()

    def set_val(self, val_name, val):
        """set a parameter in Newton's method"""
        fname = self.workdir+'/'+val_name+'_%02d.nc'%self.iter
        file_wrap.write_var(fname, val_name, val)
        self.step_completed[val_name+'_set'] = True
        self.write()

    def get_val(self, val_name):
        """get a parameter in Newton's method"""
        fname = self.workdir+'/'+val_name+'_%02d.nc'%self.iter
        return file_wrap.read_var(fname, val_name)

    def write(self):
        """write solver state to a file"""
        file_wrap.write_var(self.state_fname, 'iter', self.iter)
        for step_name in self.step_completed:
            val = 1 if self.step_completed[step_name] else 0
            file_wrap.write_var(self.state_fname, step_name, val, mode='a')

    def read(self):
        """read solver state from a file"""
        self.iter = file_wrap.read_var(self.state_fname, 'iter')
        for step_name in self.step_completed:
            self.step_completed[step_name] = file_wrap.read_var(self.state_fname, step_name) == 1

def _parse_args():
    """parse command line arguments"""

    parser = argparse.ArgumentParser(description="Newton's method example")

    parser.add_argument('--workdir', help='directory where files of state vectors are stored',
                        default='work')
    parser.add_argument('--log_fname', help='name of file logging entries are written',
                        default='newton.log')
    parser.add_argument('--solver_state_fname', help='name of file where solver state is stored',
                        default='newton_state.nc')

    return parser.parse_args()

def main(args):
    """Newton's method example"""

    logging.basicConfig(filename=args.workdir+'/'+args.log_fname, filemode='w',
                        format='%(asctime)s:%(message)s',
                        level=logging.INFO)
    logger = logging.getLogger()

    solver_state = NewtonState(workdir=args.workdir, state_fname=args.solver_state_fname)
    iterate = 0.0
    solver_state.set_val('iterate', iterate)

    comp_fcn(solver_state)
    fcn_val = solver_state.get_val('fcn')
    logger.info('iter=%d, iterate=%e, y=%e', solver_state.iter, iterate, fcn_val)

    while np.abs(fcn_val) > 1.0e-10:
        comp_increment(solver_state)
        increment = solver_state.get_val('increment')
        solver_state.inc_iter()
        iterate = iterate + increment
        solver_state.set_val('iterate', iterate)
        comp_fcn(solver_state)
        fcn_val = solver_state.get_val('fcn')
        logger.info('iter=%d, arg=%e, y=%e', solver_state.iter, iterate, fcn_val)

if __name__ == '__main__':
    main(_parse_args())
