#!/usr/bin/env python
"""Newton's method example"""

import logging
import os
import numpy as np
import file_wrap

def comp_fcn(solver_state):
    """function whose root is being found"""
    iterate = solver_state.get_val('iterate')
    fcn = np.cos(iterate)-0.7*iterate
    solver_state.set_val('fcn', fcn)

def comp_increment(solver_state):
    """Newton's method increment"""
    iterate = solver_state.get_val('iterate')
    fcn = solver_state.get_val('fcn')
    dfcn_darg = -np.sin(iterate)-0.7
    solver_state.set_val('increment', -fcn/dfcn_darg)

class NewtonState:
    """class for representing the state of the Newton's method solver"""

    def __init__(self, workdir, fname=None):
        """initialize solver state"""
        self.workdir = workdir
        try:
            os.mkdir(self.workdir)
        except FileExistsError:
            pass
        if fname is None:
            self.iter = 0
        else:
            self.read(fname)

    def inc_iter(self):
        """increment iter"""
        self.iter += 1

    def set_val(self, val_name, val):
        """set a parameter in Newton's method"""
        fname = self.workdir+'/'+val_name+'_%02d.nc'%self.iter
        file_wrap.write_var(fname, val_name, val)

    def get_val(self, val_name):
        """get a parameter in Newton's method"""
        fname = self.workdir+'/'+val_name+'_%02d.nc'%self.iter
        return file_wrap.read_var(fname, val_name)

    def write(self, fname):
        """write solver state to a file"""
        file_wrap.write_var(fname, 'iter', self.iter)

    def read(self, fname):
        """read solver state from a file"""
        self.iter = file_wrap.read_var(fname, 'iter')

def main():
    """Newton's method example"""

    logging.basicConfig(filename='newton.log', filemode='w',
                        format='%(asctime)s:%(message)s',
                        level=logging.INFO)
    logger = logging.getLogger()

    solver_state = NewtonState(workdir='work')
    iterate = 0.0
    solver_state.set_val('iterate', iterate)

    comp_fcn(solver_state)
    fcn_val = solver_state.get_val('fcn')
    logger.info("iter=%d, iterate=%e, y=%e", solver_state.iter, iterate, fcn_val)

    while np.abs(fcn_val) > 1.0e-10:
        comp_increment(solver_state)
        increment = solver_state.get_val('increment')
        solver_state.inc_iter()
        iterate = iterate + increment
        solver_state.set_val('iterate', iterate)
        comp_fcn(solver_state)
        fcn_val = solver_state.get_val('fcn')
        logger.info("iter=%d, arg=%e, y=%e", solver_state.iter, iterate, fcn_val)

if __name__ == '__main__':
    main()
