"""class for representing the state space of a model, and operations on it"""

import logging
import subprocess
import sys
import numpy as np
import netCDF4 as nc

class ModelState:
    """class for representing the state space of a model"""
    def __init__(self, fname, val=None):
        self._fname = fname
        self._varnames = ['x1', 'x2', 'x3', 'x4']
        if not val is None:
            mode = 'w'
            for varname in self._varnames:
                self.set_val(varname, val, mode)
                mode = 'a'

    def get_val(self, varname):
        """return component of ModelState corresponding to varname"""
        with nc.Dataset(self._fname, mode='r') as fptr:
            return fptr.variables[varname][:]

    def set_val(self, varname, val, mode='w'):
        """
        set component of ModelState corresponding to varname to val

        If varname already exists, overwrite it.
        Otherwise, create it and write to it.
        """
        with nc.Dataset(self._fname, mode=mode) as fptr:
            try:
                varid = fptr.variables[varname]
            except KeyError:
                varid = fptr.createVariable(varname, 'f8')
            varid[:] = val

    def mult_add(self, res_fname, scalar, other, mode='w'):
        """
        multiply and add operation
        res = self + scalar * other
        """
        res = ModelState(res_fname)
        # do not overwrite self, if res_fname is the same as self._fname
        if res_fname == self._fname:
            mode_loc = 'a'
        else:
            mode_loc = mode
        for varname in self._varnames:
            self_val = self.get_val(varname)
            if isinstance(other, float):
                other_val = other
            else:
                other_val = other.get_val(varname)
            res_val = self_val + scalar * other_val
            res.set_val(varname, val=res_val, mode=mode_loc)
            # varnames after first should be appended
            mode_loc = 'a'
        return res

    def add(self, res_fname, other, mode='w'):
        """
        addition operation
        res = self + other
        """
        return self.mult_add(res_fname, 1.0, other, mode)

    def div_diff(self, res_fname, other, delta, mode='w'):
        """
        multiply and add operation
        res = (other - self) / delta
        """
        res = ModelState(res_fname)
        # do not overwrite self, if res_fname is the same as self._fname
        if res_fname == self._fname:
            mode_loc = 'a'
        else:
            mode_loc = mode
        for varname in self._varnames:
            self_val = self.get_val(varname)
            if isinstance(other, float):
                other_val = other
            else:
                other_val = other.get_val(varname)
            if isinstance(delta, float):
                delta_val = delta
            else:
                delta_val = delta.get_val(varname)
            res_val = (other_val - self_val) / delta_val
            res.set_val(varname, val=res_val, mode=mode_loc)
            # varnames after first should be appended
            mode_loc = 'a'
        return res

    def comp_fcn(self, res_fname, solver, step):
        """
        compute function whose root is being found, store result in res

        skip computation if the corresponding step has been logged in the solver
        re-invoke top-level script and exit, after storing computed result in solver
        """
        logger = logging.getLogger(__name__)

        if solver.step_logged(step):
            logger.info('%s logged, skipping computation and returning result', step)
            return ModelState(res_fname)

        logger.info('invoking comp_fcn.sh and exiting')

        solver.log_step(step)

        subprocess.Popen(['/bin/bash', './comp_fcn.sh', self._fname, res_fname])

        sys.exit()

    def converged(self):
        """is residual small"""
        dotprod_sum = 0.0
        for varname in self._varnames:
            val = self.get_val(varname)
            dotprod_sum += val.dot(val)
        return np.sqrt(dotprod_sum) < 1.0e-10
