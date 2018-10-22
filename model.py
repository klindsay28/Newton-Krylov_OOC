"""class for representing the state space of a model, and operations on it"""

import logging
import os
import subprocess
import sys
import numpy as np
import netCDF4 as nc

class ModelState:
    """class for representing the state space of a model"""

    def __init__(self, fname=None, val=None):
        self._varnames = ['x1', 'x2', 'x3', 'x4']
        if (not fname is None) and (not val is None):
            raise ValueError('fname and val arguments must not both be provided')
        self._vals = dict()
        if not fname is None:
            with nc.Dataset(fname, mode='r') as fptr:
                for varname in self._varnames:
                    self._vals[varname] = fptr.variables[varname][:]
        if not val is None:
            for varname in self._varnames:
                self._vals[varname] = val

    def dump(self, fname, mode='w'):
        """write ModelState object to a file"""
        with nc.Dataset(fname, mode=mode) as fptr:
            for varname, val in self._vals.items():
                try:
                    varid = fptr.variables[varname]
                except KeyError:
                    varid = fptr.createVariable(varname, 'f8')
                varid[:] = val

    def get_val(self, varname):
        """return component of ModelState corresponding to varname"""
        return self._vals[varname]

    def __neg__(self):
        """unary negation operator"""
        res = ModelState()
        for varname, val in self._vals.items():
            res._vals[varname] = -val # pylint: disable=W0212
        return res

    def __add__(self, other):
        """addition operator"""
        res = ModelState()
        for varname, val in self._vals.items():
            if isinstance(other, float):
                res._vals[varname] = val + other # pylint: disable=W0212
            else:
                res._vals[varname] = val + other._vals[varname] # pylint: disable=W0212
        return res

    def __radd__(self, other):
        """reversed addition operator"""
        return self + other

    def __sub__(self, other):
        """subtraction operator"""
        res = ModelState()
        for varname, val in self._vals.items():
            if isinstance(other, float):
                res._vals[varname] = val - other # pylint: disable=W0212
            else:
                res._vals[varname] = val - other._vals[varname] # pylint: disable=W0212
        return res

    def __mul__(self, other):
        """multiplication operator"""
        res = ModelState()
        for varname, val in self._vals.items():
            if isinstance(other, float):
                res._vals[varname] = val * other # pylint: disable=W0212
            else:
                res._vals[varname] = val * other._vals[varname] # pylint: disable=W0212
        return res

    def __rmul__(self, other):
        """reversed multiplication operator"""
        return self * other

    def __truediv__(self, other):
        """division operator"""
        res = ModelState()
        for varname, val in self._vals.items():
            if isinstance(other, float):
                res._vals[varname] = val / other # pylint: disable=W0212
            else:
                res._vals[varname] = val / other._vals[varname] # pylint: disable=W0212
        return res

    def __rtruediv__(self, other):
        """reversed division operator"""
        res = ModelState()
        for varname, val in self._vals.items():
            if isinstance(other, float):
                res._vals[varname] = other / val # pylint: disable=W0212
            else:
                res._vals[varname] = other._vals[varname] / val # pylint: disable=W0212
        return res

    def comp_fcn(self, workdir, res_fname, solver, step):
        """
        compute function whose root is being found, store result in res

        skip computation if the corresponding step has been logged in the solver
        re-invoke top-level script and exit, after storing computed result in solver
        """
        logger = logging.getLogger(__name__)

        if solver.step_logged(step):
            logger.info('%s logged, skipping computation and returning result', step)
            return ModelState(fname=res_fname)

        logger.info('invoking comp_fcn.sh and exiting')

        solver.log_step(step)

        fcn_arg_fname = os.path.join(workdir, 'fcn_arg.nc')
        self.dump(fcn_arg_fname)
        subprocess.Popen(['/bin/bash', './comp_fcn.sh', fcn_arg_fname, res_fname])

        sys.exit()

    def converged(self):
        """is residual small"""
        dotprod_sum = 0.0
        for val in self._vals.values():
            dotprod_sum += val.dot(val)
        return np.sqrt(dotprod_sum) < 1.0e-10
