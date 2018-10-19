"""class for representing the state space of a model, and operations on it"""

import logging
import subprocess
import sys
import numpy as np
import netCDF4 as nc

class ModelState:
    """class for representing the state space of a model"""
    def __init__(self, fname):
        self._fname = fname
        self._varnames = ['x']

    def get(self, varname):
        """return component of ModelState corresponding to varname"""
        with nc.Dataset(self._fname, mode='r') as fptr:
            return fptr.variables[varname][:]

    def set(self, varname, val, mode='w'):
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
            valx = self.get(varname)
            valy = other.get(varname)
            valz = valx + scalar * valy
            res.set(varname, val=valz, mode=mode_loc)
            # varnames after first should be appended
            mode_loc = 'a'
        return res

    def add(self, res_fname, other, mode='w'):
        """
        addition operation
        res = self + other
        """
        return self.mult_add(res_fname, 1.0, other, mode)

    def comp_fcn(self, res_fname, solver):
        """
        compute function whose root is being found, store result in res

        skip computation if the corresponding step has been logged in the solver
        re-invoke top-level script and exit, after storing computed result in solver
        """
        logger = logging.getLogger(__name__)

        step = 'comp_fcn'

        if solver.step_logged(step):
            logger.info('%s logged, skipping computation and returning result', step)
            return ModelState(res_fname)

        logger.info('computing fcn and exiting')

        # the following block will eventually get moved to a script that invokes the GCM
        solver.log_step(step)
        iterate_val = self.get('x')
        fcn_val = np.cos(iterate_val)-0.7*iterate_val
        res = ModelState(res_fname)
        res.set('x', fcn_val)

        subprocess.Popen(['/bin/bash', './postrun.sh'])

        sys.exit()
