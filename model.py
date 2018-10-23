"""class for representing the state space of a model, and operations on it"""

import logging
import os
import subprocess
import sys
import numpy as np
import netCDF4 as nc

class ModelState:
    """class for representing the state space of a model"""

    def __init__(self, fname=None):
        self._varnames = ['x', 'y']
        self._dims = dict()
        self._dims_by_var = dict()
        self._vals = dict()
        if not fname is None:
            with nc.Dataset(fname, mode='r') as fptr:
                for dimname, dimid in fptr.dimensions.items():
                    self._dims[dimname] = dimid.size
                for varname in self._varnames:
                    varid = fptr.variables[varname]
                    self._dims_by_var[varname] = varid.dimensions
                    self._vals[varname] = varid[:]

    def dump(self, fname):
        """write ModelState object to a file"""
        with nc.Dataset(fname, mode='w') as fptr:
            for dimname, dimlen in self._dims.items():
                fptr.createDimension(dimname, dimlen)
            for varname, dims in self._dims_by_var.items():
                varid = fptr.createVariable(varname, 'f8', dims)
                varid[:] = self._vals[varname]

    def get_val(self, varname):
        """return component of ModelState corresponding to varname"""
        return self._vals[varname]

    def copy_metadata(self):
        """create ModelState object, copying metadata from self"""
        res = ModelState()
        res._dims = self._dims # pylint: disable=W0212
        res._dims_by_var = self._dims_by_var # pylint: disable=W0212
        return res

    def __neg__(self):
        """unary negation operator"""
        res = self.copy_metadata()
        for varname, val in self._vals.items():
            res._vals[varname] = -val # pylint: disable=W0212
        return res

    def __add__(self, other):
        """addition operator"""
        res = self.copy_metadata()
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
        res = self.copy_metadata()
        for varname, val in self._vals.items():
            if isinstance(other, float):
                res._vals[varname] = val - other # pylint: disable=W0212
            else:
                res._vals[varname] = val - other._vals[varname] # pylint: disable=W0212
        return res

    def __mul__(self, other):
        """multiplication operator"""
        res = self.copy_metadata()
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
        res = self.copy_metadata()
        for varname, val in self._vals.items():
            if isinstance(other, float):
                res._vals[varname] = val / other # pylint: disable=W0212
            else:
                res._vals[varname] = val / other._vals[varname] # pylint: disable=W0212
        return res

    def __rtruediv__(self, other):
        """reversed division operator"""
        res = self.copy_metadata()
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

    def dot(self, other):
        """compute dot product of self with other"""
        res = dict()
        if other is self:
            for varname, val in self._vals.items():
                if isinstance(val, float):
                    res[varname] = val * val
                elif val.ndim == 1:
                    res[varname] = np.einsum('i,i', val, val)
                elif val.ndim == 2:
                    res[varname] = np.einsum('ij,ij', val, val)
                elif val.ndim == 3:
                    res[varname] = np.einsum('ijk,ijk', val, val)
        else:
            for varname, val in self._vals.items():
                if isinstance(val, float):
                    res[varname] = val * other._vals[varname] # pylint: disable=W0212
                elif val.ndim == 1:
                    res[varname] = np.einsum('i,i', val, other._vals[varname]) # pylint: disable=W0212
                elif val.ndim == 2:
                    res[varname] = np.einsum('ij,ij', val, other._vals[varname]) # pylint: disable=W0212
                elif val.ndim == 3:
                    res[varname] = np.einsum('ijk,ijk', val, other._vals[varname]) # pylint: disable=W0212
        return res

    def converged(self):
        """is residual small"""
        return sum(self.dot(self).values()) < 1.0e-10 ** 2
