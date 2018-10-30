"""class for representing the state space of a model, and operations on it"""

import logging
import os
import subprocess
import sys
import numpy as np
import netCDF4 as nc

class ModelState:
    """class for representing the state space of a model"""

    def __init__(self, tracer_module_names, vals_fname=None):
        self._tracer_module_names = tracer_module_names
        self._module_cnt = len(tracer_module_names)
        self._tracer_modules = np.empty(shape=(self._module_cnt,), dtype=np.object)
        if not vals_fname is None:
            for ind in range(self._module_cnt):
                self._tracer_modules[ind] = TracerModule(tracer_module_names[ind], vals_fname)

    def dump(self, vals_fname):
        """dump ModelState object to a file"""
        with nc.Dataset(vals_fname, mode='w') as fptr:
            for action in ['define', 'write']:
                for ind in range(len(self._tracer_modules)):
                    self._tracer_modules[ind].dump(fptr, action)
        return self

    def log(self, msg):
        """write info of the instance to the log"""
        logger = logging.getLogger(__name__)
        val = self.norm()
        for ind in range(self._module_cnt):
            logger.info('%s,%s,%e', msg, self._tracer_module_names[ind], val[ind])

    # give ModelState operators higher priority than those of numpy
    __array_priority__ = 100

    def __neg__(self):
        """
        unary negation operator
        called to evaluate res = -self
        """
        res = ModelState(self._tracer_module_names)
        for ind in range(self._module_cnt):
            res._tracer_modules[ind] = -self._tracer_modules[ind] # pylint: disable=W0212
        return res

    def __add__(self, other):
        """
        addition operator
        called to evaluate res = self + other
        """
        res = ModelState(self._tracer_module_names)
        if isinstance(other, float):
            for ind in range(self._module_cnt):
                res._tracer_modules[ind] = self._tracer_modules[ind] + other # pylint: disable=W0212
        elif isinstance(other, np.ndarray) and other.shape == (self._module_cnt,):
            for ind in range(self._module_cnt):
                res._tracer_modules[ind] = self._tracer_modules[ind] + other[ind] # pylint: disable=W0212
        elif isinstance(other, ModelState):
            for ind in range(self._module_cnt):
                res._tracer_modules[ind] = self._tracer_modules[ind] + other._tracer_modules[ind] # pylint: disable=W0212
        else:
            return NotImplemented
        return res

    def __radd__(self, other):
        """
        reversed addition operator
        called to evaluate res = other + self
        """
        return self + other

    def __iadd__(self, other):
        """
        inplace addition operator
        called to evaluate self += other
        """
        if isinstance(other, float):
            for ind in range(self._module_cnt):
                self._tracer_modules[ind] += other
        elif isinstance(other, np.ndarray) and other.shape == (self._module_cnt,):
            for ind in range(self._module_cnt):
                self._tracer_modules[ind] += other[ind]
        elif isinstance(other, ModelState):
            for ind in range(self._module_cnt):
                self._tracer_modules[ind] += other._tracer_modules[ind] # pylint: disable=W0212
        else:
            return NotImplemented
        return self

    def __sub__(self, other):
        """
        subtraction operator
        called to evaluate res = self - other
        """
        res = ModelState(self._tracer_module_names)
        if isinstance(other, float):
            for ind in range(self._module_cnt):
                res._tracer_modules[ind] = self._tracer_modules[ind] - other # pylint: disable=W0212
        elif isinstance(other, np.ndarray) and other.shape == (self._module_cnt,):
            for ind in range(self._module_cnt):
                res._tracer_modules[ind] = self._tracer_modules[ind] - other[ind] # pylint: disable=W0212
        elif isinstance(other, ModelState):
            for ind in range(self._module_cnt):
                res._tracer_modules[ind] = self._tracer_modules[ind] - other._tracer_modules[ind] # pylint: disable=W0212
        else:
            return NotImplemented
        return res

    def __isub__(self, other):
        """
        inplace subtraction operator
        called to evaluate self -= other
        """
        if isinstance(other, float):
            for ind in range(self._module_cnt):
                self._tracer_modules[ind] -= other
        elif isinstance(other, np.ndarray) and other.shape == (self._module_cnt,):
            for ind in range(self._module_cnt):
                self._tracer_modules[ind] -= other[ind]
        elif isinstance(other, ModelState):
            for ind in range(self._module_cnt):
                self._tracer_modules[ind] -= other._tracer_modules[ind] # pylint: disable=W0212
        else:
            return NotImplemented
        return self

    def __mul__(self, other):
        """
        multiplication operator
        called to evaluate res = self * other
        """
        res = ModelState(self._tracer_module_names)
        if isinstance(other, float):
            for ind in range(self._module_cnt):
                res._tracer_modules[ind] = self._tracer_modules[ind] * other # pylint: disable=W0212
        elif isinstance(other, np.ndarray) and other.shape == (self._module_cnt,):
            for ind in range(self._module_cnt):
                res._tracer_modules[ind] = self._tracer_modules[ind] * other[ind] # pylint: disable=W0212
        elif isinstance(other, ModelState):
            for ind in range(self._module_cnt):
                res._tracer_modules[ind] = self._tracer_modules[ind] * other._tracer_modules[ind] # pylint: disable=W0212
        else:
            return NotImplemented
        return res

    def __rmul__(self, other):
        """
        reversed multiplication operator
        called to evaluate res = other * self
        """
        return self * other

    def __imul__(self, other):
        """
        inplace multiplication operator
        called to evaluate self *= other
        """
        if isinstance(other, float):
            for ind in range(self._module_cnt):
                self._tracer_modules[ind] *= other
        elif isinstance(other, np.ndarray) and other.shape == (self._module_cnt,):
            for ind in range(self._module_cnt):
                self._tracer_modules[ind] *= other[ind]
        elif isinstance(other, ModelState):
            for ind in range(self._module_cnt):
                self._tracer_modules[ind] *= other._tracer_modules[ind] # pylint: disable=W0212
        else:
            return NotImplemented
        return self

    def __truediv__(self, other):
        """
        division operator
        called to evaluate res = self / other
        """
        res = ModelState(self._tracer_module_names)
        if isinstance(other, float):
            for ind in range(self._module_cnt):
                res._tracer_modules[ind] = (1.0 / other) * self._tracer_modules[ind] # pylint: disable=W0212
        elif isinstance(other, np.ndarray) and other.shape == (self._module_cnt,):
            for ind in range(self._module_cnt):
                res._tracer_modules[ind] = (1.0 / other[ind]) * self._tracer_modules[ind] # pylint: disable=W0212
        elif isinstance(other, ModelState):
            for ind in range(self._module_cnt):
                res._tracer_modules[ind] = self._tracer_modules[ind] / other._tracer_modules[ind] # pylint: disable=W0212
        else:
            return NotImplemented
        return res

    def __rtruediv__(self, other):
        """
        reversed division operator
        called to evaluate res = other / self
        """
        res = ModelState(self._tracer_module_names)
        if isinstance(other, float):
            for ind in range(self._module_cnt):
                res._tracer_modules[ind] = other / self._tracer_modules[ind] # pylint: disable=W0212
        elif isinstance(other, np.ndarray) and other.shape == (self._module_cnt,):
            for ind in range(self._module_cnt):
                res._tracer_modules[ind] = other[ind] / self._tracer_modules[ind] # pylint: disable=W0212
        elif isinstance(other, ModelState):
            for ind in range(self._module_cnt):
                res._tracer_modules[ind] = other._tracer_modules[ind] / self._tracer_modules[ind] # pylint: disable=W0212
        else:
            return NotImplemented
        return res

    def __itruediv__(self, other):
        """
        inplace division operator
        called to evaluate self /= other
        """
        if isinstance(other, float):
            for ind in range(self._module_cnt):
                self._tracer_modules[ind] *= (1.0 / other)
        elif isinstance(other, np.ndarray) and other.shape == (self._module_cnt,):
            for ind in range(self._module_cnt):
                self._tracer_modules[ind] *= (1.0 / other[ind])
        elif isinstance(other, ModelState):
            for ind in range(self._module_cnt):
                self._tracer_modules[ind] /= other._tracer_modules[ind] # pylint: disable=W0212
        else:
            return NotImplemented
        return self

    def dot(self, other):
        """compute dot product of self with other"""
        res = np.empty(shape=(self._module_cnt,))
        for ind in range(self._module_cnt):
            res[ind] = self._tracer_modules[ind].dot(other._tracer_modules[ind]) # pylint: disable=W0212
        return res

    def norm(self):
        """compute l2 norm of self"""
        return np.sqrt(self.dot(self))

    def converged(self):
        """is residual small"""
        return all(self.norm() < 1.0e-10)

    def comp_fcn(self, res_fname, solver):
        """
        compute function whose root is being found, store result in res

        skip computation if the currstep has been logged in the solver
        re-invoke top-level script and exit, after storing computed result in solver
        """
        logger = logging.getLogger(__name__)
        logger.debug('entering, res_fname="%s"', res_fname)

        # value of currstep upon entry
        currstep_in = solver.get_currstep()

        if solver.currstep_logged():
            logger.info('"%s" logged, skipping comp_fcn.sh and returning result', currstep_in)
            return ModelState(self._tracer_module_names, res_fname)

        logger.info('"%s" not logged, invoking comp_fcn.sh and exiting', currstep_in)

        solver.set_currstep('%s invoking comp_fcn.sh'%currstep_in)

        fcn_arg_fname = os.path.join(solver.get_workdir(), 'fcn_arg.nc')
        self.dump(fcn_arg_fname)
        subprocess.Popen(['/bin/bash', './comp_fcn.sh', fcn_arg_fname, res_fname])

        logger.debug('calling exit')
        sys.exit()

    def comp_jacobian_fcn_state_prod(self, res_fname, fcn, direction, solver):
        """
        compute the product of the Jacobian of fcn with a model state direction

        assumes direction is a unit vector
        """
        logger = logging.getLogger(__name__)
        logger.debug('entering, res_fname="%s"', res_fname)

        sigma = 5.0e-4

        if not solver.currstep_logged():
            # temporarily use res_fname to store result of comp_fcn
            # we know here that comp_fcn will not return, because step has not been logged
            iterate_p_sigma = self + sigma * direction
            iterate_p_sigma.comp_fcn(res_fname, solver)

        # retrieve comp_fcn result from res_fname, and proceed with finite difference
        logger.debug('returning')
        return ((ModelState(self._tracer_module_names, res_fname) - fcn) / sigma).dump(res_fname)

class TracerModule:
    """class for representing the a collection of model tracers"""

    def __init__(self, name, vals_fname=None):
        module_varnames = {'x':['x1', 'x2'], 'y':['y']}

        self._name = name
        try:
            self._varnames = module_varnames[name]
        except KeyError:
            raise KeyError('unknown TracerModule name=', name)
        self._dims = {}
        if not vals_fname is None:
            with nc.Dataset(vals_fname, mode='r') as fptr:
                # get dims from first variable
                dimnames0 = fptr.variables[self._varnames[0]].dimensions
                for dimname in dimnames0:
                    self._dims[dimname] = fptr.dimensions[dimname].size
                # all tracers are stored in a single array
                # tracer index is the leading index
                self._vals = np.empty(shape=(len(self._varnames),) + tuple(self._dims.values()))
                # check that all vars have the same dimensions
                for varname in self._varnames:
                    if fptr.variables[varname].dimensions != dimnames0:
                        raise ValueError('not all vars have same dimensions',
                                         name, vals_fname)
                # read values
                ndims = len(self._dims)
                for varind, varname in enumerate(self._varnames):
                    varid = fptr.variables[varname]
                    if ndims == 1:
                        self._vals[varind, :] = varid[:]
                    elif ndims == 2:
                        self._vals[varind, :, :] = varid[:]
                    elif ndims == 3:
                        self._vals[varind, :, :, :] = varid[:]
                    else:
                        raise TypeError('ndims too large', name, vals_fname, ndims)

    def dump(self, fptr, action):
        """perform an action (define or write) of dumping a TracerModule object to an open file"""
        if action == 'define':
            for dimname, dimlen in self._dims.items():
                try:
                    if fptr.dimensions[dimname].size != dimlen:
                        raise ValueError(dimname, 'already exists and has wrong size')
                except KeyError:
                    fptr.createDimension(dimname, dimlen)
            dimnames = tuple(self._dims.keys())
            for varname in self._varnames:
                fptr.createVariable(varname, 'f8', dimensions=dimnames)
        elif action == 'write':
            ndims = len(self._dims)
            for varind, varname in enumerate(self._varnames):
                if ndims == 1:
                    fptr.variables[varname][:] = self._vals[varind, :]
                elif ndims == 2:
                    fptr.variables[varname][:] = self._vals[varind, :, :]
                else:
                    fptr.variables[varname][:] = self._vals[varind, :, :, :]
        else:
            raise ValueError('unknown action=', action)
        return self

    def metadata_only(self):
        """create TracerModule object, copying metadata from self"""
        res = TracerModule(self._name)
        res._dims = self._dims # pylint: disable=W0212
        return res

    def __neg__(self):
        """
        unary negation operator
        called to evaluate res = -self
        """
        res = self.metadata_only()
        res._vals = -self._vals # pylint: disable=W0212
        return res

    def __add__(self, other):
        """
        addition operator
        called to evaluate res = self + other
        """
        res = self.metadata_only()
        if isinstance(other, float):
            res._vals = self._vals + other # pylint: disable=W0212
        elif isinstance(other, TracerModule):
            res._vals = self._vals + other._vals # pylint: disable=W0212
        else:
            return NotImplemented
        return res

    def __iadd__(self, other):
        """
        inplace addition operator
        called to evaluate self += other
        """
        if isinstance(other, float):
            self._vals += other
        elif isinstance(other, TracerModule):
            self._vals += other._vals # pylint: disable=W0212
        else:
            return NotImplemented
        return self

    def __sub__(self, other):
        """
        subtraction operator
        called to evaluate res = self - other
        """
        res = self.metadata_only()
        if isinstance(other, float):
            res._vals = self._vals - other # pylint: disable=W0212
        elif isinstance(other, TracerModule):
            res._vals = self._vals - other._vals # pylint: disable=W0212
        else:
            return NotImplemented
        return res

    def __isub__(self, other):
        """
        inplace subtraction operator
        called to evaluate self -= other
        """
        if isinstance(other, float):
            self._vals -= other
        elif isinstance(other, TracerModule):
            self._vals -= other._vals # pylint: disable=W0212
        else:
            return NotImplemented
        return self

    def __mul__(self, other):
        """
        multiplication operator
        called to evaluate res = self * other
        """
        res = self.metadata_only()
        if isinstance(other, float):
            res._vals = self._vals * other # pylint: disable=W0212
        elif isinstance(other, TracerModule):
            res._vals = self._vals * other._vals # pylint: disable=W0212
        else:
            return NotImplemented
        return res

    def __rmul__(self, other):
        """
        reversed multiplication operator
        called to evaluate res = other * self
        """
        return self * other

    def __imul__(self, other):
        """
        inplace multiplication operator
        called to evaluate self *= other
        """
        if isinstance(other, float):
            self._vals *= other
        elif isinstance(other, TracerModule):
            self._vals *= other._vals # pylint: disable=W0212
        else:
            return NotImplemented
        return self

    def __truediv__(self, other):
        """
        division operator
        called to evaluate res = self / other
        """
        res = self.metadata_only()
        if isinstance(other, float):
            res._vals = (1.0 / other) * self._vals # pylint: disable=W0212
        elif isinstance(other, TracerModule):
            res._vals = self._vals / other._vals # pylint: disable=W0212
        else:
            return NotImplemented
        return res

    def __rtruediv__(self, other):
        """
        reversed division operator
        called to evaluate res = other / self
        """
        res = self.metadata_only()
        if isinstance(other, float):
            res._vals = other / self._vals # pylint: disable=W0212
        elif isinstance(other, TracerModule):
            res._vals = other._vals / self._vals # pylint: disable=W0212
        else:
            return NotImplemented
        return res

    def __itruediv__(self, other):
        """
        inplace division operator
        called to evaluate self /= other
        """
        if isinstance(other, float):
            self._vals *= (1.0 / other)
        elif isinstance(other, TracerModule):
            self._vals /= other._vals # pylint: disable=W0212
        else:
            return NotImplemented
        return self

    def dot(self, other):
        """compute dot product of self with other"""
        ndims = len(self._dims)

        if ndims == 1:
            return np.einsum('ij,ij', self._vals, other._vals) # pylint: disable=W0212
        if ndims == 2:
            return np.einsum('ijk,ijk', self._vals, other._vals) # pylint: disable=W0212
        return np.einsum('ijkl,ijkl', self._vals, other._vals) # pylint: disable=W0212
