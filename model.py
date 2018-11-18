"""class for representing the state space of a model, and operations on it"""

import importlib
import json
import logging
import os
import subprocess
import sys

import numpy as np
from netCDF4 import Dataset

# model static variables
_model_static_vars = None

# functions to commonly accessed vars in _model_static_vars
def tracer_module_cnt():
    """return number of tracer modules"""
    return len(_model_static_vars.tracer_module_names)

def tracer_module_names():
    """return list of tracer module names"""
    return _model_static_vars.tracer_module_names

def tracer_names():
    """return list of all tracer names"""
    return _model_static_vars.tracer_names

def region_cnt():
    """return number of regions specified by region_mask"""
    return _model_static_vars.region_cnt

def shadow_tracers_on():
    """are any shadow tracers being run"""
    for tracer_module_name in _model_static_vars.tracer_module_names:
        if _model_static_vars.tracer_module_defs[tracer_module_name]['shadow_tracers']:
            return True
    return False

################################################################################

class ModelStaticVars:
    """class to hold static vars"""

    def __init__(self, modelinfo, cfg_fname=None, lvl=logging.DEBUG):
        logger = logging.getLogger(__name__)
        logger.debug('entering, cfg_fname="%s"', cfg_fname)

        # import NewtonFcn and related settings
        mod_import = importlib.import_module(modelinfo['newton_fcn_modname'])
        self.fcn_lib_file = mod_import.__file__
        fcnlib = mod_import.NewtonFcn()
        self.cmd_fcn = {}
        self.cmd_fcn['comp_fcn'] = fcnlib.comp_fcn
        self.cmd_fcn['apply_precond_jacobian'] = fcnlib.apply_precond_jacobian

        self.cmd_ext = {}
        self.cmd_ext['comp_fcn'] = modelinfo.getboolean('comp_fcn_ext')
        self.cmd_ext['apply_precond_jacobian'] = modelinfo.getboolean('apply_precond_jacobian_ext')

        # extract tracer_module_names from modelinfo config object
        self.tracer_module_names = modelinfo['tracer_module_names'].split(',')

        # extract tracer_module_defs from modelinfo config object
        fname = modelinfo['tracer_module_defs_fname']
        logger.log(lvl, 'reading tracer_module_defs from %s', fname)
        with open(fname, mode='r') as fptr:
            self.tracer_module_defs = json.load(fptr)

        self._check_shadow_tracers(lvl)

        # extracer tracer_names in use from tracer_module_defs
        self.tracer_names = []
        for tracer_module_name in self.tracer_module_names:
            tracer_module_def = self.tracer_module_defs[tracer_module_name]
            self.tracer_names.extend(tracer_module_def['tracer_names'])

        # extract grid_weight from modelinfo config object
        fname = modelinfo['grid_weight_fname']
        varname = modelinfo['grid_weight_varname']
        logger.log(lvl, 'reading %s from %s for grid_weight', varname, fname)
        with Dataset(fname, mode='r') as fptr:
            fptr.set_auto_mask(False)
            grid_weight_no_region_dim = fptr.variables[varname][:]

        # extract region_mask from modelinfo config object
        fname = modelinfo['region_mask_fname']
        varname = modelinfo['region_mask_varname']
        if not fname == 'None' and not varname == 'None':
            logger.log(lvl, 'reading %s from %s for region_mask', varname, fname)
            with Dataset(fname, mode='r') as fptr:
                fptr.set_auto_mask(False)
                self.region_mask = fptr.variables[varname][:]
                if self.region_mask.shape != grid_weight_no_region_dim.shape:
                    raise RuntimeError('region_mask and grid_weight must have the same shape')
        else:
            self.region_mask = np.ones_like(grid_weight_no_region_dim, dtype=np.int32)

        # enforce that region_mask and grid_weight and both 0 where one of them is
        self.region_mask = np.where(grid_weight_no_region_dim == 0.0, 0, self.region_mask)
        grid_weight_no_region_dim = np.where(self.region_mask == 0, 0.0, grid_weight_no_region_dim)

        self.region_cnt = self.region_mask.max()

        # add region dimension to grid_weight and normalize
        self.grid_weight = np.empty(shape=(self.region_cnt,) + grid_weight_no_region_dim.shape)
        for region_ind in range(self.region_cnt):
            self.grid_weight[region_ind, :] = np.where(self.region_mask == region_ind+1,
                                                       grid_weight_no_region_dim, 0.0)
            # normalize grid_weight so that its sum is 1.0 over each region
            self.grid_weight[region_ind, :] *= 1.0 / np.sum(self.grid_weight[region_ind, :])

        # cfg_fname is stored so that it can be passed to cmd in run_cmd
        # it is not needed in stand-alone usage of model.py
        self.cfg_fname = cfg_fname

        # store contents in module level var, to enable use elsewhere
        global _model_static_vars # pylint: disable=W0603
        _model_static_vars = self

        logger.debug('returning')

    def _check_shadow_tracers(self, lvl):
        """Confirm that tracers specified in shadow_tracers are also in tracer_names."""
        # This check is done for all entries in tracer_module_defs, whether they are being used or
        # not. If a tracer module does not have any shadow tracers, add an empty shadow_tracer
        # dictionary to tracer_module_defs, to ease subsequent coding.
        logger = logging.getLogger(__name__)
        for tracer_module_name, tracer_module_def in self.tracer_module_defs.items():
            try:
                shadow_tracers = tracer_module_def['shadow_tracers']
            except KeyError:
                shadow_tracers = {}
                tracer_module_def['shadow_tracers'] = {}
                logger.log(lvl, 'tracer module %s has no shadow tracers', tracer_module_name)
            tracer_names_loc = tracer_module_def['tracer_names']
            for shadow_tracer_name, real_tracer_name in shadow_tracers.items():
                shadow_tracer_ind = tracer_names_loc.index(shadow_tracer_name)
                real_tracer_ind = tracer_names_loc.index(real_tracer_name)
                logger.log(lvl, 'tracer module %s has %s (ind=%d) as a shadow for %s (ind=%d)',
                           tracer_module_name, shadow_tracer_name, shadow_tracer_ind,
                           real_tracer_name, real_tracer_ind)

################################################################################

class ModelState:
    """class for representing the state space of a model"""

    # give ModelState operators higher priority than those of numpy
    __array_priority__ = 100

    def __init__(self, vals_fname=None):
        if _model_static_vars is None:
            msg = '_model_static_vars is None' \
                  ', ModelStaticVars.__init__ must be called before ModelState.__init__'
            raise RuntimeError(msg)
        if vals_fname is not None:
            self._tracer_modules = np.empty(shape=(tracer_module_cnt(),), dtype=np.object)
            for ind in range(tracer_module_cnt()):
                self._tracer_modules[ind] = TracerModuleState(
                    tracer_module_names()[ind], vals_fname=vals_fname)

    def dump(self, vals_fname):
        """dump ModelState object to a file"""
        with Dataset(vals_fname, mode='w') as fptr:
            for action in ['define', 'write']:
                for ind in range(tracer_module_cnt()):
                    self._tracer_modules[ind].dump(fptr, action)
        return self

    def log(self, msg=None, ind=None):
        """write info of the instance to the log"""
        if ind is None:
            for ind_tmp in range(tracer_module_cnt()):
                self.log(msg, ind_tmp)
            return

        for prefix, vals in {'mean':self.mean(), 'norm':self.norm()}.items():
            msg_full = prefix if msg is None else msg+','+prefix
            log_vals(msg_full, vals, ind)

    def copy(self):
        """return a copy of self"""
        res = ModelState()
        res._tracer_modules = np.copy(self._tracer_modules) # pylint: disable=W0212
        return res

    def __neg__(self):
        """
        unary negation operator
        called to evaluate res = -self
        """
        res = ModelState()
        res._tracer_modules = -self._tracer_modules # pylint: disable=W0212
        return res

    def __add__(self, other):
        """
        addition operator
        called to evaluate res = self + other
        """
        res = ModelState()
        if isinstance(other, float):
            res._tracer_modules = self._tracer_modules + other # pylint: disable=W0212
        elif isinstance(other, np.ndarray) and other.shape == (tracer_module_cnt(),):
            res._tracer_modules = self._tracer_modules + other # pylint: disable=W0212
        elif isinstance(other, ModelState):
            res._tracer_modules = self._tracer_modules + other._tracer_modules # pylint: disable=W0212
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
            self._tracer_modules += other
        elif isinstance(other, np.ndarray) and other.shape == (tracer_module_cnt(),):
            self._tracer_modules += other
        elif isinstance(other, ModelState):
            self._tracer_modules += other._tracer_modules # pylint: disable=W0212
        else:
            return NotImplemented
        return self

    def __sub__(self, other):
        """
        subtraction operator
        called to evaluate res = self - other
        """
        res = ModelState()
        if isinstance(other, float):
            res._tracer_modules = self._tracer_modules - other # pylint: disable=W0212
        elif isinstance(other, np.ndarray) and other.shape == (tracer_module_cnt(),):
            res._tracer_modules = self._tracer_modules - other # pylint: disable=W0212
        elif isinstance(other, ModelState):
            res._tracer_modules = self._tracer_modules - other._tracer_modules # pylint: disable=W0212
        else:
            return NotImplemented
        return res

    def __isub__(self, other):
        """
        inplace subtraction operator
        called to evaluate self -= other
        """
        if isinstance(other, float):
            self._tracer_modules -= other
        elif isinstance(other, np.ndarray) and other.shape == (tracer_module_cnt(),):
            self._tracer_modules -= other
        elif isinstance(other, ModelState):
            self._tracer_modules -= other._tracer_modules # pylint: disable=W0212
        else:
            return NotImplemented
        return self

    def __mul__(self, other):
        """
        multiplication operator
        called to evaluate res = self * other
        """
        res = ModelState()
        if isinstance(other, float):
            res._tracer_modules = self._tracer_modules * other # pylint: disable=W0212
        elif isinstance(other, np.ndarray) and other.shape == (tracer_module_cnt(),):
            res._tracer_modules = self._tracer_modules * other # pylint: disable=W0212
        elif isinstance(other, ModelState):
            res._tracer_modules = self._tracer_modules * other._tracer_modules # pylint: disable=W0212
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
            self._tracer_modules *= other
        elif isinstance(other, np.ndarray) and other.shape == (tracer_module_cnt(),):
            self._tracer_modules *= other
        elif isinstance(other, ModelState):
            self._tracer_modules *= other._tracer_modules # pylint: disable=W0212
        else:
            return NotImplemented
        return self

    def __truediv__(self, other):
        """
        division operator
        called to evaluate res = self / other
        """
        res = ModelState()
        if isinstance(other, float):
            res._tracer_modules = self._tracer_modules * (1.0 / other) # pylint: disable=W0212
        elif isinstance(other, np.ndarray) and other.shape == (tracer_module_cnt(),):
            res._tracer_modules = self._tracer_modules * (1.0 / other) # pylint: disable=W0212
        elif isinstance(other, ModelState):
            res._tracer_modules = self._tracer_modules / other._tracer_modules # pylint: disable=W0212
        else:
            return NotImplemented
        return res

    def __rtruediv__(self, other):
        """
        reversed division operator
        called to evaluate res = other / self
        """
        res = ModelState()
        if isinstance(other, float):
            res._tracer_modules = other / self._tracer_modules # pylint: disable=W0212
        elif isinstance(other, np.ndarray) and other.shape == (tracer_module_cnt(),):
            res._tracer_modules = other / self._tracer_modules # pylint: disable=W0212
        else:
            return NotImplemented
        return res

    def __itruediv__(self, other):
        """
        inplace division operator
        called to evaluate self /= other
        """
        if isinstance(other, float):
            self._tracer_modules *= (1.0 / other)
        elif isinstance(other, np.ndarray) and other.shape == (tracer_module_cnt(),):
            self._tracer_modules *= (1.0 / other)
        elif isinstance(other, ModelState):
            self._tracer_modules /= other._tracer_modules # pylint: disable=W0212
        else:
            return NotImplemented
        return self

    def mean(self):
        """compute weighted mean of self"""
        res = np.empty(shape=(tracer_module_cnt(),), dtype=np.object)
        for ind in range(tracer_module_cnt()):
            res[ind] = self._tracer_modules[ind].mean()
        return res

    def dot_prod(self, other):
        """compute weighted dot product of self with other"""
        res = np.empty(shape=(tracer_module_cnt(),), dtype=np.object)
        for ind in range(tracer_module_cnt()):
            res[ind] = self._tracer_modules[ind].dot_prod(other._tracer_modules[ind]) # pylint: disable=W0212
        return res

    def norm(self):
        """compute weighted l2 norm of self"""
        return np.sqrt(self.dot_prod(self))

    def mod_gram_schmidt(self, basis_cnt, fname_fcn, quantity):
        """
        inplace modified Gram-Schmidt projection
        return projection coefficients
        """
        h_val = np.empty(shape=(tracer_module_cnt(), basis_cnt), dtype=np.object)
        for i_val in range(0, basis_cnt):
            basis_i = ModelState(fname_fcn(quantity, i_val))
            h_val[:, i_val] = self.dot_prod(basis_i)
            self -= h_val[:, i_val] * basis_i
        return h_val

    def run_cmd(self, cmd, res_fname, solver_state):
        """
        Run a command/function from newton_fcn_modname.
        The external command is expected to take 2 arguments: in_fname, res_fname
        in_fname is populated with the contents of self

        Skip running the command if currstep generated below has been logged in solver_state.
        """
        logger = logging.getLogger(__name__)
        logger.debug('entering, cmd="%s", res_fname="%s"', cmd, res_fname)

        if _model_static_vars.cfg_fname is None:
            msg = '_model_static_vars.cfg_fname is None' \
                  ', ModelStaticVars.__init__ must be called with cfg_fname argument' \
                  ' before ModelState.run_cmd'
            raise RuntimeError(msg)

        currstep = 'calling %s for %s' % (cmd, res_fname)
        solver_state.set_currstep(currstep)

        if not _model_static_vars.cmd_ext[cmd]:
            # invoke cmd directly and return result
            res = _model_static_vars.cmd_fcn[cmd](self)
            if cmd == 'comp_fcn':
                res.zero_extra_tracers()
            return res

        if solver_state.currstep_logged():
            logger.debug('"%s" logged, skipping %s and returning result', currstep, cmd)
            res = ModelState(res_fname)
            if cmd == 'comp_fcn':
                res.zero_extra_tracers().dump(res_fname)
            return res

        logger.debug('"%s" not logged, invoking %s and exiting', currstep, cmd)

        # dump self into a file to be read by newton_fcn_modname
        cmd_in_fname = os.path.join(solver_state.get_workdir(), 'cmd_in.nc')
        self.dump(cmd_in_fname)

        args = [sys.executable, _model_static_vars.fcn_lib_file,
                '--cfg_fname', _model_static_vars.cfg_fname,
                '--postrun_cmd', 'postrun.sh',
                cmd, cmd_in_fname, res_fname]
        subprocess.Popen(args)

        logger.debug('flushing solver_state')
        solver_state.flush()

        logger.debug('raising SystemExit')
        raise SystemExit

    def comp_jacobian_fcn_state_prod(self, fcn, direction, solver_state):
        """
        compute the product of the Jacobian of fcn at self with the model state direction

        assumes direction is a unit vector
        """
        logger = logging.getLogger(__name__)
        logger.debug('entering')

        sigma = 1.0e-4 * self.norm()

        res_fname = os.path.join(solver_state.get_workdir(), 'fcn_res.nc')

        solver_state.set_currstep('comp_jacobian_fcn_state_prod_comp_fcn')
        # skip computation of peturbed state if corresponding run_cmd has already been run
        if not solver_state.currstep_logged():
            res_perturb = (self + sigma * direction).run_cmd('comp_fcn', res_fname, solver_state)
        else:
            res_perturb = ModelState(res_fname)

        # retrieve comp_fcn result from res_fname, and proceed with finite difference
        logger.debug('returning')
        return (res_perturb - fcn) / sigma

    def get_tracer_vals(self, tracer_name):
        """get tracer values"""
        for tracer_module in self._tracer_modules:
            try:
                return tracer_module.get_tracer_vals(tracer_name)
            except ValueError:
                pass
        raise ValueError('unknown tracer_name=', tracer_name)

    def set_tracer_vals(self, tracer_name, vals):
        """set tracer values"""
        for tracer_module in self._tracer_modules:
            try:
                tracer_module.set_tracer_vals(tracer_name, vals)
            except ValueError:
                pass

    def copy_shadow_tracers_to_real_tracers(self):
        """copy shadow tracers to their real counterparts"""
        for tracer_module in self._tracer_modules:
            tracer_module.copy_shadow_tracers_to_real_tracers()
        return self

    def copy_real_tracers_to_shadow_tracers(self):
        """overwrite shadow tracers with their real counterparts"""
        for tracer_module in self._tracer_modules:
            tracer_module.copy_real_tracers_to_shadow_tracers()
        return self

    def zero_extra_tracers(self):
        """set extra tracers (i.e., not being solved for) to zero"""
        for tracer_module in self._tracer_modules:
            tracer_module.zero_extra_tracers()
        return self

################################################################################

class TracerModuleState:
    """class for representing the a collection of model tracers"""

    # give TracerModuleState operators higher priority than those of numpy
    __array_priority__ = 100

    def __init__(self, tracer_module_name, dims=None, vals_fname=None):
        if _model_static_vars is None:
            msg = '_model_static_vars is None' \
                  ', ModelStaticVars.__init__ must be called before TracerModuleState.__init__'
            raise RuntimeError(msg)
        self._tracer_module_name = tracer_module_name
        self._tracer_module_def = _model_static_vars.tracer_module_defs[tracer_module_name]
        self._tracer_names = self._tracer_module_def['tracer_names']
        if dims is None != vals_fname is None:
            raise ValueError('exactly one of dims and vals_fname must be passed')
        if dims is not None:
            self._dims = dims
        if vals_fname is not None:
            self._dims = {}
            with Dataset(vals_fname, mode='r') as fptr:
                fptr.set_auto_mask(False)
                # get dims from first variable
                dimnames0 = fptr.variables[self._tracer_names[0]].dimensions
                for dimname in dimnames0:
                    self._dims[dimname] = fptr.dimensions[dimname].size
                # all tracers are stored in a single array
                # tracer index is the leading index
                self._vals = np.empty(shape=(len(self._tracer_names),) + tuple(self._dims.values()))
                # check that all vars have the same dimensions
                for tracer_name in self._tracer_names:
                    if fptr.variables[tracer_name].dimensions != dimnames0:
                        raise ValueError('not all vars have same dimensions',
                                         'tracer_module_name=', tracer_module_name,
                                         'vals_fname=', vals_fname)
                # read values
                if len(self._dims) > 3:
                    raise ValueError('ndim too large (for implementation of dot_prod)',
                                     'tracer_module_name=', tracer_module_name,
                                     'vals_fname=', vals_fname,
                                     'ndim=', len(self._dims))
                for varind, tracer_name in enumerate(self._tracer_names):
                    varid = fptr.variables[tracer_name]
                    self._vals[varind, :] = varid[:]

        # create list of indices of tracers that are extra (i.e., they are not being solved for)
        # tracers that are shadowed are automatically extra
        self._extra_tracer_inds = []
        for tracer_name in self._tracer_module_def['shadow_tracers'].values():
            self._extra_tracer_inds.append(self._tracer_names.index(tracer_name))

    def dump(self, fptr, action):
        """
        perform an action (define or write) of dumping a TracerModuleState object to an open file
        """
        if action == 'define':
            for dimname, dimlen in self._dims.items():
                try:
                    if fptr.dimensions[dimname].size != dimlen:
                        raise ValueError('dimname already exists and has wrong size',
                                         'tracer_module_name=', self._tracer_module_name,
                                         'dimname=', dimname)
                except KeyError:
                    fptr.createDimension(dimname, dimlen)
            dimnames = tuple(self._dims.keys())
            for tracer_name in self._tracer_names:
                fptr.createVariable(tracer_name, 'f8', dimensions=dimnames)
        elif action == 'write':
            for varind, tracer_name in enumerate(self._tracer_names):
                fptr.variables[tracer_name][:] = self._vals[varind, :]
        else:
            raise ValueError('unknown action=', action)
        return self

    def copy(self):
        """return a copy of self"""
        res = TracerModuleState(self._tracer_module_name, dims=self._dims)
        res._vals = np.copy(self._vals) # pylint: disable=W0212
        return res

    def __neg__(self):
        """
        unary negation operator
        called to evaluate res = -self
        """
        res = TracerModuleState(self._tracer_module_name, dims=self._dims)
        res._vals = -self._vals # pylint: disable=W0212
        return res

    def __add__(self, other):
        """
        addition operator
        called to evaluate res = self + other
        """
        res = TracerModuleState(self._tracer_module_name, dims=self._dims)
        if isinstance(other, float):
            res._vals = self._vals + other # pylint: disable=W0212
        elif isinstance(other, RegionScalars):
            res._vals = self._vals + other.broadcast(0.0) # pylint: disable=W0212
        elif isinstance(other, TracerModuleState):
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
        elif isinstance(other, RegionScalars):
            self._vals += other.broadcast(0.0)
        elif isinstance(other, TracerModuleState):
            self._vals += other._vals # pylint: disable=W0212
        else:
            return NotImplemented
        return self

    def __sub__(self, other):
        """
        subtraction operator
        called to evaluate res = self - other
        """
        res = TracerModuleState(self._tracer_module_name, dims=self._dims)
        if isinstance(other, float):
            res._vals = self._vals - other # pylint: disable=W0212
        elif isinstance(other, RegionScalars):
            res._vals = self._vals - other.broadcast(0.0) # pylint: disable=W0212
        elif isinstance(other, TracerModuleState):
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
        elif isinstance(other, RegionScalars):
            self._vals -= other.broadcast(0.0)
        elif isinstance(other, TracerModuleState):
            self._vals -= other._vals # pylint: disable=W0212
        else:
            return NotImplemented
        return self

    def __mul__(self, other):
        """
        multiplication operator
        called to evaluate res = self * other
        """
        res = TracerModuleState(self._tracer_module_name, dims=self._dims)
        if isinstance(other, float):
            res._vals = self._vals * other # pylint: disable=W0212
        elif isinstance(other, RegionScalars):
            res._vals = self._vals * other.broadcast(1.0) # pylint: disable=W0212
        elif isinstance(other, TracerModuleState):
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
        elif isinstance(other, RegionScalars):
            self._vals *= other.broadcast(1.0)
        elif isinstance(other, TracerModuleState):
            self._vals *= other._vals # pylint: disable=W0212
        else:
            return NotImplemented
        return self

    def __truediv__(self, other):
        """
        division operator
        called to evaluate res = self / other
        """
        res = TracerModuleState(self._tracer_module_name, dims=self._dims)
        if isinstance(other, float):
            res._vals = self._vals * (1.0 / other) # pylint: disable=W0212
        elif isinstance(other, RegionScalars):
            res._vals = self._vals * other.recip().broadcast(1.0) # pylint: disable=W0212
        elif isinstance(other, TracerModuleState):
            res._vals = self._vals / other._vals # pylint: disable=W0212
        else:
            return NotImplemented
        return res

    def __rtruediv__(self, other):
        """
        reversed division operator
        called to evaluate res = other / self
        """
        res = TracerModuleState(self._tracer_module_name, dims=self._dims)
        if isinstance(other, float):
            res._vals = other / self._vals # pylint: disable=W0212
        elif isinstance(other, RegionScalars):
            res._vals = other.broadcast(1.0) / self._vals # pylint: disable=W0212
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
        elif isinstance(other, RegionScalars):
            self._vals *= other.recip().broadcast(1.0)
        elif isinstance(other, TracerModuleState):
            self._vals /= other._vals # pylint: disable=W0212
        else:
            return NotImplemented
        return self

    def mean(self):
        """compute weighted mean of self"""
        ndim = len(self._dims)
        # i: region dimension
        # j: tracer dimension
        # k,l,m : grid dimensions
        # sum over model grid dimensions, leaving region and tracer dimensions
        if ndim == 1:
            tmp = np.einsum('ik,jk', _model_static_vars.grid_weight, self._vals)
        elif ndim == 2:
            tmp = np.einsum('ikl,jkl', _model_static_vars.grid_weight, self._vals)
        else:
            tmp = np.einsum('iklm,jklm', _model_static_vars.grid_weight, self._vals)
        # sum over tracer dimension, and return RegionScalars object
        return RegionScalars(np.sum(tmp, axis=-1))

    def dot_prod(self, other):
        """compute weighted dot product of self with other"""
        ndim = len(self._dims)
        # i: region dimension
        # j: tracer dimension
        # k,l,m : grid dimensions
        # sum over tracer and model grid dimensions, leaving region dimension
        if ndim == 1:
            tmp = np.einsum('ik,jk,jk', _model_static_vars.grid_weight, self._vals,
                            other._vals) # pylint: disable=W0212
        elif ndim == 2:
            tmp = np.einsum('ikl,jkl,jkl', _model_static_vars.grid_weight, self._vals,
                            other._vals) # pylint: disable=W0212
        else:
            tmp = np.einsum('iklm,jklm,jklm', _model_static_vars.grid_weight, self._vals,
                            other._vals) # pylint: disable=W0212
        # return RegionScalars object
        return RegionScalars(tmp)

    def get_tracer_vals(self, tracer_name):
        """get tracer values"""
        ind = self._tracer_names.index(tracer_name)
        return self._vals[ind, :]

    def set_tracer_vals(self, tracer_name, vals):
        """set tracer values"""
        ind = self._tracer_names.index(tracer_name)
        self._vals[ind, :] = vals

    def copy_shadow_tracers_to_real_tracers(self):
        """copy shadow tracers to their real counterparts"""
        shadow_tracers = self._tracer_module_def['shadow_tracers']
        for shadow_tracer_name, real_tracer_name in shadow_tracers.items():
            self.set_tracer_vals(real_tracer_name, self.get_tracer_vals(shadow_tracer_name))

    def copy_real_tracers_to_shadow_tracers(self):
        """overwrite shadow tracers with their real counterparts"""
        shadow_tracers = self._tracer_module_def['shadow_tracers']
        for shadow_tracer_name, real_tracer_name in shadow_tracers.items():
            self.set_tracer_vals(shadow_tracer_name, self.get_tracer_vals(real_tracer_name))

    def zero_extra_tracers(self):
        """set extra tracers (i.e., not being solved for) to zero"""
        for tracer_ind in self._extra_tracer_inds:
            self._vals[tracer_ind, :] = 0.0

################################################################################

class RegionScalars:
    """class to hold per-region scalars"""

    def __init__(self, vals):
        self._vals = np.array(vals)

    def __mul__(self, other):
        """
        multiplication operator
        called to evaluate res = self * other
        """
        if isinstance(other, float):
            return RegionScalars(self._vals * other)
        if isinstance(other, RegionScalars):
            return RegionScalars(self._vals * other._vals) # pylint: disable=W0212
        return NotImplemented

    def __rmul__(self, other):
        """
        reversed multiplication operator
        called to evaluate res = other * self
        """
        return self * other

    def __truediv__(self, other):
        """
        division operator
        called to evaluate res = self / other
        """
        if isinstance(other, float):
            return RegionScalars(self._vals / other)
        if isinstance(other, RegionScalars):
            return RegionScalars(self._vals / other._vals) # pylint: disable=W0212
        return NotImplemented

    def __rtruediv__(self, other):
        """
        reversed division operator
        called to evaluate res = other / self
        """
        if isinstance(other, float):
            return RegionScalars(other / self._vals)
        return NotImplemented

    def vals(self):
        """return vals from object"""
        return self._vals

    def recip(self):
        """return RegionScalars object with reciprocal operator applied to vals in self"""
        return RegionScalars(1.0 / self._vals)

    def sqrt(self):
        """return RegionScalars object with sqrt applied to vals in self"""
        return RegionScalars(np.sqrt(self._vals))

    def broadcast(self, fill_value):
        """
        broadcast vals from self to an array of same shape as region_mask
        values in the results are:
            fill_value    where region_mask is <= 0 (e.g. complement of computational domain)
            _vals[ind]    where region_mask == ind+1
        """
        res = np.full(shape=_model_static_vars.region_mask.shape, fill_value=fill_value)
        for region_ind in range(region_cnt()):
            res = np.where(_model_static_vars.region_mask == region_ind+1,
                           self._vals[region_ind], res)
        return res

################################################################################

def to_ndarray(array_in):
    """
    Create an ndarray, res, from an ndarray of RegionScalars.
    res.ndim is 1 greater than array_in.ndim.
    The implicit RegionScalars dimension is placed last in res.
    """

    res = np.empty(shape=array_in.shape+(region_cnt(),))

    if array_in.ndim == 0:
        res[:] = array_in[()].vals()
    elif array_in.ndim == 1:
        for ind0 in range(array_in.shape[0]):
            res[ind0, :] = array_in[ind0].vals()
    elif array_in.ndim == 2:
        for ind0 in range(array_in.shape[0]):
            for ind1 in range(array_in.shape[1]):
                res[ind0, ind1, :] = array_in[ind0, ind1].vals()
    elif array_in.ndim == 3:
        for ind0 in range(array_in.shape[0]):
            for ind1 in range(array_in.shape[1]):
                for ind2 in range(array_in.shape[2]):
                    res[ind0, ind1, ind2, :] = array_in[ind0, ind1, ind2].vals()
    else:
        raise ValueError('array_in.ndim=%d not handled' % array_in.ndim)

    return res

def to_region_scalar_ndarray(array_in):
    """
    Create an ndarray of RegionScalars, res, from an ndarray.
    res.ndim is 1 less than array_in.ndim.
    The last dimension of array_in corresponds to to implicit RegionScalars dimension in res.
    """

    if array_in.shape[-1] != region_cnt():
        raise ValueError('last dimension must have length region_cnt()')

    res = np.empty(shape=array_in.shape[:-1], dtype=np.object)

    if array_in.ndim == 1:
        res[()] = RegionScalars(array_in[:])
    elif array_in.ndim == 2:
        for ind0 in range(array_in.shape[0]):
            res[ind0] = RegionScalars(array_in[ind0, :])
    elif array_in.ndim == 3:
        for ind0 in range(array_in.shape[0]):
            for ind1 in range(array_in.shape[1]):
                res[ind0, ind1] = RegionScalars(array_in[ind0, ind1, :])
    elif array_in.ndim == 4:
        for ind0 in range(array_in.shape[0]):
            for ind1 in range(array_in.shape[1]):
                for ind2 in range(array_in.shape[2]):
                    res[ind0, ind1, ind2] = RegionScalars(array_in[ind0, ind1, ind2, :])
    else:
        raise ValueError('array_in.ndim=%d not handled' % array_in.ndim)

    return res

def lin_comb(coeff, fname_fcn, quantity):
    """compute a linear combination of ModelState objects in files"""
    res = coeff[:, 0] * ModelState(fname_fcn(quantity, 0))
    for j_val in range(1, coeff.shape[-1]):
        res += coeff[:, j_val] * ModelState(fname_fcn(quantity, j_val))
    return res

def log_vals(msg, vals, ind=None):
    """write per-tracer module values to the log"""
    logger = logging.getLogger(__name__)

    # loop over tracer modules
    if ind is None:
        for ind_tmp in range(tracer_module_cnt()):
            log_vals(msg, vals, ind_tmp)
        return

    # can now assume ind is present

    # simplify subsequent logic by converting implicit RegionScalars dimension
    # to an additional ndarray dimension
    if isinstance(vals.ravel()[0], RegionScalars):
        log_vals(msg, to_ndarray(vals), ind)
        return

    # suppress printing of last index if its span is 1
    if vals.shape[-1] == 1 and vals.ndim >= 2:
        log_vals(msg, vals[..., 0], ind)
        return

    tracer_module_name = tracer_module_names()[ind]

    if vals.ndim == 1:
        logger.info('%s[%s]=%e', msg, tracer_module_name, vals[ind])
    elif vals.ndim == 2:
        for j in range(vals.shape[1]):
            logger.info('%s[%s,%d]=%e', msg, tracer_module_name, j, vals[ind, j])
    elif vals.ndim == 3:
        for i in range(vals.shape[1]):
            for j in range(vals.shape[2]):
                logger.info('%s[%s,%d,%d]=%e', msg, tracer_module_name, i, j, vals[ind, i, j])
    else:
        raise ValueError('vals.ndim=%d not handled' % vals.ndim)
