"""class for representing the state space of a model, and operations on it"""

import importlib
import logging
import os

import numpy as np
from netCDF4 import Dataset
import yaml

# model static variables
_model_static_vars = None

# functions to commonly accessed vars in _model_static_vars
def get_region_cnt():
    """return number of regions specified by region_mask"""
    return _model_static_vars.region_cnt

def get_tracer_module_def(tracer_module_name):
    """return the tracer_module_defs dictionary from _model_static_vars"""
    return _model_static_vars.tracer_module_defs[tracer_module_name]

def get_modelinfo(key):
    """return value associated in modelinfo with key"""
    return _model_static_vars.modelinfo[key]

################################################################################

class ModelStaticVars:
    """class to hold static vars"""

    def __init__(self, modelinfo, lvl=logging.DEBUG):
        logger = logging.getLogger(__name__)
        logger.debug('ModelStaticVars:entering')

        # store modelinfo for later use
        self.modelinfo = modelinfo

        # import module with TracerModuleState and NewtonFcn
        newton_fcn_mod = importlib.import_module(modelinfo['newton_fcn_modname'])

        # store newton_fcn_mod's TracerModuleState class and an instance of class NewtonFcn
        self.tracer_module_state = newton_fcn_mod.TracerModuleState
        self.newton_fcn = newton_fcn_mod.NewtonFcn()

        # extract tracer_module_defs from modelinfo config object
        fname = modelinfo['tracer_module_defs_fname']
        logger.log(lvl, 'reading tracer_module_defs from %s', fname)
        with open(fname, mode='r') as fptr:
            self.tracer_module_defs = yaml.load(fptr)

        self._check_shadow_tracers(lvl)

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
            # If no tracers are specified, add an empty tracer_names dictionary to
            # tracer_module_defs, to ease subsequent coding.
            if 'tracer_names' not in tracer_module_def:
                logger.log(lvl, 'tracer module %s has no tracers', tracer_module_name)
                tracer_module_def['tracer_names'] = {}

            if 'shadow_tracers' in tracer_module_def:
                shadow_tracers = tracer_module_def['shadow_tracers']
            else:
                shadow_tracers = {}
                tracer_module_def['shadow_tracers'] = {}
                logger.log(lvl, 'tracer module %s has no shadow tracers', tracer_module_name)

            # Verify that shadow_tracer_name and real_tracer_name are known tracer names.
            for shadow_tracer_name, real_tracer_name in shadow_tracers.items():
                if shadow_tracer_name not in tracer_module_def['tracer_names']:
                    raise ValueError('specified shadow tracer %s in tracer module %s not known'
                                     % (shadow_tracer_name, tracer_module_name))
                if real_tracer_name not in tracer_module_def['tracer_names']:
                    raise ValueError('specified tracer %s in tracer module %s not known'
                                     % (real_tracer_name, tracer_module_name))
                logger.log(lvl, 'tracer module %s has %s as a shadow for %s',
                           tracer_module_name, shadow_tracer_name, real_tracer_name)

################################################################################

class ModelState:
    """class for representing the state space of a model"""

    # give ModelState operators higher priority than those of numpy
    __array_priority__ = 100

    def __init__(self, vals_fname=None):
        logger = logging.getLogger(__name__)
        logger.debug('ModelState:entering, vals_fname=%s', vals_fname)
        if _model_static_vars is None:
            msg = '_model_static_vars is None' \
                  ', ModelStaticVars.__init__ must be called before ModelState.__init__'
            raise RuntimeError(msg)
        self.tracer_module_names = get_modelinfo('tracer_module_names').split(',')
        self.tracer_module_cnt = len(self.tracer_module_names)
        if vals_fname is not None:
            self._tracer_modules = np.empty(shape=(self.tracer_module_cnt,), dtype=np.object)
            for tracer_module_ind, tracer_module_name in enumerate(self.tracer_module_names):
                self._tracer_modules[tracer_module_ind] = \
                    _model_static_vars.tracer_module_state(tracer_module_name,
                                                           vals_fname=vals_fname)
        logger.debug('returning')

    def tracer_names(self):
        """return list of tracer names"""
        res = []
        for tracer_module in self._tracer_modules:
            res.extend(tracer_module.tracer_names())
        return res

    def tracer_cnt(self):
        """return number of tracers"""
        return len(self.tracer_names())

    def tracer_index(self, tracer_name):
        """return the index of a tracer"""
        return self.tracer_names().index(tracer_name)

    def dump(self, vals_fname):
        """dump ModelState object to a file"""
        with Dataset(vals_fname, mode='w') as fptr:
            for action in ['define', 'write']:
                for tracer_module in self._tracer_modules:
                    tracer_module.dump(fptr, action)
        return self

    def log_vals(self, msg, vals):
        """write per-tracer module values to the log"""
        for tracer_module_ind, tracer_module in enumerate(self._tracer_modules):
            if isinstance(msg, list):
                for msg_ind, submsg in enumerate(msg):
                    if vals.ndim == 2:
                        tracer_module.log_vals(submsg, vals[msg_ind, tracer_module_ind])
                    else:
                        tracer_module.log_vals(submsg, vals[msg_ind, tracer_module_ind, :])
            else:
                if vals.ndim == 1:
                    tracer_module.log_vals(msg, vals[tracer_module_ind])
                else:
                    tracer_module.log_vals(msg, vals[tracer_module_ind, :])

    def log(self, msg=None):
        """write info of the instance to the log"""
        if msg is None:
            msg_full = ['mean', 'norm']
        else:
            msg_full = [msg+',mean', msg+',norm']
        self.log_vals(msg_full, np.stack((self.mean(), self.norm())))

    def copy(self):
        """return a copy of self"""
        res = ModelState()
        res._tracer_modules = np.empty(shape=(self.tracer_module_cnt,), dtype=np.object) # pylint: disable=W0212
        for tracer_module_ind, tracer_module in enumerate(self._tracer_modules):
            res._tracer_modules[tracer_module_ind] = tracer_module.copy() # pylint: disable=W0212
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
        elif isinstance(other, np.ndarray) and other.shape == self._tracer_modules.shape:
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
        elif isinstance(other, np.ndarray) and other.shape == self._tracer_modules.shape:
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
        elif isinstance(other, np.ndarray) and other.shape == self._tracer_modules.shape:
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
        elif isinstance(other, np.ndarray) and other.shape == self._tracer_modules.shape:
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
        elif isinstance(other, np.ndarray) and other.shape == self._tracer_modules.shape:
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
        elif isinstance(other, np.ndarray) and other.shape == self._tracer_modules.shape:
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
        elif isinstance(other, np.ndarray) and other.shape == self._tracer_modules.shape:
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
        elif isinstance(other, np.ndarray) and other.shape == self._tracer_modules.shape:
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
        elif isinstance(other, np.ndarray) and other.shape == self._tracer_modules.shape:
            self._tracer_modules *= (1.0 / other)
        elif isinstance(other, ModelState):
            self._tracer_modules /= other._tracer_modules # pylint: disable=W0212
        else:
            return NotImplemented
        return self

    def mean(self):
        """compute weighted mean of self"""
        res = np.empty(shape=self._tracer_modules.shape, dtype=np.object)
        for ind, tracer_module in enumerate(self._tracer_modules):
            res[ind] = tracer_module.mean()
        return res

    def dot_prod(self, other):
        """compute weighted dot product of self with other"""
        res = np.empty(shape=self._tracer_modules.shape, dtype=np.object)
        for ind, tracer_module in enumerate(self._tracer_modules):
            res[ind] = tracer_module.dot_prod(other._tracer_modules[ind]) # pylint: disable=W0212
        return res

    def norm(self):
        """compute weighted l2 norm of self"""
        return np.sqrt(self.dot_prod(self))

    def mod_gram_schmidt(self, basis_cnt, fname_fcn, quantity):
        """
        inplace modified Gram-Schmidt projection
        return projection coefficients
        """
        h_val = np.empty(shape=(self.tracer_module_cnt, basis_cnt), dtype=np.object)
        for i_val in range(0, basis_cnt):
            basis_i = ModelState(fname_fcn(quantity, i_val))
            h_val[:, i_val] = self.dot_prod(basis_i)
            self -= h_val[:, i_val] * basis_i
        return h_val

    def comp_fcn(self, res_fname, solver_state, hist_fname=None):
        """Compute the function whose root is being found."""
        logger = logging.getLogger(__name__)
        logger.debug('entering, res_fname="%s"', res_fname)

        cmd = 'comp_fcn'
        fcn_complete_step = '%s done for %s' % (cmd, res_fname)

        if solver_state.step_logged(fcn_complete_step):
            logger.debug('"%s" logged, returning result', fcn_complete_step)
            return ModelState(res_fname)

        logger.debug('"%s" not logged, invoking %s', fcn_complete_step, cmd)

        res = _model_static_vars.newton_fcn.comp_fcn(self, res_fname, solver_state, hist_fname)
        res.zero_extra_tracers().dump(res_fname)

        solver_state.log_step(fcn_complete_step)

        logger.debug('returning')
        return res

    def comp_jacobian_fcn_state_prod(self, fcn, direction, res_fname, solver_state):
        """
        compute the product of the Jacobian of fcn at self with the model state direction

        assumes direction is a unit vector
        """
        logger = logging.getLogger(__name__)
        logger.debug('entering')

        fcn_complete_step = 'comp_jacobian_fcn_state_prod done for %s' % (res_fname)

        if solver_state.step_logged(fcn_complete_step):
            logger.debug('"%s" logged, returning result', fcn_complete_step)
            return ModelState(res_fname)

        logger.debug('"%s" not logged, proceeding', fcn_complete_step)

        sigma = 1.0e-4 * self.norm()

        # perturbed ModelState
        perturb_ms = self + sigma * direction
        perturb_fcn_fname = os.path.join(solver_state.get_workdir(),
                                         'perturb_fcn_'+os.path.basename(res_fname))
        perturb_fcn = perturb_ms.comp_fcn(perturb_fcn_fname, solver_state)

        # compute finite difference
        res = ((perturb_fcn - fcn) / sigma).dump(res_fname)

        solver_state.log_step(fcn_complete_step)

        logger.debug('returning')
        return res

    def apply_precond_jacobian(self, res_fname, solver_state):
        """Apply preconditioner of jacobian of comp_fcn to self."""
        logger = logging.getLogger(__name__)
        logger.debug('entering, res_fname="%s"', res_fname)

        cmd = 'apply_precond_jacobian'
        fcn_complete_step = '%s done for %s' % (cmd, res_fname)

        if solver_state.step_logged(fcn_complete_step):
            logger.debug('"%s" logged, returning result', fcn_complete_step)
            return ModelState(res_fname)

        logger.debug('"%s" not logged, invoking %s', fcn_complete_step, cmd)

        res = _model_static_vars.newton_fcn.apply_precond_jacobian(self, res_fname, solver_state)

        solver_state.log_step(fcn_complete_step)

        logger.debug('returning')
        return res

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

    def shadow_tracers_on(self):
        """are any shadow tracers being run"""
        for tracer_module in self._tracer_modules:
            if tracer_module.shadow_tracers_on():
                return True
        return False

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

class TracerModuleStateBase:
    """
    Base class for representing a collection of model tracers.
    Derived classes should implement _read_vals and dump.
    """

    # give TracerModuleStateBase operators higher priority than those of numpy
    __array_priority__ = 100

    def __init__(self, tracer_module_name, dims=None, vals_fname=None):
        logger = logging.getLogger(__name__)
        logger.debug('TracerModuleStateBase:entering, vals_fname=%s', vals_fname)
        if _model_static_vars is None:
            msg = '_model_static_vars is None' \
                  ', ModelStaticVars.__init__ must be called before TracerModuleStateBase.__init__'
            raise RuntimeError(msg)
        self._tracer_module_name = tracer_module_name
        self._tracer_module_def = _model_static_vars.tracer_module_defs[tracer_module_name]
        if dims is None != vals_fname is None:
            raise ValueError('exactly one of dims and vals_fname must be passed')
        if dims is not None:
            self._dims = dims
        if vals_fname is not None:
            self._vals, self._dims = self._read_vals(tracer_module_name, vals_fname)
        logger.debug('returning')

    def _read_vals(self, tracer_module_name, vals_fname):
        """return tracer values and dimension names and lengths, read from vals_fname)"""
        raise NotImplementedError(
            '_read_vals should be implemented in classes derived from TracerModuleStateBase')

    def tracer_names(self):
        """return list of tracer names"""
        return self._tracer_module_def['tracer_names']

    def tracer_cnt(self):
        """return number of tracers"""
        return len(self.tracer_names())

    def tracer_index(self, tracer_name):
        """return the index of a tracer"""
        return self.tracer_names().index(tracer_name)

    def dump(self, fptr, action):
        """
        perform an action (define or write) of dumping a TracerModuleStateBase object
        to an open file
        """
        raise NotImplementedError(
            'dump should be implemented in classes derived from TracerModuleStateBase')

    def log_vals(self, msg, vals):
        """write per-tracer module values to the log"""
        logger = logging.getLogger(__name__)

        # simplify subsequent logic by converting implicit RegionScalars dimension
        # to an additional ndarray dimension
        if isinstance(vals, RegionScalars) \
                or isinstance(vals, np.ndarray) and isinstance(vals.ravel()[0], RegionScalars):
            self.log_vals(msg, to_ndarray(vals))
            return

        # suppress printing of last index if its span is 1
        if vals.ndim >= 1 and vals.shape[-1] == 1:
            self.log_vals(msg, vals[..., 0])
            return

        if vals.ndim == 0:
            logger.info('%s[%s]=%e', msg, self._tracer_module_name, vals)
        elif vals.ndim == 1:
            for j in range(vals.shape[0]):
                logger.info('%s[%s,%d]=%e', msg, self._tracer_module_name, j, vals[j])
        elif vals.ndim == 2:
            for i in range(vals.shape[0]):
                for j in range(vals.shape[1]):
                    logger.info('%s[%s,%d,%d]=%e', msg, self._tracer_module_name, i, j, vals[i, j])
        else:
            raise ValueError('vals.ndim=%d not handled' % vals.ndim)

    def copy(self):
        """return a copy of self"""
        res = type(self)(self._tracer_module_name, dims=self._dims)
        res._vals = np.copy(self._vals) # pylint: disable=W0212
        return res

    def __neg__(self):
        """
        unary negation operator
        called to evaluate res = -self
        """
        res = type(self)(self._tracer_module_name, dims=self._dims)
        res._vals = -self._vals # pylint: disable=W0212
        return res

    def __add__(self, other):
        """
        addition operator
        called to evaluate res = self + other
        """
        res = type(self)(self._tracer_module_name, dims=self._dims)
        if isinstance(other, float):
            res._vals = self._vals + other # pylint: disable=W0212
        elif isinstance(other, RegionScalars):
            res._vals = self._vals + other.broadcast(0.0) # pylint: disable=W0212
        elif isinstance(other, TracerModuleStateBase):
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
        elif isinstance(other, TracerModuleStateBase):
            self._vals += other._vals # pylint: disable=W0212
        else:
            return NotImplemented
        return self

    def __sub__(self, other):
        """
        subtraction operator
        called to evaluate res = self - other
        """
        res = type(self)(self._tracer_module_name, dims=self._dims)
        if isinstance(other, float):
            res._vals = self._vals - other # pylint: disable=W0212
        elif isinstance(other, RegionScalars):
            res._vals = self._vals - other.broadcast(0.0) # pylint: disable=W0212
        elif isinstance(other, TracerModuleStateBase):
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
        elif isinstance(other, TracerModuleStateBase):
            self._vals -= other._vals # pylint: disable=W0212
        else:
            return NotImplemented
        return self

    def __mul__(self, other):
        """
        multiplication operator
        called to evaluate res = self * other
        """
        res = type(self)(self._tracer_module_name, dims=self._dims)
        if isinstance(other, float):
            res._vals = self._vals * other # pylint: disable=W0212
        elif isinstance(other, RegionScalars):
            res._vals = self._vals * other.broadcast(1.0) # pylint: disable=W0212
        elif isinstance(other, TracerModuleStateBase):
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
        elif isinstance(other, TracerModuleStateBase):
            self._vals *= other._vals # pylint: disable=W0212
        else:
            return NotImplemented
        return self

    def __truediv__(self, other):
        """
        division operator
        called to evaluate res = self / other
        """
        res = type(self)(self._tracer_module_name, dims=self._dims)
        if isinstance(other, float):
            res._vals = self._vals * (1.0 / other) # pylint: disable=W0212
        elif isinstance(other, RegionScalars):
            res._vals = self._vals * other.recip().broadcast(1.0) # pylint: disable=W0212
        elif isinstance(other, TracerModuleStateBase):
            res._vals = self._vals / other._vals # pylint: disable=W0212
        else:
            return NotImplemented
        return res

    def __rtruediv__(self, other):
        """
        reversed division operator
        called to evaluate res = other / self
        """
        res = type(self)(self._tracer_module_name, dims=self._dims)
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
        elif isinstance(other, TracerModuleStateBase):
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
        return self._vals[self.tracer_index(tracer_name), :]

    def set_tracer_vals(self, tracer_name, vals):
        """set tracer values"""
        self._vals[self.tracer_index(tracer_name), :] = vals

    def shadow_tracers_on(self):
        """are any shadow tracers being run"""
        return bool(self._tracer_module_def['shadow_tracers'])

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

    def extra_tracer_inds(self):
        """
        return list of indices of tracers that are extra (i.e., they are not being solved for)
        the indices are with respect to self
        tracers that are shadowed are automatically extra
        """
        res = []
        for tracer_name in self._tracer_module_def['shadow_tracers'].values():
            res.append(self.tracer_index(tracer_name))
        return res

    def zero_extra_tracers(self):
        """set extra tracers (i.e., not being solved for) to zero"""
        for tracer_ind in self.extra_tracer_inds():
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
        for region_ind in range(get_region_cnt()):
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

    if isinstance(array_in, RegionScalars):
        return np.array(array_in.vals())

    res = np.empty(shape=array_in.shape+(get_region_cnt(),))

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

    if array_in.shape[-1] != get_region_cnt():
        raise ValueError('last dimension must have length get_region_cnt()')

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

def gen_precond_jacobian(hist_fname, solver_state):
    """Generate file(s) needed for preconditioner of jacobian of comp_fcn."""
    logger = logging.getLogger(__name__)
    logger.debug('entering, hist_fname="%s"', hist_fname)

    cmd = 'gen_precond_jacobian'
    fcn_complete_step = '%s done for %s' % (cmd, hist_fname)

    if solver_state.step_logged(fcn_complete_step):
        logger.debug('"%s" logged, returning', fcn_complete_step)
        return

    logger.debug('"%s" not logged, proceeding', fcn_complete_step)

    _model_static_vars.newton_fcn.gen_precond_jacobian(hist_fname, solver_state)

    solver_state.log_step(fcn_complete_step)

    logger.debug('returning')
