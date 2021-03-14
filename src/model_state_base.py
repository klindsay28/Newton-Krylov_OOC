"""base class for representing the state space of a model, and operations on it"""

import collections
import copy
import logging
import os
from datetime import datetime
from inspect import signature

import numpy as np
from netCDF4 import Dataset

from .solver_state import action_step_log_wrap
from .tracer_module_state_base import TracerModuleStateBase
from .utils import (
    class_name,
    create_dimensions_verify,
    create_vars,
    dict_update_verify,
    extract_dimensions,
    get_subclasses,
)


class ModelStateBase:
    """class for representing the state space of a model"""

    # give ModelStateBase operators higher priority than those of numpy
    __array_priority__ = 100

    model_config_obj = None

    def __init__(self, fname):
        logger = logging.getLogger(__name__)
        logger.debug('ModelStateBase, fname="%s"', fname)

        # confirm that model_config_obj has been set for this instance
        if self.model_config_obj is None:
            raise RuntimeError(
                "self.model_config_obj is None, it should be set in derived class"
            )

        modelinfo = self.model_config_obj.modelinfo
        tracer_module_names = modelinfo["tracer_module_names"].split(",")
        self.tracer_modules = np.empty(len(tracer_module_names), dtype=object)
        tracer_module_defs = self.model_config_obj.tracer_module_defs

        pos_args = ["self", "tracer_module_name", "fname"]

        for ind, tracer_module_name in enumerate(tracer_module_names):
            tracer_module_def = tracer_module_defs[tracer_module_name]
            tracer_module_state_class = _get_tracer_module_state_class(
                modelinfo["model_name"], tracer_module_name, tracer_module_def
            )
            logger.debug(
                "using class %s from %s for tracer module %s",
                tracer_module_state_class.__name__,
                tracer_module_state_class.__module__,
                tracer_module_name,
            )
            kwargs = {
                arg: getattr(self, arg)
                for arg in signature(tracer_module_state_class.__init__).parameters
                if arg not in pos_args
            }
            self.tracer_modules[ind] = tracer_module_state_class(
                tracer_module_name, fname, **kwargs
            )

        self.tracer_cnt = sum(
            tracer_module.tracer_cnt for tracer_module in self.tracer_modules
        )

    def comp_fcn(self, res_fname, solver_state, hist_fname=None):
        """
        evalute function being solved with Newton's method
        implemented in derived classes
        """
        raise NotImplementedError("Method must be implemented in derived class")

    def apply_precond_jacobian(self, precond_fname, res_fname, solver_state):
        """
        apply preconditioner of jacobian of comp_fcn to model state object, self
        implemented in derived classes
        """
        raise NotImplementedError("Method must be implemented in derived class")

    def tracer_names(self):
        """return list of tracer names"""
        res = []
        for tracer_module in self.tracer_modules:
            res.extend(tracer_module.tracer_names())
        return res

    def tracer_index(self, tracer_name):
        """return the index of a tracer"""
        return self.tracer_names().index(tracer_name)

    def dump(self, fname, caller=None):
        """dump ModelStateBase object to a file"""
        logger = logging.getLogger(__name__)
        logger.debug('fname="%s"', fname)
        if fname is None:
            return self
        with Dataset(fname, mode="w", format="NETCDF3_64BIT_OFFSET") as fptr:
            datestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            name = class_name(self) + ".dump"
            msg = datestamp + ": created by " + name
            if caller is not None:
                msg = msg + " called from " + caller
            else:
                raise ValueError("caller unknown")
            fptr.history = msg
            for action in ["define", "write"]:
                for tracer_module in self.tracer_modules:
                    tracer_module.dump(fptr, action)
        return self

    def log_vals(self, msg, vals):
        """write per-tracer module values to the log"""
        for tracer_module_ind, tracer_module in enumerate(self.tracer_modules):
            if isinstance(msg, list):
                for msg_ind, submsg in enumerate(msg):
                    if vals.ndim == 2:
                        tracer_module.log_vals(submsg, vals[msg_ind, tracer_module_ind])
                    else:
                        tracer_module.log_vals(
                            submsg, vals[msg_ind, tracer_module_ind, :]
                        )
            else:
                if vals.ndim == 1:
                    tracer_module.log_vals(msg, vals[tracer_module_ind])
                else:
                    tracer_module.log_vals(msg, vals[tracer_module_ind, :])

    def log(self, msg=None):
        """write info of the instance to the log"""
        if msg is None:
            msg_full = ["mean", "norm"]
        else:
            msg_full = [msg + ",mean", msg + ",norm"]
        mean_vals = self.mean()
        norm_vals = self.norm()
        self.log_vals(msg_full, np.stack((mean_vals, norm_vals)))

    @action_step_log_wrap(step="ModelStateBase.def_stats_vars", per_iteration=False)
    # pylint: disable=unused-argument
    def def_stats_vars(self, stats_file, hist_fname, solver_state):
        """define model specific stats variables"""

        dimensions = {}
        vars_metadata = {}
        with Dataset(hist_fname, mode="r") as fptr_hist:
            fptr_hist.set_auto_mask(False)
            for tracer_module in self.tracer_modules:
                dimensions_to_add = tracer_module.stats_dimensions(fptr_hist)
                dict_update_verify(dimensions, dimensions_to_add)

                vars_metadata_to_add = tracer_module.stats_vars_metadata(fptr_hist)
                dict_update_verify(vars_metadata, vars_metadata_to_add)

        stats_file.def_dimensions(dimensions)
        stats_file.def_vars(vars_metadata)

    @action_step_log_wrap(
        step="ModelStateBase.put_stats_vars_iteration_invariant", per_iteration=False
    )
    # pylint: disable=unused-argument
    def put_stats_vars_iteration_invariant(self, stats_file, hist_fname, solver_state):
        """put values of iteration-invariant stats variables"""
        name_vals_dict = {}
        with Dataset(hist_fname, mode="r") as fptr_hist:
            fptr_hist.set_auto_mask(False)
            for tracer_module in self.tracer_modules:
                name_vals_to_add = tracer_module.stats_vars_vals_iteration_invariant(
                    fptr_hist
                )
                dict_update_verify(name_vals_dict, name_vals_to_add)
        stats_file.put_vars_iteration_invariant(name_vals_dict)

    @action_step_log_wrap(step="ModelStateBase.put_stats_vars")
    def put_stats_vars(self, stats_file, hist_fname, solver_state):
        """put values of stats variables for the current iteration"""
        name_vals_dict = {}
        with Dataset(hist_fname, mode="r") as fptr_hist:
            fptr_hist.set_auto_mask(False)
            for tracer_module in self.tracer_modules:
                name_vals_to_add = tracer_module.stats_vars_vals(fptr_hist)
                dict_update_verify(name_vals_dict, name_vals_to_add)
        stats_file.put_vars(solver_state.get_iteration(), name_vals_dict)

    def __neg__(self):
        """
        unary negation operator
        called to evaluate res = -self
        """
        res = copy.copy(self)
        res.tracer_modules = -self.tracer_modules
        return res

    def __add__(self, other):
        """
        addition operator
        called to evaluate res = self + other
        """
        res = copy.copy(self)
        if isinstance(other, ModelStateBase):
            res.tracer_modules = self.tracer_modules + other.tracer_modules
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
        if isinstance(other, ModelStateBase):
            self.tracer_modules += other.tracer_modules
        else:
            return NotImplemented
        return self

    def __sub__(self, other):
        """
        subtraction operator
        called to evaluate res = self - other
        """
        res = copy.copy(self)
        if isinstance(other, ModelStateBase):
            res.tracer_modules = self.tracer_modules - other.tracer_modules
        else:
            return NotImplemented
        return res

    def __isub__(self, other):
        """
        inplace subtraction operator
        called to evaluate self -= other
        """
        if isinstance(other, ModelStateBase):
            self.tracer_modules -= other.tracer_modules
        else:
            return NotImplemented
        return self

    def __mul__(self, other):
        """
        multiplication operator
        called to evaluate res = self * other
        """
        res = copy.copy(self)
        if isinstance(other, float):
            res.tracer_modules = self.tracer_modules * other
        elif isinstance(other, np.ndarray):
            res.tracer_modules = self.tracer_modules * other
        elif isinstance(other, ModelStateBase):
            res.tracer_modules = self.tracer_modules * other.tracer_modules
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
            self.tracer_modules *= other
        elif isinstance(other, np.ndarray):
            self.tracer_modules *= other
        elif isinstance(other, ModelStateBase):
            self.tracer_modules *= other.tracer_modules
        else:
            return NotImplemented
        return self

    def __truediv__(self, other):
        """
        division operator
        called to evaluate res = self / other
        """
        res = copy.copy(self)
        if isinstance(other, float):
            res.tracer_modules = self.tracer_modules * (1.0 / other)
        elif isinstance(other, np.ndarray):
            res.tracer_modules = self.tracer_modules * (1.0 / other)
        elif isinstance(other, ModelStateBase):
            res.tracer_modules = self.tracer_modules / other.tracer_modules
        else:
            return NotImplemented
        return res

    def __rtruediv__(self, other):
        """
        reversed division operator
        called to evaluate res = other / self
        """
        res = copy.copy(self)
        if isinstance(other, float):
            res.tracer_modules = other / self.tracer_modules
        elif isinstance(other, np.ndarray):
            res.tracer_modules = other / self.tracer_modules
        else:
            return NotImplemented
        return res

    def __itruediv__(self, other):
        """
        inplace division operator
        called to evaluate self /= other
        """
        if isinstance(other, float):
            self.tracer_modules *= 1.0 / other
        elif isinstance(other, np.ndarray):
            self.tracer_modules *= 1.0 / other
        elif isinstance(other, ModelStateBase):
            self.tracer_modules /= other.tracer_modules
        else:
            return NotImplemented
        return self

    def mean(self):
        """compute weighted mean of self"""
        res = np.empty(self.tracer_modules.shape, dtype=object)
        for ind, tracer_module in enumerate(self.tracer_modules):
            res[ind] = tracer_module.mean()
        return res

    def dot_prod(self, other):
        """compute weighted dot product of self with other"""
        res = np.empty(self.tracer_modules.shape, dtype=object)
        for ind, tracer_module in enumerate(self.tracer_modules):
            res[ind] = tracer_module.dot_prod(other.tracer_modules[ind])
        return res

    def norm(self):
        """compute weighted l2 norm of self"""
        return np.sqrt(self.dot_prod(self))

    def mod_gram_schmidt(self, basis_cnt, fname_fcn, quantity):
        """
        inplace modified Gram-Schmidt projection
        return projection coefficients
        """
        h_val = np.empty((len(self.tracer_modules), basis_cnt), dtype=object)
        for i_val in range(0, basis_cnt):
            basis_i = type(self)(fname_fcn(quantity, i_val))
            h_val[:, i_val] = self.dot_prod(basis_i)
            self -= h_val[:, i_val] * basis_i
        return h_val

    def hist_vars_for_precond_list(self):
        """Return list of hist vars needed for preconditioner of jacobian of comp_fcn"""
        res = []
        precond_matrix_defs = self.model_config_obj.precond_matrix_defs
        for matrix_name in self.precond_matrix_list() + ["base"]:
            precond_matrix_def = precond_matrix_defs[matrix_name]
            for varname in precond_matrix_def["hist_to_precond_varnames"]:
                if varname not in res:
                    res.append(varname)
        return res

    def precond_matrix_list(self):
        """Return list of precond matrices being used"""
        res = []
        for tracer_module in self.tracer_modules:
            res.extend(tracer_module.precond_matrix_list())
        return res

    def tracer_names_per_precond_matrix(self):
        """Return OrderedDict of tracer names for each precond matrix"""
        res = collections.OrderedDict()
        for tracer_module in self.tracer_modules:
            tracer_module.append_tracer_names_per_precond_matrix(res)
        return res

    @action_step_log_wrap(
        step="ModelStateBase.gen_precond_jacobian {precond_fname}", per_iteration=False
    )
    # pylint: disable=unused-argument
    def gen_precond_jacobian(self, hist_fname, precond_fname, solver_state):
        """
        Generate file(s) needed for preconditioner of jacobian of comp_fcn
        evaluated at self
        """
        logger = logging.getLogger(__name__)
        logger.debug('hist_fname="%s", precond_fname="%s"', hist_fname, precond_fname)

        hist_vars = self.hist_vars_for_precond_list()

        with Dataset(hist_fname, mode="r") as fptr_in, Dataset(
            precond_fname, "w", format="NETCDF3_64BIT_OFFSET"
        ) as fptr_out:
            datestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            fcn_name = class_name(self) + ".gen_precond_jacobian"
            msg = datestamp + ": created by " + fcn_name
            history_in = getattr(fptr_in, "history", None)
            fptr_out.history = (
                msg if history_in is None else "\n".join([msg, history_in])
            )

            for action in ["define", "write"]:
                _precond_dims(hist_vars, fptr_in, fptr_out, action)

                if action == "define":
                    vars_metadata = {}

                # define/write output vars
                for hist_var in hist_vars:
                    hist_varname, _, time_op = hist_var.partition(":")

                    # skip coordinate variables, they are added by _precond_dims
                    if hist_varname in fptr_out.dimensions:
                        continue

                    hist_var = fptr_in.variables[hist_varname]

                    dimensions = _precond_dimensions_for_hist_var(
                        fptr_in, hist_varname, time_op
                    )

                    var_metadata = {
                        "datatype": hist_var.datatype,
                        "dimensions": tuple(dimensions),
                        "attrs": hist_var.__dict__,
                    }

                    # remove cell_methods if time dimension is being referenced and
                    # it doesn't exist in result (should really just remove substring)
                    if "cell_methods" in var_metadata["attrs"]:
                        attrs = var_metadata["attrs"]
                        cell_methods = attrs["cell_methods"]
                        if "time:" in cell_methods and "time" not in dimensions:
                            del attrs["cell_methods"]

                    if time_op == "mean":
                        precond_varname = hist_varname + "_mean"
                        var_metadata["attrs"]["long_name"] += ", mean over time dim"
                        vals = hist_var[:].mean(axis=0)
                    elif time_op == "log_mean":
                        precond_varname = hist_varname + "_log_mean"
                        var_metadata["attrs"]["long_name"] += ", log mean over time dim"
                        vals = np.exp(np.log(hist_var[:]).mean(axis=0))
                    else:
                        precond_varname = hist_varname
                        vals = hist_var[:]

                    if action == "define":
                        vars_metadata[precond_varname] = var_metadata
                    else:
                        fptr_out.variables[precond_varname][:] = vals

                if action == "define":
                    create_vars(fptr_out, vars_metadata)

    def comp_fcn_postprocess(self, res_fname, caller):
        """
        apply postprocessing to comp_fcn result in self that is common to all comp_fcn
        methods
        """
        fcn_name = class_name(self) + ".comp_fcn_postprocess"
        caller = fcn_name + " called from " + caller
        return self.zero_extra_tracers().apply_region_mask().dump(res_fname, caller)

    def comp_jacobian_fcn_state_prod(self, fcn, direction, res_fname, solver_state):
        """
        compute the product of the Jacobian of fcn at self with the model state
        direction

        assumes direction is a unit vector
        """
        logger = logging.getLogger(__name__)
        logger.debug('res_fname="%s"', res_fname)

        fcn_complete_step = "comp_jacobian_fcn_state_prod complete for %s" % res_fname

        if solver_state.step_logged(fcn_complete_step):
            logger.debug('"%s" logged, returning result', fcn_complete_step)
            return type(self)(res_fname)
        logger.debug('"%s" not logged, proceeding', fcn_complete_step)

        sigma = 1.0e-4 * self.norm()

        # set sigma to 1.0 where it is 0.0
        for sigma_tracer_module in sigma:
            sigma_vals = sigma_tracer_module.vals()
            if any(sigma_vals == 0.0):
                sigma_vals[:] = np.where(sigma_vals == 0.0, 1.0, sigma_vals)

        # perturbed ModelStateBase
        perturb_ms = self + sigma * direction
        perturb_fcn_fname = os.path.join(
            solver_state.get_workdir(), "perturb_fcn_" + os.path.basename(res_fname)
        )
        perturb_fcn = perturb_ms.comp_fcn(perturb_fcn_fname, solver_state)

        # compute finite difference
        caller = class_name(self) + ".comp_jacobian_fcn_state_prod"
        res = ((perturb_fcn - fcn) / sigma).dump(res_fname, caller)

        solver_state.log_step(fcn_complete_step)

        return res

    def get_tracer_vals(self, tracer_name):
        """get tracer values"""
        for tracer_module in self.tracer_modules:
            try:
                return tracer_module.get_tracer_vals(tracer_name)
            except ValueError:
                pass
        msg = "unknown tracer_name=%s" % tracer_name
        raise ValueError(msg)

    def set_tracer_vals(self, tracer_name, vals):
        """set tracer values"""
        for tracer_module in self.tracer_modules:
            try:
                tracer_module.set_tracer_vals(tracer_name, vals)
            except ValueError:
                pass

    def shadow_tracers_on(self):
        """are any shadow tracers being run"""
        for tracer_module in self.tracer_modules:
            if tracer_module.shadow_tracers_on():
                return True
        return False

    def copy_shadow_tracers_to_real_tracers(self):
        """copy shadow tracers to their real counterparts"""
        for tracer_module in self.tracer_modules:
            tracer_module.copy_shadow_tracers_to_real_tracers()
        return self

    def copy_real_tracers_to_shadow_tracers(self):
        """overwrite shadow tracers with their real counterparts"""
        for tracer_module in self.tracer_modules:
            tracer_module.copy_real_tracers_to_shadow_tracers()
        return self

    def zero_extra_tracers(self):
        """set extra tracers (i.e., not being solved for) to zero"""
        for tracer_module in self.tracer_modules:
            tracer_module.zero_extra_tracers()
        return self

    def apply_region_mask(self):
        """set _vals to zero where region_mask == 0"""
        for tracer_module in self.tracer_modules:
            tracer_module.apply_region_mask()
        return self


def _precond_dims(hist_vars, fptr_in, fptr_out, action):
    """define netCDF4 dimensions needed for hist_vars from hist_fname"""
    vars_metadata = {}
    for hist_var in hist_vars:
        hist_varname, _, time_op = hist_var.partition(":")

        dimensions = _precond_dimensions_for_hist_var(fptr_in, hist_varname, time_op)

        if action == "define":
            create_dimensions_verify(fptr_out, dimensions)

        for dimname in dimensions:
            if dimname in fptr_in.variables and dimname not in vars_metadata:
                vars_metadata[dimname] = {
                    "datatype": fptr_in.variables[dimname].datatype,
                    "dimensions": (dimname,),
                    "attrs": fptr_in.variables[dimname].__dict__,
                }
                if action == "write":
                    fptr_out.variables[dimname][:] = fptr_in.variables[dimname][:]

    if action == "define":
        create_vars(fptr_out, vars_metadata)


def _precond_dimensions_for_hist_var(fptr_hist, hist_varname, time_op):
    """
    return dict of dimensions for hist_varname's representation in the precond file
    """
    dimensions = extract_dimensions(fptr_hist, hist_varname)
    # drop time if dimensional reduction will be applied
    if time_op in ("mean", "log_mean"):
        del dimensions["time"]
    # drop singleton time dimension
    if dimensions.get("time", None) == 1:
        del dimensions["time"]
    return dimensions


def lin_comb(res_type, coeff, fname_fcn, quantity):
    """compute a linear combination of ModelStateBase objects in files"""
    res = coeff[:, 0] * res_type(fname_fcn(quantity, 0))
    for j_val in range(1, coeff.shape[-1]):
        res += coeff[:, j_val] * res_type(fname_fcn(quantity, j_val))
    return res


def _get_tracer_module_state_class(model_name, tracer_module_name, tracer_module_def):
    """return tracer module state class for tracer_module_name"""

    tracer_module_state_class = TracerModuleStateBase

    # look for model specific derived class
    mod_name = ".".join(["src", model_name, "tracer_module_state"])
    subclasses = get_subclasses(mod_name, tracer_module_state_class)
    if len(subclasses) > 0:
        tracer_module_state_class = subclasses[0]

    # look for tracer module specific derived class
    py_mod_name = tracer_module_def.get("py_mod_name", tracer_module_name)
    mod_name = ".".join(["src", model_name, py_mod_name])
    subclasses = get_subclasses(mod_name, tracer_module_state_class)
    if len(subclasses) > 0:
        tracer_module_state_class = subclasses[0]

    return tracer_module_state_class
