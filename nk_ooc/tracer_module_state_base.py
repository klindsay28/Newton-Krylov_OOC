"""base class for representing tracer modules, and operations on them"""

import copy
import logging

import numpy as np

from . import utils


class TracerModuleStateBase:
    """
    Base class for representing a collection of model tracers.
    Derived classes should implement _read_vals and dump.
    """

    # give TracerModuleStateBase operators higher priority than those of numpy
    __array_priority__ = 100

    model_config_obj = None

    def __init__(self, tracer_module_name, fname, model_config_obj):
        logger = logging.getLogger(__name__)
        logger.debug(
            'TracerModuleStateBase, tracer_module_name="%s", fname="%s"',
            tracer_module_name,
            fname,
        )

        TracerModuleStateBase.model_config_obj = model_config_obj

        self.name = tracer_module_name
        self._tracer_module_def = self.model_config_obj.tracer_module_defs[
            tracer_module_name
        ]
        self.tracer_cnt = len(self._tracer_module_def["tracers"])
        # units common to all tracers
        self.units = utils.attr_common(self._tracer_module_def["tracers"], "units")
        self._vals, self._dimensions = self._read_vals(fname)

    def _read_vals(self, fname):
        """
        return tracer values and dimension names and lengths, read from fname)
        implemented in derived classes
        """
        raise NotImplementedError("Method must be implemented in derived class")

    def dump(self, fptr, action):
        """
        perform an action (define or write) of dumping a TracerModuleState object
        to an open file
        implemented in derived classes
        """
        raise NotImplementedError("Method must be implemented in derived class")

    def stats_dimensions(self, fptr):
        """return dimensions to be used in stats file for tracer module"""
        raise NotImplementedError("Method must be implemented in derived class")

    def stats_vars_metadata(self, fptr_hist):
        """
        return dict of metadata for vars to appear in the stats file for this tracer
        module
        """
        raise NotImplementedError("Method must be implemented in derived class")

    def stats_vars_vals_iteration_invariant(self, fptr_hist):
        """return iteration-invariant tracer module specific stats variables"""
        raise NotImplementedError("Method must be implemented in derived class")

    def tracer_names(self):
        """return list of tracer names"""
        return list(self._tracer_module_def["tracers"])

    def tracer_index(self, tracer_name):
        """return the index of a tracer"""
        return self.tracer_names().index(tracer_name)

    def stats_vars_tracer_like(self):
        """
        return list of tracer-like vars in hist file to be processed for the stats file
        """
        return self.tracer_names()

    def apply_limiter(self, base):
        """
        apply limiter scalef to self to ensure base + scalef * self is within bounds
        return scalef value
        """
        logger = logging.getLogger(__name__)

        if not self.has_bounds():
            return 1.0

        scalef = 1.0
        for tracer_name in self._tracer_module_def["tracers"]:
            lob, upb = self.get_bounds(tracer_name)
            if lob is None and upb is None:
                continue
            self_vals = self.get_tracer_vals(tracer_name)
            base_vals = base.get_tracer_vals(tracer_name)
            scalef = min(scalef, utils.comp_scalef_lob(base_vals, self_vals, lob))
            scalef = min(scalef, utils.comp_scalef_upb(base_vals, self_vals, upb))

        if scalef < 1.0:
            logger.info("applying scalef[%s]=%e", self.name, scalef)
            self *= scalef

        return scalef

    def has_bounds(self):
        """Return if bounds applied to this tracer module."""
        if "bounds" in self._tracer_module_def:
            return True
        for tracer_metadata in self._tracer_module_def["tracers"].values():
            if "bounds" in tracer_metadata:
                return True
        return False

    def get_bounds(self, tracer_name):
        """
        Return tuple of lower and uppoer bounds for tracer_name.
        Note that either, or both, of the the returned values can be None,
        indicating that the corresponding bound is not specified.
        """
        lob, upb = None, None
        for metadata in [
            self._tracer_module_def,
            self._tracer_module_def["tracers"][tracer_name],
        ]:
            if "bounds" in metadata:
                lob = metadata["bounds"].get("lob", lob)
                upb = metadata["bounds"].get("upb", upb)
        return lob, upb

    def log_vals(self, msg, vals):
        """write per-tracer module values to the log"""
        logger = logging.getLogger(__name__)

        # suppress printing of last index if its span is 1
        if vals.ndim >= 1 and vals.shape[-1] == 1:
            self.log_vals(msg, vals[..., 0])
            return

        if vals.ndim == 0:
            logger.info("%s[%s]=%e", msg, self.name, vals)
        elif vals.ndim == 1:
            for j in range(vals.shape[0]):
                logger.info("%s[%s,%d]=%e", msg, self.name, j, vals[j])
        elif vals.ndim == 2:
            for i in range(vals.shape[0]):
                for j in range(vals.shape[1]):
                    logger.info("%s[%s,%d,%d]=%e", msg, self.name, i, j, vals[i, j])
        else:
            msg = f"vals.ndim={vals.ndim} not handled"
            raise ValueError(msg)

    def __neg__(self):
        """
        unary negation operator
        called to evaluate res = -self
        """
        res = copy.copy(self)
        res._vals = -self._vals
        return res

    def __add__(self, other):
        """
        addition operator
        called to evaluate res = self + other
        """
        res = copy.copy(self)
        if isinstance(other, TracerModuleStateBase):
            res._vals = self._vals + other._vals
        else:
            return NotImplemented
        return res

    def __iadd__(self, other):
        """
        inplace addition operator
        called to evaluate self += other
        """
        if isinstance(other, TracerModuleStateBase):
            self._vals += other._vals
        else:
            return NotImplemented
        return self

    def __sub__(self, other):
        """
        subtraction operator
        called to evaluate res = self - other
        """
        res = copy.copy(self)
        if isinstance(other, TracerModuleStateBase):
            res._vals = self._vals - other._vals
        else:
            return NotImplemented
        return res

    def __isub__(self, other):
        """
        inplace subtraction operator
        called to evaluate self -= other
        """
        if isinstance(other, TracerModuleStateBase):
            self._vals -= other._vals
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
            res._vals = self._vals * other
        elif isinstance(other, np.ndarray):
            if other.shape == (self.model_config_obj.region_cnt,):
                res._vals = self._vals * self.broadcast_region_vals(other)
            else:
                return NotImplemented
        elif isinstance(other, TracerModuleStateBase):
            res._vals = self._vals * other._vals
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
        elif isinstance(other, np.ndarray):
            if other.shape == (self.model_config_obj.region_cnt,):
                self._vals *= self.broadcast_region_vals(other)
            else:
                return NotImplemented
        elif isinstance(other, TracerModuleStateBase):
            self._vals *= other._vals
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
            res._vals = self._vals * (1.0 / other)
        elif isinstance(other, np.ndarray):
            if other.shape == (self.model_config_obj.region_cnt,):
                res._vals = self._vals * self.broadcast_region_vals(1.0 / other)
            else:
                return NotImplemented
        elif isinstance(other, TracerModuleStateBase):
            res._vals = self._vals / other._vals
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
            res._vals = other / self._vals
        elif isinstance(other, np.ndarray):
            if other.shape == (self.model_config_obj.region_cnt,):
                res._vals = self.broadcast_region_vals(other) / self._vals
            else:
                return NotImplemented
        else:
            return NotImplemented
        return res

    def __itruediv__(self, other):
        """
        inplace division operator
        called to evaluate self /= other
        """
        if isinstance(other, float):
            self._vals *= 1.0 / other
        elif isinstance(other, np.ndarray):
            if other.shape == (self.model_config_obj.region_cnt,):
                self._vals *= self.broadcast_region_vals(1.0 / other)
            else:
                return NotImplemented
        elif isinstance(other, TracerModuleStateBase):
            self._vals /= other._vals
        else:
            return NotImplemented
        return self

    def mean(self):
        """compute weighted mean of self"""
        matrix = self.model_config_obj.region_mean_sparse
        res = np.zeros(matrix.shape[0])
        for tracer_ind in range(self.tracer_cnt):
            res += matrix.dot(self._vals[tracer_ind, ...].reshape(-1))
        return np.array(res)

    def dot_prod(self, other):
        """compute weighted dot product of self with other"""
        matrix = self.model_config_obj.region_mean_sparse
        res = np.zeros(matrix.shape[0])
        for tracer_ind in range(self.tracer_cnt):
            res += matrix.dot(
                self._vals[tracer_ind, ...].reshape(-1)
                * other._vals[  # pylint: disable=protected-access
                    tracer_ind, ...
                ].reshape(-1)
            )
        return np.array(res)

    def precond_matrix_list(self):
        """Return list of precond matrices being used"""
        res = []
        for tracer_metadata in self._tracer_module_def["tracers"].values():
            if "precond_matrix" in tracer_metadata:
                precond_matrix_name = tracer_metadata["precond_matrix"]
                if precond_matrix_name not in res:
                    res.append(precond_matrix_name)
        return res

    def append_tracer_names_per_precond_matrix(self, res):
        """Append tracer names for each precond matrix to res"""
        # process tracers in order of tracer_names
        for tracer_name, tracer_metadata in self._tracer_module_def["tracers"].items():
            if "precond_matrix" in tracer_metadata:
                precond_matrix_name = tracer_metadata["precond_matrix"]
                if precond_matrix_name not in res:
                    res[precond_matrix_name] = [tracer_name]
                else:
                    res[precond_matrix_name].append(tracer_name)

    def get_tracer_vals_all(self):
        """get all tracer values"""
        return self._vals

    def set_tracer_vals_all(self, vals, reseat_vals=False):
        """set all tracer values"""
        if reseat_vals:
            self._vals = vals
        else:
            self._vals[:] = vals

    def get_tracer_vals(self, tracer_name):
        """get tracer values"""
        return self._vals[self.tracer_index(tracer_name), ...]

    def set_tracer_vals(self, tracer_name, vals):
        """set tracer values"""
        self._vals[self.tracer_index(tracer_name), ...] = vals

    def shadow_tracers_on(self):
        """are any shadow tracers being run"""
        for tracer_metadata in self._tracer_module_def["tracers"].values():
            if "shadows" in tracer_metadata:
                return True
        return False

    def copy_shadow_tracers_to_real_tracers(self):
        """copy shadow tracers to their real counterparts"""
        for tracer_name, tracer_metadata in self._tracer_module_def["tracers"].items():
            if "shadows" in tracer_metadata:
                self.set_tracer_vals(
                    tracer_metadata["shadows"], self.get_tracer_vals(tracer_name)
                )

    def copy_real_tracers_to_shadow_tracers(self):
        """overwrite shadow tracers with their real counterparts"""
        for tracer_name, tracer_metadata in self._tracer_module_def["tracers"].items():
            if "shadows" in tracer_metadata:
                self.set_tracer_vals(
                    tracer_name, self.get_tracer_vals(tracer_metadata["shadows"])
                )

    def extra_tracer_inds(self):
        """
        return list of indices of tracers that are extra
            (i.e., they are not being solved for)
        the indices are with respect to self
        tracers that are shadowed are automatically extra
        """
        res = []
        for tracer_metadata in self._tracer_module_def["tracers"].values():
            if "shadows" in tracer_metadata:
                res.append(self.tracer_index(tracer_metadata["shadows"]))
        return res

    def zero_extra_tracers(self):
        """set extra tracers (i.e., not being solved for) to zero"""
        for tracer_ind in self.extra_tracer_inds():
            self._vals[tracer_ind, ...] = 0.0

    def apply_region_mask(self):
        """set _vals to zero where region_mask == 0"""
        region_mask = self.model_config_obj.region_mask
        for tracer_ind in range(self.tracer_cnt):
            self._vals[tracer_ind, ...] = np.where(
                region_mask != 0, self._vals[tracer_ind, ...], 0.0
            )

    def broadcast_region_vals(self, vals, fill_value=1.0):
        """
        broadcast values in vals to an array of same shape as region_mask
        values in the results are:
            fill_value  where region_mask is <= 0
                        (e.g. complement of computational domain)
            vals[ind]   where region_mask == ind+1
        """
        region_mask = self.model_config_obj.region_mask
        res = np.full(shape=region_mask.shape, fill_value=fill_value)
        for region_ind, val in enumerate(vals):
            res = np.where(region_mask == region_ind + 1, val, res)
        return res
