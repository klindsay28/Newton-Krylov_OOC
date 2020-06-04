"""base class for representing tracer modules, and operations on them"""

import copy
import logging

import numpy as np

from . import model_config
from .region_scalars import RegionScalars, to_ndarray
from .utils import attr_common


class TracerModuleStateBase:
    """
    Base class for representing a collection of model tracers.
    Derived classes should implement _read_vals and dump.
    """

    # give TracerModuleStateBase operators higher priority than those of numpy
    __array_priority__ = 100

    def __init__(self, tracer_module_name, fname):
        logger = logging.getLogger(__name__)
        logger.debug(
            'TracerModuleStateBase, tracer_module_name="%s", fname="%s"',
            tracer_module_name,
            fname,
        )
        if model_config.model_config_obj is None:
            msg = (
                "model_config.model_config_obj is None, %s must be called before %s"
                % ("ModelConfig.__init__", "TracerModuleStateBase.__init__")
            )
            raise RuntimeError(msg)
        self.name = tracer_module_name
        self._tracer_module_def = model_config.model_config_obj.tracer_module_defs[
            tracer_module_name
        ]
        # units common to all tracers
        self.units = attr_common(self._tracer_module_def["tracers"], "units")
        self._vals, self._dims = self._read_vals(  # pylint: disable=no-member
            tracer_module_name, fname
        )

    def tracer_names(self):
        """return list of tracer names"""
        return list(self._tracer_module_def["tracers"])

    def tracer_cnt(self):
        """return number of tracers"""
        return len(self._tracer_module_def["tracers"])

    def tracer_index(self, tracer_name):
        """return the index of a tracer"""
        return self.tracer_names().index(tracer_name)

    def tracer_metadata(self, tracer_name):
        """return tracer's metadata"""
        return self._tracer_module_def["tracers"][tracer_name]

    def log_vals(self, msg, vals):
        """write per-tracer module values to the log"""
        logger = logging.getLogger(__name__)

        # simplify subsequent logic by converting implicit RegionScalars dimension
        # to an additional ndarray dimension
        if (
            isinstance(vals, RegionScalars)
            or isinstance(vals, np.ndarray)
            and isinstance(vals.ravel()[0], RegionScalars)
        ):
            self.log_vals(msg, to_ndarray(vals))
            return

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
                    logger.info(
                        "%s[%s,%d,%d]=%e", msg, self.name, i, j, vals[i, j],
                    )
        else:
            msg = "vals.ndim=%d not handled" % vals.ndim
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
        elif isinstance(other, RegionScalars):
            res._vals = self._vals * other.broadcast(
                model_config.model_config_obj.region_mask
            )
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
        elif isinstance(other, RegionScalars):
            self._vals *= other.broadcast(model_config.model_config_obj.region_mask)
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
        elif isinstance(other, RegionScalars):
            res._vals = self._vals * other.recip().broadcast(
                model_config.model_config_obj.region_mask
            )
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
        elif isinstance(other, RegionScalars):
            res._vals = (
                other.broadcast(model_config.model_config_obj.region_mask) / self._vals
            )
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
        elif isinstance(other, RegionScalars):
            self._vals *= other.recip().broadcast(
                model_config.model_config_obj.region_mask
            )
        elif isinstance(other, TracerModuleStateBase):
            self._vals /= other._vals
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
            tmp = np.einsum(
                "ik,jk", model_config.model_config_obj.grid_weight, self._vals
            )
        elif ndim == 2:
            tmp = np.einsum(
                "ikl,jkl", model_config.model_config_obj.grid_weight, self._vals
            )
        else:
            tmp = np.einsum(
                "iklm,jklm", model_config.model_config_obj.grid_weight, self._vals
            )
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
            tmp = np.einsum(
                "ik,jk,jk",
                model_config.model_config_obj.grid_weight,
                self._vals,
                other._vals,  # pylint: disable=protected-access
            )
        elif ndim == 2:
            tmp = np.einsum(
                "ikl,jkl,jkl",
                model_config.model_config_obj.grid_weight,
                self._vals,
                other._vals,  # pylint: disable=protected-access
            )
        else:
            tmp = np.einsum(
                "iklm,jklm,jklm",
                model_config.model_config_obj.grid_weight,
                self._vals,
                other._vals,  # pylint: disable=protected-access
            )
        # return RegionScalars object
        return RegionScalars(tmp)

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
        return self._vals[self.tracer_index(tracer_name), :]

    def set_tracer_vals(self, tracer_name, vals):
        """set tracer values"""
        self._vals[self.tracer_index(tracer_name), :] = vals

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
            self._vals[tracer_ind, :] = 0.0

    def apply_region_mask(self):
        """set _vals to zero where region_mask == 0"""
        for tracer_ind in range(self.tracer_cnt()):
            self._vals[tracer_ind, :] = np.where(
                model_config.model_config_obj.region_mask != 0,
                self._vals[tracer_ind, :],
                0.0,
            )
