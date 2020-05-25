"""class for representing the state space of a model, and operations on it"""

import collections
from datetime import datetime
import logging

import numpy as np
from netCDF4 import Dataset

from . import model_config
from .model_config import get_precond_matrix_def, get_modelinfo
from .region_scalars import RegionScalars, to_ndarray
from .stats_file import stats_file_append_vals

################################################################################


class ModelStateBase:
    """class for representing the state space of a model"""

    # give ModelStateBase operators higher priority than those of numpy
    __array_priority__ = 100

    def __init__(self, tracer_module_state_class, vals_fname=None):
        logger = logging.getLogger(__name__)
        logger.debug('ModelStateBase, vals_fname="%s"', vals_fname)
        if model_config.model_config_obj is None:
            msg = (
                "model_config.model_config_obj is None, %s must be called before %s"
                % ("ModelConfig.__init__", "ModelStateBase.__init__",)
            )
            raise RuntimeError(msg)
        if not issubclass(tracer_module_state_class, TracerModuleStateBase):
            msg = (
                "tracer_module_state_class must be a subclass of TracerModuleStateBase"
            )
            raise ValueError(msg)
        self.tracer_module_names = get_modelinfo("tracer_module_names").split(",")
        self.tracer_module_cnt = len(self.tracer_module_names)
        if vals_fname is not None:
            self._tracer_modules = np.empty((self.tracer_module_cnt,), dtype=np.object)
            for tracer_module_ind, tracer_module_name in enumerate(
                self.tracer_module_names
            ):
                self._tracer_modules[tracer_module_ind] = tracer_module_state_class(
                    tracer_module_name, vals_fname=vals_fname
                )

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

    def tracer_metadata(self, tracer_name):
        """return tracer's metadata"""
        for tracer_module in self._tracer_modules:
            try:
                return tracer_module.tracer_metadata(tracer_name)
            except KeyError:
                pass
        msg = "unknown tracer_name=%s" % tracer_name
        raise ValueError(msg)

    def dump(self, vals_fname, caller=None):
        """dump ModelStateBase object to a file"""
        logger = logging.getLogger(__name__)
        logger.debug('vals_fname="%s"', vals_fname)
        with Dataset(vals_fname, mode="w", format="NETCDF3_64BIT_OFFSET") as fptr:
            datestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            name = __name__ + ".ModelStateBase.dump"
            msg = datestamp + ": created by " + name
            if caller is not None:
                msg = msg + " called from " + caller
            else:
                raise ValueError("caller unknown")
            setattr(fptr, "history", msg)
            for action in ["define", "write"]:
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
                        tracer_module.log_vals(
                            submsg, vals[msg_ind, tracer_module_ind, :]
                        )
            else:
                if vals.ndim == 1:
                    tracer_module.log_vals(msg, vals[tracer_module_ind])
                else:
                    tracer_module.log_vals(msg, vals[tracer_module_ind, :])

    def log(self, msg=None, stats_info=None):
        """write info of the instance to the log"""
        if msg is None:
            msg_full = ["mean", "norm"]
        else:
            msg_full = [msg + ",mean", msg + ",norm"]
        mean_vals = self.mean()
        norm_vals = self.norm()
        self.log_vals(msg_full, np.stack((mean_vals, norm_vals)))

        if stats_info is not None and stats_info["append_vals"]:
            stats_file_append_vals(
                stats_info["fname"],
                stats_info["iteration"],
                stats_info["varname_root"] + "_mean",
                mean_vals,
            )
            stats_file_append_vals(
                stats_info["fname"],
                stats_info["iteration"],
                stats_info["varname_root"] + "_norm",
                norm_vals,
            )

    def copy(self):
        """return a copy of self"""
        res = type(self)()  # pylint: disable=E1120
        res._tracer_modules = np.empty(
            (self.tracer_module_cnt,), dtype=np.object
        )  # pylint: disable=W0212
        for tracer_module_ind, tracer_module in enumerate(self._tracer_modules):
            res._tracer_modules[
                tracer_module_ind
            ] = tracer_module.copy()  # pylint: disable=W0212
        return res

    def __neg__(self):
        """
        unary negation operator
        called to evaluate res = -self
        """
        res = type(self)()  # pylint: disable=E1120
        res._tracer_modules = -self._tracer_modules  # pylint: disable=W0212
        return res

    def __add__(self, other):
        """
        addition operator
        called to evaluate res = self + other
        """
        res = type(self)()  # pylint: disable=E1120
        if isinstance(other, ModelStateBase):
            res._tracer_modules = (
                self._tracer_modules + other._tracer_modules
            )  # pylint: disable=W0212
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
            self._tracer_modules += other._tracer_modules  # pylint: disable=W0212
        else:
            return NotImplemented
        return self

    def __sub__(self, other):
        """
        subtraction operator
        called to evaluate res = self - other
        """
        res = type(self)()  # pylint: disable=E1120
        if isinstance(other, ModelStateBase):
            res._tracer_modules = (
                self._tracer_modules - other._tracer_modules
            )  # pylint: disable=W0212
        else:
            return NotImplemented
        return res

    def __isub__(self, other):
        """
        inplace subtraction operator
        called to evaluate self -= other
        """
        if isinstance(other, ModelStateBase):
            self._tracer_modules -= other._tracer_modules  # pylint: disable=W0212
        else:
            return NotImplemented
        return self

    def __mul__(self, other):
        """
        multiplication operator
        called to evaluate res = self * other
        """
        res = type(self)()  # pylint: disable=E1120
        if isinstance(other, float):
            res._tracer_modules = self._tracer_modules * other  # pylint: disable=W0212
        elif (
            isinstance(other, np.ndarray) and other.shape == self._tracer_modules.shape
        ):
            res._tracer_modules = self._tracer_modules * other  # pylint: disable=W0212
        elif isinstance(other, ModelStateBase):
            res._tracer_modules = (
                self._tracer_modules * other._tracer_modules
            )  # pylint: disable=W0212
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
        elif (
            isinstance(other, np.ndarray) and other.shape == self._tracer_modules.shape
        ):
            self._tracer_modules *= other
        elif isinstance(other, ModelStateBase):
            self._tracer_modules *= other._tracer_modules  # pylint: disable=W0212
        else:
            return NotImplemented
        return self

    def __truediv__(self, other):
        """
        division operator
        called to evaluate res = self / other
        """
        res = type(self)()  # pylint: disable=E1120
        if isinstance(other, float):
            res._tracer_modules = self._tracer_modules * (
                1.0 / other
            )  # pylint: disable=W0212
        elif (
            isinstance(other, np.ndarray) and other.shape == self._tracer_modules.shape
        ):
            res._tracer_modules = self._tracer_modules * (
                1.0 / other
            )  # pylint: disable=W0212
        elif isinstance(other, ModelStateBase):
            res._tracer_modules = (
                self._tracer_modules / other._tracer_modules
            )  # pylint: disable=W0212
        else:
            return NotImplemented
        return res

    def __rtruediv__(self, other):
        """
        reversed division operator
        called to evaluate res = other / self
        """
        res = type(self)()  # pylint: disable=E1120
        if isinstance(other, float):
            res._tracer_modules = other / self._tracer_modules  # pylint: disable=W0212
        elif (
            isinstance(other, np.ndarray) and other.shape == self._tracer_modules.shape
        ):
            res._tracer_modules = other / self._tracer_modules  # pylint: disable=W0212
        else:
            return NotImplemented
        return res

    def __itruediv__(self, other):
        """
        inplace division operator
        called to evaluate self /= other
        """
        if isinstance(other, float):
            self._tracer_modules *= 1.0 / other
        elif (
            isinstance(other, np.ndarray) and other.shape == self._tracer_modules.shape
        ):
            self._tracer_modules *= 1.0 / other
        elif isinstance(other, ModelStateBase):
            self._tracer_modules /= other._tracer_modules  # pylint: disable=W0212
        else:
            return NotImplemented
        return self

    def mean(self):
        """compute weighted mean of self"""
        res = np.empty(self._tracer_modules.shape, dtype=np.object)
        for ind, tracer_module in enumerate(self._tracer_modules):
            res[ind] = tracer_module.mean()
        return res

    def dot_prod(self, other):
        """compute weighted dot product of self with other"""
        res = np.empty(self._tracer_modules.shape, dtype=np.object)
        for ind, tracer_module in enumerate(self._tracer_modules):
            res[ind] = tracer_module.dot_prod(
                other._tracer_modules[ind]
            )  # pylint: disable=W0212
        return res

    def norm(self):
        """compute weighted l2 norm of self"""
        return np.sqrt(self.dot_prod(self))

    def mod_gram_schmidt(self, basis_cnt, fname_fcn, quantity):
        """
        inplace modified Gram-Schmidt projection
        return projection coefficients
        """
        h_val = np.empty((self.tracer_module_cnt, basis_cnt), dtype=np.object)
        for i_val in range(0, basis_cnt):
            basis_i = type(self)(fname_fcn(quantity, i_val))
            h_val[:, i_val] = self.dot_prod(basis_i)
            self -= h_val[:, i_val] * basis_i
        return h_val

    def hist_vars_for_precond_list(self):
        """Return list of hist vars needed for preconditioner of jacobian of comp_fcn"""
        res = []
        for matrix_name in self.precond_matrix_list() + ["base"]:
            precond_matrix_def = get_precond_matrix_def(matrix_name)
            if "hist_to_precond_var_names" in precond_matrix_def:
                for var_name in precond_matrix_def["hist_to_precond_var_names"]:
                    if var_name not in res:
                        res.append(var_name)
        return res

    def precond_matrix_list(self):
        """Return list of precond matrices being used"""
        res = []
        for tracer_module in self._tracer_modules:
            res.extend(tracer_module.precond_matrix_list())
        return res

    def tracer_names_per_precond_matrix(self):
        """Return OrderedDict of tracer names for each precond matrix"""
        res = collections.OrderedDict()
        for tracer_module in self._tracer_modules:
            tracer_module.append_tracer_names_per_precond_matrix(res)
        return res

    def get_tracer_vals(self, tracer_name):
        """get tracer values"""
        for tracer_module in self._tracer_modules:
            try:
                return tracer_module.get_tracer_vals(tracer_name)
            except ValueError:
                pass
        msg = "unknown tracer_name=%s" % tracer_name
        raise ValueError(msg)

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

    def apply_region_mask(self):
        """set _vals to zero where region_mask == 0"""
        for tracer_module in self._tracer_modules:
            tracer_module.apply_region_mask()
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
        logger.debug(
            'TracerModuleStateBase, tracer_module_name="%s", vals_fname="%s"',
            tracer_module_name,
            vals_fname,
        )
        if model_config.model_config_obj is None:
            msg = (
                "model_config.model_config_obj is None, %s must be called before %s"
                % ("ModelConfig.__init__", "TracerModuleStateBase.__init__",)
            )
            raise RuntimeError(msg)
        self._tracer_module_name = tracer_module_name
        self._tracer_module_def = model_config.model_config_obj.tracer_module_defs[
            tracer_module_name
        ]
        if (dims is None) == (vals_fname is None):
            msg = "exactly one of dims and vals_fname must be passed"
            raise ValueError(msg)
        if dims is not None:
            self._dims = dims
        if vals_fname is not None:
            self._vals, self._dims = self._read_vals(  # pylint: disable=E1111
                tracer_module_name, vals_fname
            )

    def _read_vals(self, tracer_module_name, vals_fname):
        """return tracer values and dimension names and lengths, read from vals_fname)"""
        msg = "% should be implemented in classes derived from %s" % (
            "_read_vals",
            "TracerModuleStateBase",
        )
        raise NotImplementedError(msg)

    def tracer_names(self):
        """return list of tracer names"""
        return list(self._tracer_module_def.keys())

    def tracer_cnt(self):
        """return number of tracers"""
        return len(self._tracer_module_def)

    def tracer_index(self, tracer_name):
        """return the index of a tracer"""
        return self.tracer_names().index(tracer_name)

    def tracer_metadata(self, tracer_name):
        """return tracer's metadata"""
        return self._tracer_module_def[tracer_name]

    def dump(self, fptr, action):
        """
        perform an action (define or write) of dumping a TracerModuleStateBase object
        to an open file
        """
        msg = "% should be implemented in classes derived from %s" % (
            "dump",
            "TracerModuleStateBase",
        )
        raise NotImplementedError(msg)

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
            logger.info("%s[%s]=%e", msg, self._tracer_module_name, vals)
        elif vals.ndim == 1:
            for j in range(vals.shape[0]):
                logger.info("%s[%s,%d]=%e", msg, self._tracer_module_name, j, vals[j])
        elif vals.ndim == 2:
            for i in range(vals.shape[0]):
                for j in range(vals.shape[1]):
                    logger.info(
                        "%s[%s,%d,%d]=%e",
                        msg,
                        self._tracer_module_name,
                        i,
                        j,
                        vals[i, j],
                    )
        else:
            msg = "vals.ndim=%d not handled" % vals.ndim
            raise ValueError(msg)

    def copy(self):
        """return a copy of self"""
        res = type(self)(self._tracer_module_name, dims=self._dims)
        res._vals = np.copy(self._vals)  # pylint: disable=W0212
        return res

    def __neg__(self):
        """
        unary negation operator
        called to evaluate res = -self
        """
        res = type(self)(self._tracer_module_name, dims=self._dims)
        res._vals = -self._vals  # pylint: disable=W0212
        return res

    def __add__(self, other):
        """
        addition operator
        called to evaluate res = self + other
        """
        res = type(self)(self._tracer_module_name, dims=self._dims)
        if isinstance(other, TracerModuleStateBase):
            res._vals = self._vals + other._vals  # pylint: disable=W0212
        else:
            return NotImplemented
        return res

    def __iadd__(self, other):
        """
        inplace addition operator
        called to evaluate self += other
        """
        if isinstance(other, TracerModuleStateBase):
            self._vals += other._vals  # pylint: disable=W0212
        else:
            return NotImplemented
        return self

    def __sub__(self, other):
        """
        subtraction operator
        called to evaluate res = self - other
        """
        res = type(self)(self._tracer_module_name, dims=self._dims)
        if isinstance(other, TracerModuleStateBase):
            res._vals = self._vals - other._vals  # pylint: disable=W0212
        else:
            return NotImplemented
        return res

    def __isub__(self, other):
        """
        inplace subtraction operator
        called to evaluate self -= other
        """
        if isinstance(other, TracerModuleStateBase):
            self._vals -= other._vals  # pylint: disable=W0212
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
            res._vals = self._vals * other  # pylint: disable=W0212
        elif isinstance(other, RegionScalars):
            res._vals = self._vals * other.broadcast(  # pylint: disable=W0212
                model_config.model_config_obj.region_mask
            )
        elif isinstance(other, TracerModuleStateBase):
            res._vals = self._vals * other._vals  # pylint: disable=W0212
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
            self._vals *= other._vals  # pylint: disable=W0212
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
            res._vals = self._vals * (1.0 / other)  # pylint: disable=W0212
        elif isinstance(other, RegionScalars):
            res._vals = self._vals * other.recip().broadcast(  # pylint: disable=W0212
                model_config.model_config_obj.region_mask
            )
        elif isinstance(other, TracerModuleStateBase):
            res._vals = self._vals / other._vals  # pylint: disable=W0212
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
            res._vals = other / self._vals  # pylint: disable=W0212
        elif isinstance(other, RegionScalars):
            res._vals = (
                other.broadcast(
                    model_config.model_config_obj.region_mask
                )  # pylint: disable=W0212
                / self._vals
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
            self._vals /= other._vals  # pylint: disable=W0212
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
                other._vals,
            )  # pylint: disable=W0212
        elif ndim == 2:
            tmp = np.einsum(
                "ikl,jkl,jkl",
                model_config.model_config_obj.grid_weight,
                self._vals,
                other._vals,
            )  # pylint: disable=W0212
        else:
            tmp = np.einsum(
                "iklm,jklm,jklm",
                model_config.model_config_obj.grid_weight,
                self._vals,
                other._vals,
            )  # pylint: disable=W0212
        # return RegionScalars object
        return RegionScalars(tmp)

    def precond_matrix_list(self):
        """Return list of precond matrices being used"""
        res = []
        for tracer_metadata in self._tracer_module_def.values():
            if "precond_matrix" in tracer_metadata:
                precond_matrix_name = tracer_metadata["precond_matrix"]
                if precond_matrix_name not in res:
                    res.append(precond_matrix_name)
        return res

    def append_tracer_names_per_precond_matrix(self, res):
        """Append tracer names for each precond matrix to res"""
        # process tracers in order of tracer_names
        for tracer_name, tracer_metadata in self._tracer_module_def.items():
            if "precond_matrix" in tracer_metadata:
                precond_matrix_name = tracer_metadata["precond_matrix"]
                if precond_matrix_name not in res:
                    res[precond_matrix_name] = [tracer_name]
                else:
                    res[precond_matrix_name].append(tracer_name)

    def get_tracer_vals(self, tracer_name):
        """get tracer values"""
        return self._vals[self.tracer_index(tracer_name), :]

    def set_tracer_vals(self, tracer_name, vals):
        """set tracer values"""
        self._vals[self.tracer_index(tracer_name), :] = vals

    def shadow_tracers_on(self):
        """are any shadow tracers being run"""
        for tracer_metadata in self._tracer_module_def.values():
            if "shadows" in tracer_metadata:
                return True
        return False

    def copy_shadow_tracers_to_real_tracers(self):
        """copy shadow tracers to their real counterparts"""
        for tracer_name, tracer_metadata in self._tracer_module_def.items():
            if "shadows" in tracer_metadata:
                self.set_tracer_vals(
                    tracer_metadata["shadows"], self.get_tracer_vals(tracer_name)
                )

    def copy_real_tracers_to_shadow_tracers(self):
        """overwrite shadow tracers with their real counterparts"""
        for tracer_name, tracer_metadata in self._tracer_module_def.items():
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
        for tracer_metadata in self._tracer_module_def.values():
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


################################################################################


def lin_comb(res_type, coeff, fname_fcn, quantity):
    """compute a linear combination of ModelStateBase objects in files"""
    res = coeff[:, 0] * res_type(fname_fcn(quantity, 0))
    for j_val in range(1, coeff.shape[-1]):
        res += coeff[:, j_val] * res_type(fname_fcn(quantity, j_val))
    return res
