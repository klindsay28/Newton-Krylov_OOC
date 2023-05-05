"""base class for representing tracer modules, and operations on them"""

import copy
import logging

import numpy as np
import xarray as xr
from netCDF4 import Dataset

from .utils import attr_common, comp_scalef_lob, comp_scalef_upb, extract_dimensions


class TracerModuleStateBase:
    """
    Base class for representing a collection of model tracers.
    Derived classes should implement dump.
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
        self.units = attr_common(self._tracer_module_def["tracers"], "units")
        self._tracer_varname_suffix = None
        self._dataset = self._load_dataset(fname)

        # Determine if all tracers have same shape, to see if {get,set}_tracer_vals_all
        # should be supported.
        self._tracers_same_shape = True
        for tracer_ind, tracer_name in enumerate(self._tracer_module_def["tracers"]):
            shape = self._dataset[tracer_name].shape
            if tracer_ind == 0:
                shape0 = shape
            elif shape != shape0:
                self._tracers_same_shape = False

    def _load_dataset(self, fname):
        """
        return xarray Dataset of tracer module tracers
        """
        ds = xr.Dataset()
        with Dataset(fname, mode="r") as fptr:
            fptr.set_auto_mask(False)
            for tracer_name in self._tracer_module_def["tracers"]:
                region_mask = self.get_grid_vars(tracer_name)["region_mask"]
                if self._tracer_varname_suffix is not None:
                    varname = f"{tracer_name}_{self._tracer_varname_suffix}"
                else:
                    varname = tracer_name
                dimensions = extract_dimensions(fptr, varname)
                if tuple(dimensions.values()) != region_mask.shape:
                    raise ValueError(
                        f"unexpected dimension lengths for {varname} in {fname}"
                    )
                ds[tracer_name] = xr.DataArray(
                    fptr.variables[varname][:], dims=tuple(dimensions)
                )
        return ds

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

    def stats_vars_tracer_like(self):
        """
        return list of tracer-like vars in hist file to be processed for the stats file
        """
        return list(self._tracer_module_def["tracers"])

    def get_grid_vars(self, tracer_name):
        """return dict of grid_vars for tracer_name"""
        tracer_metadata = self._tracer_module_def["tracers"][tracer_name]
        region_mask_varname = tracer_metadata["region_mask_varname"]
        return self.model_config_obj.grid_vars[region_mask_varname]

    def apply_limiter(self, base):
        """
        apply limiter scalef to self to ensure base + scalef * self is within bounds
        return scalef value
        """
        if not self.has_bounds():
            return 1.0

        scalef = np.ones(self.model_config_obj.region_cnt)
        scalef_tracer = np.ones(self.model_config_obj.region_cnt)
        for tracer_name in self._tracer_module_def["tracers"]:
            region_mask = self.get_grid_vars(tracer_name)["region_mask"]

            lob, upb = self.get_bounds(tracer_name)
            if lob is not None:
                comp_scalef_lob(
                    self.model_config_obj.region_cnt,
                    region_mask,
                    base.get_tracer_vals(tracer_name),
                    self.get_tracer_vals(tracer_name),
                    lob,
                    out=scalef_tracer,
                )
                np.minimum(scalef, scalef_tracer, out=scalef)
            if upb is not None:
                comp_scalef_upb(
                    self.model_config_obj.region_cnt,
                    region_mask,
                    base.get_tracer_vals(tracer_name),
                    self.get_tracer_vals(tracer_name),
                    upb,
                    out=scalef_tracer,
                )
                np.minimum(scalef, scalef_tracer, out=scalef)

        if (scalef < 1.0).any():
            self.log_vals("applying scalef", scalef)
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
        res._dataset = -self._dataset
        return res

    def __add__(self, other):
        """
        addition operator
        called to evaluate res = self + other
        """
        res = copy.copy(self)
        if isinstance(other, TracerModuleStateBase):
            res._dataset = self._dataset + other._dataset
        else:
            return NotImplemented
        return res

    def __iadd__(self, other):
        """
        inplace addition operator
        called to evaluate self += other
        """
        if isinstance(other, TracerModuleStateBase):
            self._dataset += other._dataset
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
            res._dataset = self._dataset - other._dataset
        else:
            return NotImplemented
        return res

    def __isub__(self, other):
        """
        inplace subtraction operator
        called to evaluate self -= other
        """
        if isinstance(other, TracerModuleStateBase):
            self._dataset -= other._dataset
        else:
            return NotImplemented
        return self

    def __mul__(self, other):
        """
        multiplication operator
        called to evaluate res = self * other
        """
        res = copy.copy(self)
        if isinstance(other, (int, float)):
            res._dataset = self._dataset * other
        elif isinstance(other, np.ndarray):
            if other.shape == (self.model_config_obj.region_cnt,):
                res._dataset = self._dataset.copy(deep=True)
                for tracer_name in self._tracer_module_def["tracers"]:
                    vals = res.get_tracer_vals(tracer_name)
                    vals[:] *= res.broadcast_region_vals(other, tracer_name)
                    res.set_tracer_vals(tracer_name, vals)
            else:
                return NotImplemented
        elif isinstance(other, TracerModuleStateBase):
            res._dataset = self._dataset * other._dataset
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
        if isinstance(other, (int, float)):
            self._dataset *= other
        elif isinstance(other, np.ndarray):
            if other.shape == (self.model_config_obj.region_cnt,):
                for tracer_name in self._tracer_module_def["tracers"]:
                    vals = self.get_tracer_vals(tracer_name)
                    vals[:] *= self.broadcast_region_vals(other, tracer_name)
                    self.set_tracer_vals(tracer_name, vals)
            else:
                return NotImplemented
        elif isinstance(other, TracerModuleStateBase):
            self._dataset *= other._dataset
        else:
            return NotImplemented
        return self

    def __truediv__(self, other):
        """
        division operator
        called to evaluate res = self / other
        """
        res = copy.copy(self)
        if isinstance(other, (int, float)):
            res._dataset = self._dataset * (1.0 / other)
        elif isinstance(other, np.ndarray):
            if other.shape == (self.model_config_obj.region_cnt,):
                res._dataset = self._dataset.copy(deep=True)
                for tracer_name in self._tracer_module_def["tracers"]:
                    vals = res.get_tracer_vals(tracer_name)
                    vals[:] *= res.broadcast_region_vals(1.0 / other, tracer_name)
                    res.set_tracer_vals(tracer_name, vals)
            else:
                return NotImplemented
        elif isinstance(other, TracerModuleStateBase):
            res._dataset = self._dataset / other._dataset
        else:
            return NotImplemented
        return res

    def __rtruediv__(self, other):
        """
        reversed division operator
        called to evaluate res = other / self
        """
        res = copy.copy(self)
        if isinstance(other, (int, float)):
            res._dataset = other / self._dataset
        elif isinstance(other, np.ndarray):
            if other.shape == (self.model_config_obj.region_cnt,):
                res._dataset = self._dataset.copy(deep=True)
                for tracer_name in self._tracer_module_def["tracers"]:
                    vals = res.get_tracer_vals(tracer_name)
                    vals[:] = res.broadcast_region_vals(other, tracer_name) / vals[:]
                    res.set_tracer_vals(tracer_name, vals)
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
        if isinstance(other, (int, float)):
            self._dataset *= 1.0 / other
        elif isinstance(other, np.ndarray):
            if other.shape == (self.model_config_obj.region_cnt,):
                for tracer_name in self._tracer_module_def["tracers"]:
                    vals = self.get_tracer_vals(tracer_name)
                    vals[:] *= self.broadcast_region_vals(1.0 / other, tracer_name)
                    self.set_tracer_vals(tracer_name, vals)
            else:
                return NotImplemented
        elif isinstance(other, TracerModuleStateBase):
            self._dataset /= other._dataset
        else:
            return NotImplemented
        return self

    def mean(self):
        """compute weighted mean of self"""
        res = np.zeros(self.model_config_obj.region_cnt)
        for tracer_name in self._tracer_module_def["tracers"]:
            matrix = self.get_grid_vars(tracer_name)["region_comp_mean_matrix"]
            res += matrix.dot(self.get_tracer_vals(tracer_name).reshape(-1))
        return np.array(res)

    def dot_prod(self, other):
        """compute weighted dot product of self with other"""
        res = np.zeros(self.model_config_obj.region_cnt)
        for tracer_name in self._tracer_module_def["tracers"]:
            matrix = self.get_grid_vars(tracer_name)["region_comp_mean_matrix"]
            res += matrix.dot(
                self.get_tracer_vals(tracer_name).reshape(-1)
                * other.get_tracer_vals(tracer_name).reshape(-1)
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
        if not self._tracers_same_shape:
            raise RuntimeError(
                "get_tracer_vals_all not supported if tracers have varying shape, "
                f"name={self.name}"
            )
        return np.stack(
            [
                self.get_tracer_vals(varname)
                for varname in self._tracer_module_def["tracers"]
            ]
        )

    def set_tracer_vals_all(self, vals_all, reseat_vals=False):
        """set all tracer values"""
        if not self._tracers_same_shape:
            raise RuntimeError(
                "get_tracer_vals_all not supported if tracers have varying shape, "
                f"name={self.name}"
            )
        if reseat_vals:
            # create new Dataset using values from vals_all argument and
            # dimensions from existing Dataset
            ds = xr.Dataset()
            for vals, tracer_name in zip(vals_all, self._tracer_module_def["tracers"]):
                ds[tracer_name] = xr.DataArray(
                    vals, dims=self._dataset[tracer_name].dims
                )
            self._dataset = ds
        else:
            for vals, tracer_name in zip(vals_all, self._tracer_module_def["tracers"]):
                self.set_tracer_vals(tracer_name, vals)

    def get_tracer_vals(self, tracer_name):
        """get values for tracer with name tracer_name"""
        return self._dataset[tracer_name].values

    def set_tracer_vals(self, tracer_name, vals):
        """set values of tracer with name tracer_name to vals"""
        self._dataset[tracer_name].values[:] = vals

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

    def extra_tracer_names(self):
        """
        return list of names of tracers that are extra
            (i.e., they are not being solved for)
        tracers that are shadowed are automatically extra
        """
        res = []
        for tracer_metadata in self._tracer_module_def["tracers"].values():
            if "shadows" in tracer_metadata:
                res.append(tracer_metadata["shadows"])
        return res

    def zero_extra_tracers(self):
        """set extra tracers (i.e., not being solved for) to zero"""
        for tracer_name in self.extra_tracer_names():
            self.set_tracer_vals(tracer_name, 0.0)
        return self

    def apply_region_mask(self):
        """set tracer values to zero where region_mask == 0"""
        for tracer_name in self._tracer_module_def["tracers"]:
            region_mask = self.get_grid_vars(tracer_name)["region_mask"]
            vals = self.get_tracer_vals(tracer_name)
            vals[:] = np.where(region_mask != 0, vals, 0.0)
            self.set_tracer_vals(tracer_name, vals)

    def broadcast_region_vals(self, vals, tracer_name, fill_value=1.0):
        """
        broadcast values in vals to an array of same shape as region_mask for tracer
        with name tracer_name
        values in the results are:
            fill_value  where region_mask is <= 0
                        (e.g. complement of computational domain)
            vals[ind]   where region_mask == ind+1
        """
        region_mask = self.get_grid_vars(tracer_name)["region_mask"]
        res = np.full(shape=region_mask.shape, fill_value=fill_value)
        for region_ind, val in enumerate(vals):
            res = np.where(region_mask == region_ind + 1, val, res)
        return res
