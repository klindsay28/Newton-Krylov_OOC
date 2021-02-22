"""class to hold per-region scalars"""

import numpy as np


class RegionScalars:
    """class to hold per-region scalars"""

    region_cnt = 0

    def __init__(self, vals):
        self._vals = np.array(vals)

    def __eq__(self, other):
        """
        equality operator
        called to evaluate res == other
        """
        if isinstance(other, RegionScalars):
            return np.all(other._vals == self._vals)
        return NotImplemented

    def __mul__(self, other):
        """
        multiplication operator
        called to evaluate res = self * other
        """
        if isinstance(other, float):
            return RegionScalars(self._vals * other)
        if isinstance(other, RegionScalars):
            return RegionScalars(self._vals * other._vals)
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
            return RegionScalars(self._vals / other._vals)
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
        """
        return RegionScalars object with reciprocal operator applied to vals in self
        """
        return RegionScalars(1.0 / self._vals)

    def sqrt(self):
        """return RegionScalars object with sqrt applied to vals in self"""
        return RegionScalars(np.sqrt(self._vals))

    def broadcast(self, region_mask, fill_value=1.0):
        """
        broadcast vals from self to an array of same shape as region_mask
        values in the results are:
            fill_value    where region_mask is <= 0
                            (e.g. complement of computational domain)
            _vals[ind]    where region_mask == ind+1
        """
        res = np.full(shape=region_mask.shape, fill_value=fill_value)
        for region_ind in range(self.region_cnt):
            res = np.where(region_mask == region_ind + 1, self._vals[region_ind], res)
        return res


def to_ndarray(array_in):
    """
    Create an ndarray, res, from an ndarray of RegionScalars.
    res.ndim is 1 greater than array_in.ndim.
    The implicit RegionScalars dimension is placed last in res.
    """

    if isinstance(array_in, RegionScalars):
        return np.array(array_in.vals())

    res = np.empty(array_in.shape + (RegionScalars.region_cnt,))

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
        msg = "array_in.ndim=%d not handled" % array_in.ndim
        raise ValueError(msg)

    return res


def to_region_scalar_ndarray(array_in):
    """
    Create an ndarray of RegionScalars, res, from an ndarray.
    res.ndim is 1 less than array_in.ndim.
    The last dimension of array_in corresponds to to implicit RegionScalars dimension in
    res.
    """

    if array_in.shape[-1] != RegionScalars.region_cnt:
        msg = "last dimension must have length %d" % RegionScalars.region_cnt
        raise ValueError(msg)

    res = np.empty(array_in.shape[:-1], dtype=object)

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
        msg = "array_in.ndim=%d not handled" % array_in.ndim
        raise ValueError(msg)

    return res
