"""test functions in region_scalars.py"""

import numpy as np
import pytest

from nk_ooc.region_scalars import RegionScalars, to_ndarray, to_region_scalar_ndarray

from .share import config_test_problem


@pytest.mark.parametrize("ndim", [0, 1, 2, 3])
def test_to_ndarray(ndim):
    """test to_ndarray for different arg_in ranks"""

    config_test_problem()

    arg_in_shape = tuple(range(3, 3 + ndim))
    arg_in = np.full(arg_in_shape, RegionScalars(1.0))
    expected_shape = arg_in_shape + (1,)
    expected = np.full(expected_shape, [1.0])

    result = to_ndarray(arg_in)
    assert result.shape == expected.shape
    assert np.all(result == expected)


@pytest.mark.parametrize("ndim", [0, 1, 2, 3])
def test_to_region_scalar_ndarray(ndim):
    """test to_region_scalar_ndarray for different arg_in ranks"""

    config_test_problem()

    expected_shape = tuple(range(3, 3 + ndim))
    expected = np.full(expected_shape, RegionScalars(1.0))
    arg_in_shape = expected_shape + (1,)
    arg_in = np.full(arg_in_shape, [1.0])

    result = to_region_scalar_ndarray(arg_in)
    assert result.shape == expected.shape
    assert np.all(result == expected)
