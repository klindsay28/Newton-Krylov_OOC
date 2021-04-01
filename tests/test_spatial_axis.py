"""test functions in spatial_axis.py"""

import os

import numpy as np
import pytest

from nk_ooc.spatial_axis import (
    SpatialAxis,
    spatial_axis_defn_dict,
    spatial_axis_from_defn_dict,
    spatial_axis_from_file,
)
from nk_ooc.utils import mkdir_exist_okay


def gen_test_axis():
    """
    Return SpatialAxis object for test purposes.
    Layer thicknesses are 1, 2, 3, 4.
    """
    axisname = "depth"
    edges = np.array([0.0, 1.0, 3.0, 6.0, 10.0])
    return SpatialAxis(axisname, edges)


def verify_test_axis(axis):
    """verify values in test SpatialAxis object"""
    assert axis.axisname == "depth"
    assert (axis.edges == np.array([0.0, 1.0, 3.0, 6.0, 10.0])).all()
    assert axis.units == "m"
    assert len(axis) == 4
    assert (axis.mid == np.array([0.5, 2.0, 4.5, 8.0])).all()
    assert (axis.delta == np.array([1.0, 2.0, 3.0, 4.0])).all()
    assert (axis.delta_mid == np.array([1.5, 2.5, 3.5])).all()


def test_roundtrip():
    """
    Verify that dumping a SpatialAxis object to a file and reading it back recovers the
    original values. This exercises
    1: creating a SpatialAxis object from layer edge values,
    2: dumping the SpatialAxis object to a file,
    3: creating a SpatialAxis object from a file.
    """

    axis = gen_test_axis()
    verify_test_axis(axis)

    repo_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    workdir = os.path.join(repo_root, "tests", "workdir")
    mkdir_exist_okay(workdir)
    fname = os.path.join(workdir, "test_axis.nc")
    caller = "test_spatial_axis.test_roundtrip"
    axis.dump(fname, caller)

    axisname = axis.axisname
    edges_varname = axis.dump_names["edges"]

    # verify spatial_axis_from_file with or without edges_varname specified
    verify_test_axis(spatial_axis_from_file(fname, axisname, edges_varname))
    verify_test_axis(spatial_axis_from_file(fname, axisname))


def test_int_vals_mid_1d():
    """verify values returned by int_vals_mid for 1d vals"""

    axis = gen_test_axis()

    vals_ones = np.ones(len(axis))

    # verify that length mismatch raises exception
    with pytest.raises(ValueError):
        axis.int_vals_mid(vals_ones[1:], 0)

    # verify integral of ones
    expected = axis.edges[-1] - axis.edges[0]
    assert axis.int_vals_mid(vals_ones, 0) == expected
    assert axis.int_vals_mid(vals_ones, -1) == expected

    # verify integral of midpoints, that midpoint rule for linear functions is exact
    vals = axis.mid
    expected = 0.5 * (axis.edges[-1] ** 2 - axis.edges[0] ** 2)
    assert axis.int_vals_mid(vals, 0) == expected
    assert axis.int_vals_mid(vals, -1) == expected


def test_int_vals_mid_2d():
    """verify values returned by int_vals_mid for 2d vals"""

    axis1 = gen_test_axis()

    # axis2 has delta==1 and length one less than axis1's length
    # this ensures expected length mismatches actually differ
    axis2 = SpatialAxis("ypos", np.arange(len(axis1)))
    assert len(axis2) == len(axis1) - 1
    assert (axis2.delta == 1.0).all()

    vals_ones = np.ones((len(axis1), len(axis2)))

    # verify that length mismatch with 'correct' axis ind raises exception
    with pytest.raises(ValueError):
        axis1.int_vals_mid(vals_ones[1:, :], 0)
    with pytest.raises(ValueError):
        axis2.int_vals_mid(vals_ones[:, 1:], 1)

    # verify that specifying wrong axis ind raises exception
    with pytest.raises(ValueError):
        axis1.int_vals_mid(vals_ones, 1)
    with pytest.raises(ValueError):
        axis2.int_vals_mid(vals_ones, 0)

    # verify integral of ones along axis1
    expected = axis1.edges[-1] - axis1.edges[0]
    assert (axis1.int_vals_mid(vals_ones, 0) == expected).all()
    assert (axis1.int_vals_mid(vals_ones, -2) == expected).all()

    # verify integral of ones along axis2
    expected = axis2.edges[-1] - axis2.edges[0]
    assert (axis2.int_vals_mid(vals_ones, 1) == expected).all()
    assert (axis2.int_vals_mid(vals_ones, -1) == expected).all()


def test_spatial_axis_defn_dict():
    """test spatial_axis_defn_dict"""
    defn_dict = spatial_axis_defn_dict()
    assert defn_dict["axisname"]["value"] == "depth"


def test_spatial_axis_from_defn_dict():
    """test spatial_axis_from_defn_dict"""
    defn_dict = spatial_axis_defn_dict()
    depth = spatial_axis_from_defn_dict(defn_dict=defn_dict)
    assert depth.axisname == "depth"
    assert len(depth) == 30


def test_remap_linear_interpolant_1pt():
    """test remap_linear_interpolant with 1 value"""
    defn_dict = spatial_axis_defn_dict(edge_end=50.0, nlevs=5, delta_ratio_max=1.0)
    depth = spatial_axis_from_defn_dict(defn_dict=defn_dict)

    xvals = np.array([-5.0])
    yvals = np.ones(1)
    assert (depth.remap_linear_interpolant(xvals, yvals) == np.ones(5)).all()

    xvals = np.array([25.0])
    yvals = np.ones(1)
    assert (depth.remap_linear_interpolant(xvals, yvals) == np.ones(5)).all()

    xvals = np.array([55.0])
    yvals = np.ones(1)
    assert (depth.remap_linear_interpolant(xvals, yvals) == np.ones(5)).all()


def test_remap_linear_interpolant_2pt():
    """test remap_linear_interpolant with 1 value"""
    defn_dict = spatial_axis_defn_dict(edge_end=50.0, nlevs=5, delta_ratio_max=1.0)
    depth = spatial_axis_from_defn_dict(defn_dict=defn_dict)

    xvals = np.array([-15.0, -5.0])
    yvals = np.array([1.0, 2.0])
    assert (depth.remap_linear_interpolant(xvals, yvals) == np.full(5, 2.0)).all()

    xvals = np.array([-15.0, 25.0])
    yvals = np.array([0.0, 8.0])
    res = depth.remap_linear_interpolant(xvals, yvals)
    print(res)
    expected = np.array([4.0, 6.0, 7.75, 8.0, 8.0])
    assert (res == expected).all()

    xvals = np.array([5.0, 25.0])
    yvals = np.array([0.0, 8.0])
    res = depth.remap_linear_interpolant(xvals, yvals)
    print(res)
    expected = np.array([0.5, 4.0, 7.5, 8.0, 8.0])
    assert (res == expected).all()

    xvals = np.array([22.5, 27.5])
    yvals = np.array([0.0, 8.0])
    res = depth.remap_linear_interpolant(xvals, yvals)
    print(res)
    expected = np.array([0.0, 0.0, 4.0, 8.0, 8.0])
    assert (res == expected).all()

    xvals = np.array([42.5, 47.5])
    yvals = np.array([0.0, 8.0])
    res = depth.remap_linear_interpolant(xvals, yvals)
    print(res)
    expected = np.array([0.0, 0.0, 0.0, 0.0, 4.0])
    assert (res == expected).all()

    xvals = np.array([45.0, 55.0])
    yvals = np.array([0.0, 8.0])
    res = depth.remap_linear_interpolant(xvals, yvals)
    print(res)
    expected = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
    assert (res == expected).all()
