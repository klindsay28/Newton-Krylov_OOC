"""test functions in spatial_axis.py"""

import numpy as np

from src.spatial_axis import SpatialAxis, spatial_axis_defn_dict


def test_spatial_axis_defn_dict():
    """test spatial_axis_defn_dict"""
    defn_dict = spatial_axis_defn_dict()
    assert defn_dict["axisname"]["value"] == "depth"


def test_spatial_axis_init():
    """test SpatialAxis"""
    defn_dict = spatial_axis_defn_dict()
    depth = SpatialAxis(defn_dict=defn_dict)
    assert depth.axisname == "depth"
    assert len(depth) == 30


def test_remap_linear_interpolant_1pt():
    """test remap_linear_interpolant with 1 value"""
    defn_dict = spatial_axis_defn_dict(edge_end=50.0, nlevs=5, delta_ratio_max=1.0)
    depth = SpatialAxis(defn_dict=defn_dict)

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
    depth = SpatialAxis(defn_dict=defn_dict)

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
