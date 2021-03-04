"""test functions in spatial_axis.py"""

import os

import git
import numpy as np

from src.spatial_axis import (
    SpatialAxis,
    spatial_axis_defn_dict,
    spatial_axis_from_defn_dict,
    spatial_axis_from_file,
)
from src.utils import mkdir_exist_okay


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

    repo_root = git.Repo(search_parent_directories=True).working_dir
    workdir = os.path.join(repo_root, "tests", "workdir")
    mkdir_exist_okay(workdir)
    fname = os.path.join(workdir, "test_axis.nc")
    caller = "test_spatial_axis.test_roundtrip"
    axis.dump(fname, caller)

    axisname = axis.axisname
    edges_varname = axis.dump_names["edges"]
    axis_new = spatial_axis_from_file(fname, axisname, edges_varname)

    verify_test_axis(axis_new)


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
