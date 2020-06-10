"""test functions in spatial_axis.py"""

import numpy as np
import pytest

from src.test_problem.spatial_axis import (
    SpatialAxis,
    spatial_axis_defn_dict,
)


def test_spatial_axis_defn_dict():
    """test spatial_axis_defn_dict"""
    defn_dict = spatial_axis_defn_dict()
    assert defn_dict["axisname"]["value"] == "depth"


def test_SpatialAxis():
    """test SpatialAxis"""
    defn_dict = spatial_axis_defn_dict()
    depth = SpatialAxis(defn_dict=defn_dict)
    assert depth.axisname == "depth"
    assert len(depth) == 30


@pytest.mark.parametrize(
    "vals_units, expected",
    [
        ("years", "years m"),
        ("mmol / m^3", "mmol / m^2"),
        ("mmol / m^3 / d", "mmol / m^2 / d"),
        ("1 / d", "m / d"),
        ("mol / m^3", "mol / m^2"),
    ],
)
def test_int_vals_mid_units(vals_units, expected):
    """test int_vals_mid_units"""
    defn_dict = spatial_axis_defn_dict(units="m")
    depth = SpatialAxis(defn_dict=defn_dict)
    assert depth.int_vals_mid_units(vals_units) == expected
