"""test functions in spatial_axis.py"""

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
