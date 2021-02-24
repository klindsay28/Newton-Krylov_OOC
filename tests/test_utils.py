"""test functions in spatial_axis.py"""

import pytest

from src.utils import units_str_format


@pytest.mark.parametrize(
    "units_str, expected",
    [
        ("years m", "years m"),
        ("mmol / m^3 m", "mmol / m^2"),
        ("mmol / m^3 / d m", "mmol / m^2 / d"),
        ("1 / d m", "m / d"),
        ("mol / m^3 m", "mol / m^2"),
        ("(years) (m)", "years m"),
        ("(mmol / m^3) (m)", "mmol / m^2"),
        ("(mmol / m^3 / d) (m)", "mmol / m^2 / d"),
        ("(1 / d) (m)", "m / d"),
        ("(mol / m^3) (m)", "mol / m^2"),
        ("m years", "years m"),
        ("m mmol / m^3", "mmol / m^2"),
        ("m mmol / m^3 / d", "mmol / m^2 / d"),
        ("m 1 / d", "m / d"),
        ("m mol / m^3", "mol / m^2"),
    ],
)
def test_units_str_format(units_str, expected):
    """test units_str_format"""
    assert units_str_format(units_str) == expected
