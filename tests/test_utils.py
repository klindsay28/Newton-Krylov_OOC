"""test functions in utils.py"""

import os

import pytest

from src import utils


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("1.0 + 2.0", 3.0),
        ("1.0 + 2.0 * 3.0", 7.0),
        ("(1.0 + 2.0) * 3.0", 9.0),
        ("(1.0 + 2.0) / 3.0", 1.0),
        ("2.0 ** 3.0", 8.0),
        ("10.0 + -2.0", 8.0),
        ("10.0 - 2.0", 8.0),
    ],
)
def test_eval_expr(expr, expected):
    """test eval_expr"""
    assert utils.eval_expr(expr) == expected


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
    assert utils.units_str_format(units_str) == expected


def test_isclose_all_vars():
    """test isclose_all_vars"""
    repo_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    input_dir = os.path.join(repo_root, "input", "tests")
    fname_1 = os.path.join(input_dir, "isclose_1.nc")
    fname_2 = os.path.join(input_dir, "isclose_2.nc")
    assert utils.isclose_all_vars(fname_1, fname_2, rtol=1.0e-5, atol=1.0e-5)
