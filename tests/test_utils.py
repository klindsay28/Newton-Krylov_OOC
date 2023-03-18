"""test functions in utils.py"""

import os

import numpy as np
import pytest

from nk_ooc import utils


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
    fname_1 = os.path.join(input_dir, "isclose_base.nc")

    # vars are identical to themselves
    assert utils.isclose_all_vars(fname_1, fname_1, rtol=0.0, atol=0.0)
    assert utils.isclose_all_vars(fname_1, fname_1, rtol=1.0e-5, atol=1.0e-5)

    # Values in isclose_same.nc on disk differ from those in isclose_base.nc, but they
    # are the same when units are taken into account. The values, before and after the
    # change of units, are exactly representable in floating point arithmetic.
    fname_2 = os.path.join(input_dir, "isclose_same.nc")
    assert utils.isclose_all_vars(fname_1, fname_2, rtol=0.0, atol=0.0)
    assert utils.isclose_all_vars(fname_1, fname_2, rtol=1.0e-5, atol=1.0e-5)

    # Values in isclose_diff.nc differ from those in isclose_base.nc, unless the
    # tolerances are loose enough.
    fname_2 = os.path.join(input_dir, "isclose_diff.nc")
    assert not utils.isclose_all_vars(fname_1, fname_2, rtol=0.0, atol=0.0)
    assert not utils.isclose_all_vars(fname_1, fname_2, rtol=1.0e-8, atol=1.0e-8)
    assert utils.isclose_all_vars(fname_1, fname_2, rtol=1.0e-5, atol=1.0e-5)


@pytest.mark.parametrize("lout", [True, False])
def test_min_by_region(lout):
    """test min_by_region with specific values"""
    vals = np.arange(24.0).reshape((4, 6))
    region_mask = np.empty(vals.shape, dtype=np.int32)

    # single region_mask value
    region_mask[:] = 1
    region_cnt = region_mask.max()
    expected = np.array(0.0)
    if lout:
        out = np.empty(region_cnt)
        utils.min_by_region(region_cnt, region_mask, vals, out=out)
    else:
        out = utils.min_by_region(region_cnt, region_mask, vals)
    assert (out == expected).all()

    # region_mask equals 1st index + 1
    for ind in range(vals.shape[0]):
        region_mask[ind, :] = 1 + ind
    region_cnt = region_mask.max()
    expected = vals[:, 0]
    if lout:
        out = np.empty(region_cnt)
        utils.min_by_region(region_cnt, region_mask, vals, out=out)
    else:
        out = utils.min_by_region(region_cnt, region_mask, vals)
    assert (out == expected).all()

    # region_mask equals 1st index // 2 + 1
    for ind in range(vals.shape[0]):
        region_mask[ind, :] = 1 + ind // 2
    region_cnt = region_mask.max()
    expected = vals[::2, 0]
    if lout:
        out = np.empty(region_cnt)
        utils.min_by_region(region_cnt, region_mask, vals, out=out)
    else:
        out = utils.min_by_region(region_cnt, region_mask, vals)
    assert (out == expected).all()

    # region_mask equals 2nd index + 1
    for ind in range(vals.shape[1]):
        region_mask[:, ind] = 1 + ind
    region_cnt = region_mask.max()
    expected = vals[0, :]
    if lout:
        out = np.empty(region_cnt)
        utils.min_by_region(region_cnt, region_mask, vals, out=out)
    else:
        out = utils.min_by_region(region_cnt, region_mask, vals)
    assert (out == expected).all()

    # region_mask equals 2nd index // 2 + 1
    for ind in range(vals.shape[1]):
        region_mask[:, ind] = 1 + ind // 2
    region_cnt = region_mask.max()
    expected = vals[0, ::2]
    if lout:
        out = np.empty(region_cnt)
        utils.min_by_region(region_cnt, region_mask, vals, out=out)
    else:
        out = utils.min_by_region(region_cnt, region_mask, vals)
    assert (out == expected).all()


@pytest.mark.parametrize("lout", [True, False])
def test_comp_scalef(lout):
    """test comp_scalef_lob, comp_scalef_upb with specific values"""

    region_cnt = 7
    shape = (3, region_cnt)
    region_mask = np.zeros(shape, dtype=np.int32)
    base = np.ones(shape)
    increment = np.ones(shape)
    lob = 0.0
    expected = np.empty(region_cnt)

    region_ind = 0

    # base > lob, all increments positive, scalef = 1
    region_mask[:, region_ind] = region_ind + 1
    expected[region_ind] = 1.0

    region_ind += 1

    # base > lob, some increments negative, base + increment > lob, scalef = 1
    region_mask[:, region_ind] = region_ind + 1
    increment[0, region_ind] = -0.5
    expected[region_ind] = 1.0

    region_ind += 1

    # base > lob, some increments negative, base + increment == lob, scalef = 1
    region_mask[:, region_ind] = region_ind + 1
    increment[0, region_ind] = -0.5
    increment[1, region_ind] = -1.0
    expected[region_ind] = 1.0

    region_ind += 1

    # base > lob, some increments negative, base + increment < lob, scalef = .5
    region_mask[:, region_ind] = region_ind + 1
    increment[0, region_ind] = -0.5
    increment[1, region_ind] = -1.0
    increment[2, region_ind] = -2.0
    expected[region_ind] = 0.5

    region_ind += 1

    # base == lob, all increments positive, scalef = 1
    region_mask[:, region_ind] = region_ind + 1
    base[:, region_ind] = lob
    expected[region_ind] = 1.0

    region_ind += 1

    # base == lob, some increments zero, scalef = 1
    region_mask[:, region_ind] = region_ind + 1
    base[:, region_ind] = lob
    increment[0, region_ind] = 0.0
    expected[region_ind] = 1.0

    region_ind += 1

    # base == lob, some increments negative, scalef = 0
    region_mask[:, region_ind] = region_ind + 1
    base[:, region_ind] = lob
    increment[0, region_ind] = 0.0
    increment[1, region_ind] = -1.0
    expected[region_ind] = 0.0

    assert region_cnt == region_mask.max()

    if lout:
        out = np.empty(region_cnt)
        utils.comp_scalef_lob(region_cnt, region_mask, base, increment, lob, out=out)
    else:
        out = utils.comp_scalef_lob(region_cnt, region_mask, base, increment, lob)
    assert (out == expected).all()

    if lout:
        out = np.empty(region_cnt)
        utils.comp_scalef_upb(region_cnt, region_mask, -base, -increment, -lob, out=out)
    else:
        out = utils.comp_scalef_upb(region_cnt, region_mask, -base, -increment, -lob)
    assert (out == expected).all()
