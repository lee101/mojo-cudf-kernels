"""Parity tests for the exponentially weighted moving average.

The reference in `tests/reference.py` rebuilds the adjusted branch from an
explicit weight vector via `np.cumprod`, so the two implementations share no
arithmetic beyond the final dot product.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import mojo_cudf_kernels as mck
from reference import ewm_reference

RTOL = 1e-11


def _beta(com):
    return com / (com + 1.0)


@pytest.mark.parametrize("com", [0.1, 0.5, 1.0, 3.0, 10.0])
def test_adjusted_ewm_matches_weight_vector_reference(com):
    rng = np.random.default_rng(21)
    data = rng.standard_normal(512)
    expect = ewm_reference(data, None, _beta(com), adjust=True)
    got = mck.ewm(data, com=com, adjust=True)
    np.testing.assert_allclose(got.data, expect, rtol=RTOL, atol=1e-13)


@pytest.mark.parametrize("com", [0.1, 0.5, 1.0, 3.0, 10.0])
def test_unadjusted_ewm_matches_reference(com):
    rng = np.random.default_rng(22)
    data = rng.standard_normal(512)
    expect = ewm_reference(data, None, _beta(com), adjust=False)
    got = mck.ewm(data, com=com, adjust=False)
    np.testing.assert_allclose(got.data, expect, rtol=RTOL, atol=1e-13)


def test_both_branches_agree_at_the_first_element():
    # A branch that seeds with the identity instead of x[0] would disagree
    # here; this is the single sharpest check on the seeding rule.
    data = [12.0, 5.0, 7.0, 1.0]
    adjusted = mck.ewm(data, alpha=0.3, adjust=True)
    plain = mck.ewm(data, alpha=0.3, adjust=False)
    np.testing.assert_allclose(adjusted.data[0], data[0], rtol=0, atol=0)
    np.testing.assert_allclose(plain.data[0], data[0], rtol=0, atol=0)
    assert adjusted.data[1] != plain.data[1]


def test_unadjusted_ewm_satisfies_its_recurrence():
    # cudf derives beta = com / (com + 1) and runs y_i = beta * y_(i-1) +
    # (1 - beta) * x_i, so beta is pandas' (1 - alpha), not its alpha.
    alpha, data = 0.25, [2.0, 4.0, 1.0, 9.0, 3.0]
    got = mck.ewm(data, alpha=alpha, adjust=False).data
    for i in range(1, len(data)):
        expected = (1.0 - alpha) * got[i - 1] + alpha * data[i]
        np.testing.assert_allclose(got[i], expected, rtol=1e-13, atol=0)
    # Getting the two weights the wrong way round is the obvious bug here.
    wrong = alpha * got[0] + (1.0 - alpha) * data[1]
    assert abs(got[1] - wrong) > 1e-3


def test_ewm_with_nulls_carries_and_downweights():
    data = [5.0, 0.0, 3.0, 0.0, 8.5]
    valid = np.array([1, 0, 1, 0, 1], dtype=bool)
    for adjust in (True, False):
        com = 0.5
        expect = ewm_reference(data, valid, _beta(com), adjust)
        got = mck.ewm(mck.Column(data, valid), com=com, adjust=adjust)
        np.testing.assert_allclose(
            got.data, expect, rtol=1e-10, atol=1e-12, err_msg=f"adjust={adjust}"
        )
    # cudf builds the ewm column with an empty null mask, so nothing is null.
    assert mck.ewm(mck.Column(data, valid), com=0.5).nulls == 0


def test_ewm_leading_nulls_are_nan():
    data = [0.0, 4.0, 6.0]
    valid = np.array([0, 1, 1], dtype=bool)
    got = mck.ewm(mck.Column(data, valid), com=1.0, adjust=True)
    assert math.isnan(got.data[0])
    np.testing.assert_allclose(got.data[1], 4.0, rtol=0, atol=0)


def test_ewm_all_null_column():
    column = mck.Column([1.0, 2.0], np.array([0, 0], dtype=bool))
    got = mck.ewm(column, com=1.0)
    assert np.all(np.isfinite(got.data)) or np.all(np.isnan(got.data))
    assert got.nulls == 0


def test_center_of_mass_matches_cudf():
    # cudf's get_center_of_mass, including the domain checks it raises on.
    assert mck.center_of_mass(com=0.5) == 0.5
    assert mck.center_of_mass(span=3.0) == 1.0
    assert mck.center_of_mass(span=1.5) == 0.25
    assert mck.center_of_mass(halflife=0.5) == 1.0 / (
        1 - math.exp(math.log(0.5) / 0.5)
    ) - 1.0
    assert mck.center_of_mass(alpha=0.5) == 1.0
    assert mck.center_of_mass(alpha=0.25) == 3.0
    for kwargs in ({"span": 0.5}, {"halflife": 0.0}, {"alpha": 0.0}, {"alpha": 1.5}, {"com": -1.0}):
        with pytest.raises(ValueError):
            mck.center_of_mass(**kwargs)
    with pytest.raises(ValueError):
        mck.center_of_mass(com=1.0, span=2.0)
    with pytest.raises(ValueError):
        mck.center_of_mass()


def test_ewm_parameter_aliases_give_the_same_curve():
    data = [3.0, 1.0, 4.0, 1.0, 5.0]
    a = mck.ewm(data, alpha=0.4).data
    b = mck.ewm(data, com=1.5).data          # (1 - 0.4) / 0.4
    c = mck.ewm(data, span=4.0).data         # (4 - 1) / 2 = 1.5
    np.testing.assert_allclose(a, b, rtol=1e-14, atol=0)
    np.testing.assert_allclose(a, c, rtol=1e-14, atol=0)


def test_ewm_needs_one_parameter():
    with pytest.raises(ValueError):
        mck.ewm([1.0, 2.0])
