"""The seed interval of the hard-obstacle tables: shortcuts against the bootstrap."""

import math

import numpy as np
import pytest

from wut_murl_craftax.analysis.hard import _ci
from wut_murl_craftax.evaluation.stats import stratified_bootstrap_ci

N = 200


@pytest.mark.parametrize("values", [[0.25, 0.25, 0.25], [1.0] * 10, [0.0, 0.0]])
def test_equal_seeds_give_what_the_bootstrap_gives(values: list[float]) -> None:
    want = stratified_bootstrap_ci(
        {"seeds": values}, lambda d: float(d["seeds"].mean()), n=N
    )
    assert tuple(_ci(np.array(values), N)) == pytest.approx(tuple(want))


def test_a_single_seed_is_a_zero_width_interval() -> None:
    est = _ci(np.array([0.4]), N)
    assert est.lo == est.hi == est.estimate == pytest.approx(0.4)


def test_no_seed_is_nan() -> None:
    assert all(math.isnan(v) for v in _ci(np.array([]), N))
