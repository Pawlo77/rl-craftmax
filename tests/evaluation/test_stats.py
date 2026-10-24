"""Tests for the statistics protocol: intervals, Holm-corrected tests, spread, bands."""

import logging

import numpy as np
import pytest
from scipy import stats as sps

from wut_murl_craftax.evaluation import stats
from wut_murl_craftax.evaluation.stats import (
    Interval,
    curve_band,
    holm,
    iqm,
    mann_whitney_holm,
    mean_se,
    probability_of_improvement,
    seed_spread,
    step_grid,
    stratified_bootstrap_ci,
    t_ci,
)

SCORES = [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 7.0, 8.0, 9.0]


def pooled_iqm(resampled: dict[str, np.ndarray]) -> float:
    """The aggregate statistic of the protocol: IQM of the pooled strata."""
    return iqm(np.concatenate(list(resampled.values())))


# --- point estimates --------------------------------------------------------------


def test_iqm_of_eight_scores() -> None:
    assert iqm([1, 2, 3, 4, 5, 6, 7, 8]) == 4.5


def test_probability_of_improvement_extremes() -> None:
    assert probability_of_improvement([1, 2, 3], [1, 2, 3]) == 0.5
    assert probability_of_improvement([4, 5, 6], [1, 2, 3]) == 1.0
    assert probability_of_improvement([1, 2, 3], [4, 5, 6]) == 0.0


def test_mean_se() -> None:
    mean, se = mean_se([10, 11, 12])
    assert mean == 11
    assert se == pytest.approx(1 / np.sqrt(3))


def test_mean_se_of_one_score_has_no_error() -> None:
    mean, se = mean_se([4.0])
    assert mean == 4.0
    assert np.isnan(se)


# --- stratified bootstrap ---------------------------------------------------------


def test_bootstrap_ci_contains_the_point_estimate() -> None:
    ci = stratified_bootstrap_ci({"a": SCORES, "b": SCORES[::-1]}, pooled_iqm, n=2000)
    assert isinstance(ci, Interval)
    assert ci.estimate == pytest.approx(iqm(SCORES + SCORES))
    assert ci.lo <= ci.estimate <= ci.hi
    assert ci.lo < ci.hi


def test_bootstrap_ci_is_reproducible_per_seed() -> None:
    data = {"a": SCORES}
    first = stratified_bootstrap_ci(data, pooled_iqm, n=500, seed=3)
    again = stratified_bootstrap_ci(data, pooled_iqm, n=500, seed=3)
    other = stratified_bootstrap_ci(data, pooled_iqm, n=500, seed=4)
    assert first == again
    assert first != other


def test_bootstrap_never_mixes_strata() -> None:
    def pure(resampled: dict[str, np.ndarray]) -> float:
        ok = set(resampled["low"]) <= {0.0, 1.0} and set(resampled["high"]) <= {9.0}
        return float(ok and len(resampled["low"]) == 2 and len(resampled["high"]) == 3)

    ci = stratified_bootstrap_ci(
        {"low": [0.0, 1.0], "high": [9.0, 9.0, 9.0]}, pure, n=300
    )
    assert (ci.lo, ci.estimate, ci.hi) == (1.0, 1.0, 1.0)


def test_bootstrap_default_is_ten_thousand_percentile_resamples() -> None:
    seen: list[int] = []

    def count(resampled: dict[str, np.ndarray]) -> float:
        seen.append(0)
        return float(resampled["a"].mean())

    stratified_bootstrap_ci({"a": [1.0, 2.0, 3.0]}, count)
    assert len(seen) == 10_000 + 1  # the resamples and the point estimate


def test_bootstrap_ci_of_a_difference_over_two_strata() -> None:
    x, y = [5.0, 6.0, 7.0, 8.0], [1.0, 2.0, 3.0, 4.0]
    ci = stratified_bootstrap_ci(
        {"x": x, "y": y}, lambda d: probability_of_improvement(d["x"], d["y"]), n=500
    )
    assert ci.estimate == 1.0
    assert ci.lo == ci.hi == 1.0


def test_bootstrap_rejects_an_empty_stratum() -> None:
    with pytest.raises(ValueError, match="empty"):
        stratified_bootstrap_ci({"a": []}, pooled_iqm, n=10)


# --- Holm-Bonferroni and Mann-Whitney ----------------------------------------------


def test_holm_adjusted_p_values() -> None:
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_holm_caps_at_one_and_keeps_the_order() -> None:
    adjusted = holm([0.9, 0.5, 0.001])
    assert adjusted == pytest.approx([1.0, 1.0, 0.003])


def test_mann_whitney_matches_scipy_and_holm_adjusts() -> None:
    x = [8.1, 9.2, 7.7, 10.4, 9.9, 8.8, 9.5, 10.1, 8.4, 9.0]
    y = [6.2, 7.1, 5.9, 8.0, 7.4, 6.8, 7.7, 6.5, 7.0, 6.1]
    z = [8.0, 9.0, 7.5, 10.0, 9.5, 8.5, 9.2, 10.2, 8.3, 9.1]
    table = mann_whitney_holm({"x>y": (x, y), "x~z": (x, z)})
    ref = sps.mannwhitneyu(x, y, alternative="two-sided")
    assert table.loc["x>y", "u"] == pytest.approx(ref.statistic)
    assert table.loc["x>y", "p"] == pytest.approx(ref.pvalue)
    ref_z = sps.mannwhitneyu(x, z, alternative="two-sided")
    assert table.loc["x~z", "p"] == pytest.approx(ref_z.pvalue)
    p_min, p_max = sorted(table["p"])
    assert sorted(table["p_holm"]) == pytest.approx([2 * p_min, max(2 * p_min, p_max)])
    assert table.loc["x>y", "reject"]
    assert not table.loc["x~z", "reject"]


def test_mann_whitney_holm_family_is_the_whole_table() -> None:
    rng = np.random.default_rng(0)
    pairs = {f"p{i}": (rng.normal(i, 1, 10), rng.normal(0, 1, 10)) for i in range(4)}
    table = mann_whitney_holm(pairs)
    assert list(table.index) == list(pairs)
    assert table["p_holm"].tolist() == pytest.approx(holm(table["p"].tolist()))
    assert (table["p_holm"] >= table["p"]).all()


# --- t interval, spread, bands -----------------------------------------------------


def test_mann_whitney_warns_when_no_pair_can_reach_alpha(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        table = mann_whitney_holm({"1B": ([4.0, 5.0, 6.0], [1.0, 2.0, 3.0])})
    assert "cannot reach" in caplog.text
    assert "1B" in caplog.text
    assert not table.loc["1B", "reject"]
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        mann_whitney_holm({"1M": (SCORES, SCORES[::-1])})
    assert not caplog.text


def test_t_ci_of_three_scores() -> None:
    ci = t_ci([10, 11, 12])
    assert ci.estimate == 11
    assert ci.hi - ci.estimate == pytest.approx(2.484, abs=1e-3)
    assert ci.estimate - ci.lo == pytest.approx(2.484, abs=1e-3)


def test_t_ci_needs_two_scores() -> None:
    with pytest.raises(ValueError, match="two"):
        t_ci([1.0])


def test_seed_spread() -> None:
    spread = seed_spread([2.0, 4.0, 6.0, 8.0])
    assert spread["sd"] == pytest.approx(np.std([2, 4, 6, 8], ddof=1))
    assert spread["iqr"] == pytest.approx(3.0)
    assert spread["range"] == 6.0
    assert spread["cv"] == pytest.approx(spread["sd"] / 5.0)


def test_seed_spread_needs_a_score() -> None:
    with pytest.raises(ValueError, match="at least one"):
        seed_spread([])


def test_seed_spread_of_a_zero_mean_has_no_cv() -> None:
    assert np.isnan(seed_spread([-1.0, 1.0])["cv"])


def test_step_grid_is_log_spaced_over_the_first_five_percent() -> None:
    grid = step_grid(1_000_000)
    assert len(grid) == 200
    assert grid[-1] == 1_000_000
    assert (np.diff(grid) > 0).all()
    head = grid[grid <= 50_000]
    assert 0 < len(head) < 200
    ratios = grid[1 : len(head)] / grid[: len(head) - 1]
    assert ratios == pytest.approx(np.full(len(head) - 1, ratios[0]))
    tail = np.diff(grid[len(head) :])
    assert tail == pytest.approx(np.full(len(tail), tail[0]))


def make_curves(offsets: list[float]) -> list[tuple[np.ndarray, np.ndarray]]:
    steps = np.linspace(10_000, 1_000_000, 100)
    return [(steps, steps / 1e6 * 10 + o) for o in offsets]


def test_curve_band_mean_and_iqm() -> None:
    offsets = [-1.0, -0.5, 0.0, 0.0, 0.5, 1.0, 0.2, -0.2]
    curves = make_curves(offsets)
    grid = np.array([100_000.0, 500_000.0, 1_000_000.0])
    mean_band = curve_band(curves, grid, n=500)
    assert list(mean_band.columns) == ["env_steps", "estimate", "lo", "hi"]
    assert mean_band["estimate"].tolist() == pytest.approx([
        1.0 + np.mean(offsets),
        5.0 + np.mean(offsets),
        10.0 + np.mean(offsets),
    ])
    assert (mean_band["lo"] <= mean_band["estimate"]).all()
    assert (mean_band["estimate"] <= mean_band["hi"]).all()
    iqm_band = curve_band(curves, grid, statistic="iqm", n=500)
    assert iqm_band["estimate"].tolist() == pytest.approx([
        1.0 + iqm(offsets),
        5.0 + iqm(offsets),
        10.0 + iqm(offsets),
    ])


def test_curve_band_is_reproducible_and_flat_without_spread() -> None:
    curves = make_curves([0.0, 0.0, 0.0])
    grid = np.array([200_000.0, 800_000.0])
    band = curve_band(curves, grid, n=200)
    assert (band["lo"] == band["hi"]).all()
    a = curve_band(make_curves([0.0, 1.0, 2.0]), grid, n=200, seed=1)
    b = curve_band(make_curves([0.0, 1.0, 2.0]), grid, n=200, seed=1)
    assert a.equals(b)


def test_curve_band_is_nan_where_a_seed_has_not_started() -> None:
    curves = make_curves([0.0, 1.0])
    band = curve_band(curves, np.array([1_000.0, 500_000.0]), n=100)
    assert band.loc[0, ["estimate", "lo", "hi"]].isna().all()
    assert band.loc[1, "estimate"] == pytest.approx(5.5)


def test_curve_band_holds_the_last_value_only_within_one_logging_gap() -> None:
    curves = make_curves([0.0, 1.0])
    band = curve_band(curves, np.array([1_005_000.0, 1_100_000.0]), n=100)
    assert band.loc[0, "estimate"] == pytest.approx(10.5)
    assert band.loc[1, ["estimate", "lo", "hi"]].isna().all()


def test_curve_band_skips_non_finite_points_of_a_seed() -> None:
    steps, values = make_curves([0.0])[0]
    holed = values.copy()
    holed[[40, 41, 42]] = np.nan  # updates without a finished episode
    band = curve_band([(steps, values), (steps, holed)], np.array([steps[41]]), n=100)
    assert band.loc[0, "estimate"] == pytest.approx(values[41])
    assert np.isfinite(band.loc[0, ["lo", "hi"]]).all()


def test_curve_band_chunks_do_not_change_the_draws(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    curves = make_curves([0.0, 1.0, 2.5, -1.0, 0.3])
    grid = np.array([100_000.0, 600_000.0])
    chunked = curve_band(curves, grid, n=2500, seed=2)  # 2.5 chunks of 1000
    monkeypatch.setattr(stats, "CHUNK", 10_000)
    whole = curve_band(curves, grid, n=2500, seed=2)
    assert chunked.equals(whole)
    assert (chunked["lo"] < chunked["hi"]).all()


def test_curve_band_rejects_unknown_statistic() -> None:
    with pytest.raises(ValueError, match="statistic"):
        curve_band(make_curves([0.0, 1.0]), np.array([5e5]), statistic="max")
