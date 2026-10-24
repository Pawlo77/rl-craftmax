"""Tests for learning stability (drawdowns, drops, collapse) and sensitivity."""

import json

import numpy as np
import pandas as pd
import pytest

from wut_murl_craftax.evaluation.stability import (
    collapsed,
    config_scores,
    dispersion,
    drawdowns,
    drops,
    recovery_times,
    robust_share,
    sensitivity,
    smoothed,
)

N = 1000
STEP = 1000.0
STEPS = np.arange(1, N + 1) * STEP


def plateau(level: float = 10.0) -> np.ndarray:
    return np.full(N, level)


def dip(depth: float, start: int, length: int | None) -> np.ndarray:
    """A plateau of 10 that falls by ``depth`` at ``start`` for ``length`` updates."""
    c = plateau()
    c[start : None if length is None else start + length] *= 1 - depth
    return c


# --- smoothing and drawdown ---------------------------------------------------------


def test_smoothed_is_a_rolling_median_of_one_percent() -> None:
    raw = plateau()
    raw[500] = 1000.0  # a one-update spike is removed by a window of 10
    assert (smoothed(raw) == 10.0).all()
    assert smoothed(raw, window_frac=0.0001)[500] == 1000.0


def test_smoothed_keeps_the_length_and_a_long_step() -> None:
    raw = dip(0.5, 400, None)
    c = smoothed(raw)
    assert len(c) == N
    assert c[:395].tolist() == [10.0] * 395
    assert c[405:].tolist() == [5.0] * 595


def test_drawdown_is_the_fall_from_the_running_max() -> None:
    c = np.array([2.0, 4.0, 3.0, 4.0, 5.0, 2.5])
    assert drawdowns(c) == pytest.approx([0, 0, 0.25, 0, 0, 0.5])


def test_drawdown_is_undefined_below_one_percent_of_max() -> None:
    d = drawdowns(np.array([0.2, 0.5, 0.4, 2.0, 1.0]))
    assert np.isnan(d[:3]).all()
    assert d[3:] == pytest.approx([0.0, 0.5])


# --- drops and collapse -------------------------------------------------------------


def test_sixty_percent_fall_without_recovery_is_one_collapse() -> None:
    c = smoothed(dip(0.6, 500, None))
    assert collapsed(c)
    assert len(drops(c)) == 1


def test_short_deep_dip_is_a_drop_and_no_collapse() -> None:
    c = smoothed(dip(0.25, 500, 30))  # 3% of the budget
    assert len(drops(c)) == 1
    assert not collapsed(c)


def test_a_dip_shorter_than_two_percent_is_no_drop() -> None:
    assert drops(smoothed(dip(0.3, 500, 15))) == []


def test_a_dip_below_twenty_percent_is_no_drop() -> None:
    assert drops(smoothed(dip(0.15, 500, 100))) == []


def test_noise_without_dips_gives_no_drop() -> None:
    rng = np.random.default_rng(0)
    raw = np.linspace(2.0, 10.0, N) * (1 + 0.05 * rng.standard_normal(N))
    assert drops(smoothed(raw)) == []
    assert not collapsed(smoothed(raw))


def test_deep_dip_that_recovers_is_no_collapse() -> None:
    assert not collapsed(smoothed(dip(0.6, 500, 100)))


def test_deep_dip_ending_within_ten_percent_of_the_max_is_no_collapse() -> None:
    c = plateau()
    c[500:600] = 4.0
    c[600:] = 9.5
    assert not collapsed(c)


def test_two_separate_dips_are_two_drops() -> None:
    c = plateau()
    c[200:240] *= 0.7
    c[600:650] *= 0.6
    assert drops(c) == [(200, 240), (600, 650)]


def test_drop_bounds_are_a_half_open_index_interval() -> None:
    assert drops(dip(0.3, 500, 30)) == [(500, 530)]


def test_drop_at_the_end_of_the_run_is_kept() -> None:
    assert drops(dip(0.3, 970, None)) == [(970, N)]


# --- NaN scores (updates without a finished episode) ---------------------------------


def with_nans(c: np.ndarray, start: int, stop: int) -> np.ndarray:
    out = c.copy()
    out[start:stop] = np.nan
    return out


def test_a_nan_stretch_does_not_hide_a_collapse() -> None:
    raw = with_nans(dip(0.6, 300, None), 100, 130)
    c = smoothed(raw)
    assert np.isnan(c).any()
    assert collapsed(c)
    assert drops(c) == [(300, N)]
    assert np.isfinite(drawdowns(c)[200:]).all()


def test_a_nan_stretch_inside_a_drop_does_not_split_it() -> None:
    c = with_nans(dip(0.4, 300, 200), 380, 420)
    assert drops(c) == [(300, 500)]
    assert recovery_times(c, STEPS) == [200 * STEP]


def test_leading_nans_leave_the_drawdown_undefined() -> None:
    c = with_nans(plateau(), 0, 50)
    d = drawdowns(c)
    assert np.isnan(d[:50]).all()
    assert (d[50:] == 0).all()


def test_a_trailing_nan_stretch_uses_the_last_known_value_for_collapse() -> None:
    assert collapsed(with_nans(dip(0.6, 300, None), 950, N))
    assert not collapsed(with_nans(dip(0.6, 300, 100), 950, N))


def test_dispersion_ignores_nan_differences() -> None:
    c = np.array([0.0, 1.0, np.nan, 3.0, 4.0, 5.0])
    assert dispersion(c) == pytest.approx(0.0)
    assert np.isnan(dispersion(np.array([np.nan, np.nan, 1.0])))


# --- recovery ------------------------------------------------------------------------


def test_recovery_time_of_a_known_dip() -> None:
    c = dip(0.3, 500, 40)
    assert recovery_times(c, STEPS) == [40 * STEP]


def test_recovery_needs_to_get_within_five_percent() -> None:
    c = plateau()
    c[500:540] = 6.0
    c[540:600] = 9.2  # d = 0.08: still not recovered
    c[600:] = 9.6
    assert recovery_times(c, STEPS) == [100 * STEP]


def test_a_drop_that_never_recovers_has_infinite_recovery_time() -> None:
    assert recovery_times(dip(0.6, 500, None), STEPS) == [np.inf]


def test_recovery_needs_one_step_per_update() -> None:
    with pytest.raises(ValueError, match="steps"):
        recovery_times(plateau(), STEPS[:-1])


def test_no_drop_gives_no_recovery_time() -> None:
    assert recovery_times(plateau(), STEPS) == []


# --- the drawdown floor (R92) -------------------------------------------------------


def test_the_floor_defaults_to_one_point() -> None:
    c = np.array([0.2, 0.5, 0.4, 2.0, 1.0])
    np.testing.assert_array_equal(drawdowns(c), drawdowns(c, floor=1.0))


def test_a_lower_floor_defines_drawdowns_of_a_low_curve() -> None:
    d = drawdowns(np.array([0.2, 0.5, 0.4, 2.0, 1.0]), floor=0.3)
    assert np.isnan(d[0])
    assert d[1:] == pytest.approx([0.0, 0.2, 0.0, 0.5])


def test_drops_and_collapse_see_a_low_curve_only_with_a_lower_floor() -> None:
    low = dip(0.6, 500, None) / 20  # a plateau of 0.5% falling to 0.2%
    c = smoothed(low)
    assert drops(c) == []
    assert not collapsed(c)
    assert drops(c, floor=0.1) == [(500, N)]
    assert collapsed(c, floor=0.1)
    assert recovery_times(low, STEPS) == []
    assert recovery_times(low, STEPS, floor=0.1) == [np.inf]


# --- dispersion ----------------------------------------------------------------------


def test_dispersion_is_the_iqr_of_first_differences() -> None:
    c = np.cumsum([0, 1, 1, 1, 1, 5, -3, 1, 1])
    assert dispersion(c) == pytest.approx(
        np.subtract(*np.percentile(np.diff(c), [75, 25]))
    )


def test_dispersion_of_a_straight_line_is_zero() -> None:
    assert dispersion(np.linspace(0, 10, 50)) == pytest.approx(0.0)


# --- sensitivity ---------------------------------------------------------------------

LR = {1e-4: 0.4, 3e-4: 1.0, 1e-3: 0.6}
"""Final score per learning rate; the other knob does nothing."""
NOISE = [-0.01, 0.0, 0.01]
"""Per-seed offsets (sd 0.01) around a config's mean."""


def tuning_table(
    effect: dict[float, float], other: dict[str, float], jitter: float = 0.0
) -> pd.DataFrame:
    """Runs of a grid; ``jitter`` adds config-level noise (sd) on top of seed noise."""
    rng = np.random.default_rng(0)
    rows = []
    for lr, base in effect.items():
        for clip, shift in other.items():
            level = base + shift + (rng.normal(0, jitter) if jitter else 0.0)
            for seed, eps in enumerate(NOISE):
                cfg = {"method": "ppo", "lr": lr, "clip": clip, "seed_free": 1}
                rows.append({
                    "run_id": f"{lr}-{clip}-{seed}",
                    "config_hash": f"{lr}-{clip}",
                    "seed": seed,
                    "final_score": level + eps,
                    "config_json": json.dumps(cfg),
                })
    return pd.DataFrame(rows)


@pytest.fixture
def grid() -> pd.DataFrame:
    return tuning_table(LR, {"0.1": 0.0, "0.2": 0.0})


def test_only_the_learning_rate_is_sensitive(grid: pd.DataFrame) -> None:
    table = sensitivity(grid).set_index("hyperparameter")
    assert list(table.index) == ["lr", "clip"]  # most sensitive first
    assert table.loc["lr", "sensitivity"] == pytest.approx(0.6, abs=1e-6)
    assert table.loc["clip", "sensitivity"] == pytest.approx(0.0, abs=1e-9)
    assert table.loc["lr", "noise_floor"] == pytest.approx(0.01 / 1.0, abs=1e-6)
    assert not table.loc["lr", "insensitive"]
    assert table.loc["clip", "insensitive"]
    assert table.loc["lr", "n_values"] == 3
    assert table.loc["lr", "best_value"] == 3e-4
    assert table.loc["lr", "worst_value"] == 1e-4


def test_config_noise_alone_stays_below_the_noise_floor() -> None:
    table = sensitivity(tuning_table(LR, {"0.1": 0.0, "0.2": 0.0}, jitter=0.01))
    by_h = table.set_index("hyperparameter")
    assert 0 < by_h.loc["clip", "sensitivity"] < by_h.loc["clip", "noise_floor"]
    assert by_h.loc["clip", "insensitive"]
    assert by_h.loc["lr", "sensitivity"] == pytest.approx(0.6, abs=0.05)
    assert not by_h.loc["lr", "insensitive"]


def test_constants_and_seed_like_keys_are_not_hyperparameters() -> None:
    table = tuning_table(LR, {"0.1": 0.0, "0.2": 0.0})
    table["config_json"] = [
        json.dumps({**json.loads(c), "seed": s})
        for c, s in zip(table.config_json, table.seed, strict=True)
    ]
    assert set(sensitivity(table)["hyperparameter"]) == {"lr", "clip"}


def test_sensitivity_uses_scores_relative_to_the_best_config() -> None:
    table = tuning_table(LR, {"0.1": 0.0, "0.2": -0.5})
    by_h = sensitivity(table).set_index("hyperparameter")
    scores = config_scores(table).set_index("config_hash")
    assert scores["r_c"].max() == 1.0
    # marginal mean of r_c per clip value: 0.1 -> mean(0.4, 1.0, 0.6), 0.2 -> shifted
    best = 1.0
    r = {
        v: np.mean([(b + s) / best for b in LR.values()])
        for v, s in {"0.1": 0.0, "0.2": -0.5}.items()
    }
    assert by_h.loc["clip", "sensitivity"] == pytest.approx(
        r["0.1"] - r["0.2"], abs=1e-6
    )


def test_config_scores_and_robust_share(grid: pd.DataFrame) -> None:
    scores = config_scores(grid)
    assert len(scores) == 6
    assert scores["n_seeds"].eq(3).all()
    # lr 3e-4 (two configs) scores 1.0; the rest are far below 0.9
    assert robust_share(scores) == pytest.approx(2 / 6)


def test_sensitivity_needs_a_positive_best_config() -> None:
    table = tuning_table({1e-4: -1.0, 3e-4: -2.0}, {"0.1": 0.0})
    with pytest.raises(ValueError, match="positive"):
        sensitivity(table)


def test_sensitivity_of_identical_configs_is_empty() -> None:
    table = tuning_table({3e-4: 1.0}, {"0.1": 0.0})
    assert sensitivity(table).empty
    assert "sensitivity" in sensitivity(table).columns
