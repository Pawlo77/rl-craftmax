"""Tests for the abort rules: non-finite losses and the flag rules."""

import numpy as np
import pytest

from wut_murl_craftax.orchestration.abort import first_nonfinite, flag_hits
from wut_murl_craftax.training.records import LOSS_COLUMNS


def losses(rows: int = 4) -> np.ndarray:
    return np.ones((rows, len(LOSS_COLUMNS)))


def test_the_earliest_non_finite_loss_names_its_update_and_metric() -> None:
    table = losses()
    table[2, LOSS_COLUMNS.index("entropy")] = np.inf
    table[1, LOSS_COLUMNS.index("grad_norm")] = np.nan
    assert first_nonfinite(table, LOSS_COLUMNS, {}) == (1, "grad_norm")
    assert first_nonfinite(losses(), LOSS_COLUMNS, {}) is None


def test_only_the_reported_loss_columns_are_checked() -> None:
    table = losses()
    table[:, LOSS_COLUMNS.index("policy_loss")] = np.nan
    assert first_nonfinite(table, ("value_loss", "grad_norm"), {}) is None
    assert first_nonfinite(table, LOSS_COLUMNS, {}) == (0, "policy_loss")


def test_method_metrics_count_when_they_are_losses() -> None:
    extra = {
        "ppo/value_loss": np.array([1.0, np.nan]),
        "ppo/lr": np.array([np.nan] * 2),
    }
    assert first_nonfinite(losses(2), LOSS_COLUMNS, extra) == (1, "ppo/value_loss")
    assert first_nonfinite(losses(2), LOSS_COLUMNS, {"ppo/lr": extra["ppo/lr"]}) is None


KL = {"name": "kl_high", "metric": "ppo/approx_kl", "above": 0.1, "updates": 3}


def test_a_flag_needs_its_updates_in_a_row() -> None:
    hot, cold = 0.2, 0.01
    assert flag_hits([KL], {"ppo/approx_kl": [cold, hot, hot, hot, cold]}, 5) == {
        "kl_high": 4
    }
    assert flag_hits([KL], {"ppo/approx_kl": [hot, hot, cold, hot, hot]}, 5) == {}
    assert flag_hits([KL], {"ppo/approx_kl": [hot, hot]}, 5) == {}


def test_a_flag_reports_the_first_time_it_held() -> None:
    series = {"ppo/approx_kl": [0.2] * 6}
    assert flag_hits([KL], series, 6) == {"kl_high": 3}


def test_before_and_after_are_shares_of_the_budget() -> None:
    low = {"name": "entropy_collapse", "metric": "ppo/entropy", "below": 0.2}
    series = {"ppo/entropy": [3.0] * 5 + [0.1] + [3.0] * 4}  # update 6 of 10
    assert flag_hits([{**low, "before": 0.5}], series, 10) == {}
    assert flag_hits([{**low, "before": 0.6}], series, 10) == {"entropy_collapse": 6}
    assert flag_hits([{**low, "after": 0.7}], series, 10) == {}
    assert flag_hits([{**low, "after": 0.6}], series, 10) == {"entropy_collapse": 6}


def test_nan_and_missing_metrics_never_flag() -> None:
    neg = {"name": "ev", "metric": "ppo/explained_variance", "below": 0.0}
    assert flag_hits([neg], {"ppo/explained_variance": [np.nan] * 3}, 3) == {}
    assert flag_hits([neg], {}, 3) == {}


def test_several_rules_are_independent() -> None:
    ev = {"name": "ev", "metric": "ppo/explained_variance", "below": 0.0}
    series = {"ppo/approx_kl": [0.2] * 3, "ppo/explained_variance": [0.5, -1.0, 0.5]}
    assert flag_hits([KL, ev], series, 3) == {"kl_high": 3, "ev": 2}


@pytest.mark.parametrize("length", [1, 3])
def test_a_series_shorter_than_the_run_still_flags_by_its_own_updates(
    length: int,
) -> None:
    rule = {"name": "x", "metric": "m", "above": 0.0}
    assert flag_hits([rule], {"m": [1.0] * length}, 100) == {"x": 1}
