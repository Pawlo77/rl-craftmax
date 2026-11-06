"""Tests for the 1M vs 1B angles: rank, advantage, scaling, precision, compute."""

import numpy as np
import pandas as pd
import pytest
from _compare_fixture import make_frames

from wut_murl_craftax.analysis import scale

N_BOOT = 100


@pytest.fixture(scope="module")
def frames() -> dict[str, pd.DataFrame]:
    return make_frames()


def _means(frames: dict[str, pd.DataFrame], budget: str) -> pd.Series:
    runs = frames["runs"]
    runs = runs[(runs["config_ref"] == "tuned") & (runs["budget"] == budget)]
    return runs.groupby("method")["final_score"].mean()


def test_rank_agreement_is_the_correlation_of_the_two_budget_means(
    frames: dict[str, pd.DataFrame],
) -> None:
    t = scale.rank_agreement(frames, n_boot=N_BOOT)
    both = pd.concat([_means(frames, "1m"), _means(frames, "1b")], axis=1).dropna()
    assert len(t) == 1
    want = both.corr(method="spearman").iloc[0, 1]
    assert t.loc[0, "spearman"] == pytest.approx(want)
    assert t.loc[0, "methods"] == len(both)
    assert t.loc[0, "spearman_lo"] <= t.loc[0, "spearman"] <= t.loc[0, "spearman_hi"]


def test_flipped_pairs_are_pairs_whose_order_differs(
    frames: dict[str, pd.DataFrame],
) -> None:
    t = scale.flipped_pairs(frames, n_boot=N_BOOT)
    m1, mb = _means(frames, "1m"), _means(frames, "1b")
    for row in t.itertuples():
        assert np.sign(m1[row.a] - m1[row.b]) != np.sign(mb[row.a] - mb[row.b])
        assert row.a < row.b
        both = (row.diff_1m_lo > 0 or row.diff_1m_hi < 0) and (
            row.diff_1b_lo > 0 or row.diff_1b_hi < 0
        )
        assert row.resolved == both


def test_advantage_is_the_relative_gain_over_the_base_control_at_each_budget(
    frames: dict[str, pd.DataFrame],
) -> None:
    t = scale.advantage_table(frames, n_boot=N_BOOT)
    assert set(t["method"]).isdisjoint({"ppo", "ppo-rnn", "pqn-rnn"})
    m1, mb = _means(frames, "1m"), _means(frames, "1b")
    row = t[t["method"] == "sil"].iloc[0]
    assert row["base"] == "ppo-rnn"
    assert row["gain_1m"] == pytest.approx(100 * (m1["sil"] / m1["ppo-rnn"] - 1))
    assert row["gain_1b"] == pytest.approx(100 * (mb["sil"] / mb["ppo-rnn"] - 1))
    assert row["change"] == pytest.approx(row["gain_1b"] - row["gain_1m"])
    assert row["change_lo"] <= row["change"] <= row["change_hi"]


def test_scaling_is_the_log_ratio_of_the_means(
    frames: dict[str, pd.DataFrame],
) -> None:
    t = scale.scaling_table(frames, n_boot=N_BOOT).set_index("method")
    m1, mb = _means(frames, "1m"), _means(frames, "1b")
    assert t.loc["ppo", "log_ratio"] == pytest.approx(np.log(mb["ppo"] / m1["ppo"]))
    assert (t["log_ratio_lo"] <= t["log_ratio_hi"]).all()


def test_precision_gives_the_smallest_difference_three_seeds_resolve(
    frames: dict[str, pd.DataFrame],
) -> None:
    t = scale.precision_table(frames).set_index("method")
    runs = frames["runs"]
    x = runs[(runs["budget"] == "1m") & (runs["method"] == "ppo")]["final_score"]
    sd = x.std(ddof=1)
    assert t.loc["ppo", "sd_1m"] == pytest.approx(sd)
    assert t.loc["ppo", "mde_3_seeds"] == pytest.approx(
        (1.959964 + 0.841621) * sd * np.sqrt(2 / 3), rel=1e-4
    )
    assert (t["halfwidth_1b"] > 0).all()


def test_compute_table_marks_the_pareto_front_of_score_against_hours(
    frames: dict[str, pd.DataFrame],
) -> None:
    t = scale.compute_table(frames)
    assert {"benchmark", "profile", "method", "hours", "score", "pareto"} <= set(
        t.columns
    )
    for _, g in t.groupby(["benchmark", "profile"]):
        for row in g.itertuples():
            beaten = ((g["hours"] <= row.hours) & (g["score"] >= row.score)) & (
                (g["hours"] < row.hours) | (g["score"] > row.score)
            )
            assert row.pareto == (not beaten.any())
