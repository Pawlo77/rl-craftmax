"""Tests for the hard-scenario summaries: floors, achievements, deaths, stalls, plots.

`frame` builds eval-episode frames with chosen values (one row per episode, the columns
the summaries read plus a few replay-only ones that must stay out of the results); the
export fixture checks the summaries on real parquet types and over several units.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from wut_murl_craftax.analysis.hard import (
    ACHIEVEMENTS,
    NO_DEATHS,
    RARE_RATE,
    achievement_status,
    death_causes,
    floor_reach,
    plot_achievement_status,
    plot_death_causes,
    plot_floor_reach,
    plot_stall_lengths,
    stall_ccdf,
    stall_starts,
    stall_stats,
    stall_table,
)
from wut_murl_craftax.analysis.load import load_phases, units
from wut_murl_craftax.evaluation.stats import Interval, stratified_bootstrap_ci

N = 200
"""Bootstrap resamples in the tests (the default is 10,000)."""
NAMES = (
    "collect_wood", "place_table", "defeat_zombie", "collect_diamond", "make_arrow",
    "enter_dungeon", "enter_sewers",
)  # fmt: skip
"""Achievement columns of the built frames: basic ones plus floor 1 and floor 3."""
PNG = b"\x89PNG\r\n\x1a\n"


def stall(
    start: int, end: int, floor: int, ending: str, by: str = ""
) -> dict[str, Any]:
    """A stall struct as the export stores it, without a replay trace."""
    return {
        "start": start,
        "end": end,
        "length": end - start,
        "floor": floor,
        "ending": ending,
        "ended_by": by,
        "actions": np.zeros(3, dtype=np.int64),
        "intrinsics": None,
        "inventory": None,
    }


def run_rows(
    seed: int,
    episodes: list[dict[str, Any]],
    *,
    update: int = 244,
    mode: str = "stochastic",
    method: str = "m",
    ref: str = "tuned",
    budget: str = "1m",
) -> list[dict[str, Any]]:
    """Episode rows of one run at one checkpoint; each dict overrides the defaults.

    The special key ``unlocked`` ({name: step}) sets ``ach/`` and ``unlock_step/``.
    """
    rows = []
    for i, fields in enumerate(episodes):
        fields = dict(fields)
        unlocked = fields.pop("unlocked", {})
        stalls = fields.pop("stalls", [])
        rows.append({
            "run_id": f"{method}-{ref}-{seed}",
            "method": method,
            "config_ref": ref,
            "budget": budget,
            "seed": seed,
            "update": update,
            "env_steps": update * 4096,
            "mode": mode,
            "eval_seed": 10000 + i,
            "length": 1000,
            "end_type": "timeout",
            "cause_of_death": "",
            "max_floor": 0,
            "stalls": stalls,
            "n_stalls": len(stalls),
            **{f"ach/{a}": a in unlocked for a in NAMES},
            **{f"unlock_step/{a}": np.int32(unlocked.get(a, -1)) for a in NAMES},
            # replay-only fields: NaN without an offline replay (evaluation Task 6)
            "coverage/0": np.nan,
            "gathered/wood": np.nan,
            "survival/level/food": np.nan,
            **fields,
        })
    return rows


def frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """The eval-episode frame of ``rows``, the stalls column an object column."""
    out = pd.DataFrame(rows)
    out["stalls"] = pd.Series([np.array(s, dtype=object) for s in out["stalls"]])
    return out


def floors(seed: int, values: list[int], **kw: Any) -> list[dict[str, Any]]:
    """Final stochastic episodes of a run that reached the given floors."""
    return run_rows(seed, [{"max_floor": f} for f in values], **kw)


# ---------------------------------------------------------------------- floors


def floor_frame() -> pd.DataFrame:
    """Three seeds with four final episodes each, plus rows that must be ignored."""
    rows = []
    for seed, values in enumerate(([0, 1, 1, 2], [0, 0, 1, 1], [0, 0, 0, 1])):
        rows += floors(seed, values)
        rows += floors(seed, [5] * 4, update=0)  # an earlier checkpoint
        rows += floors(seed, [8] * 4, mode="greedy")  # the other mode
    return frame(rows)


def test_floor_reach_rates_per_seed_then_mean() -> None:
    out = floor_reach(floor_frame(), n=N).set_index("floor")
    assert list(out.index) == list(range(9))
    assert out["rate"].iloc[:4].tolist() == pytest.approx([1.0, 0.5, 1 / 12, 0.0])
    assert out["reached_seeds"].iloc[:4].tolist() == [3, 3, 1, 0]
    assert set(out["n_seeds"]) == {3}
    assert set(out["n_episodes"]) == {12}
    assert out.loc[1, "floor_name"] == "dungeon"


def test_floor_reach_interval_is_the_shared_bootstrap_over_seeds() -> None:
    row = floor_reach(floor_frame(), n=N).set_index("floor").loc[1]
    want = stratified_bootstrap_ci(
        {"seeds": [0.75, 0.5, 0.25]}, lambda d: float(d["seeds"].mean()), n=N
    )
    assert isinstance(want, Interval)
    assert (row["rate"], row["lo"], row["hi"]) == pytest.approx(tuple(want))
    assert row["lo"] < row["rate"] < row["hi"]


def test_floor_reach_default_resamples_are_the_protocol_10000() -> None:
    out = floor_reach(floor_frame()).set_index("floor")
    want = stratified_bootstrap_ci(
        {"seeds": [0.75, 0.5, 0.25]}, lambda d: float(d["seeds"].mean())
    )
    assert out.loc[1, "lo"] == pytest.approx(want.lo)
    assert out.loc[1, "hi"] == pytest.approx(want.hi)


def test_floor_reach_status_follows_the_thresholds() -> None:
    out = floor_reach(floor_frame(), n=N).set_index("floor")
    assert out["status"].tolist() == ["common"] * 3 + ["never"] * 6
    # 1 episode of 100 reaches floor 3 (rare); exactly 5% is not rare (< 5%)
    rare = frame(floors(0, [3] + [0] * 99) + floors(0, [3] + [0] * 19, update=0))
    out = floor_reach(rare, n=N).set_index("floor")
    assert out.loc[3, "status"] == "rare"
    assert out.loc[3, "reached_seeds"] == 1
    edge = frame(floors(0, [1] + [0] * 19))
    assert RARE_RATE == 0.05
    assert floor_reach(edge, n=N).set_index("floor").loc[1, "status"] == "common"


def test_floor_reach_keeps_units_apart_and_budgets() -> None:
    rows = floors(0, [0, 1]) + floors(0, [0, 0], ref="simhash")
    rows += floors(0, [0, 2], method="n") + floors(0, [0, 0], budget="1b")
    out = floor_reach(frame(rows), n=N)
    keys = out[["method", "config_ref", "budget"]].drop_duplicates()
    assert len(keys) == 4
    one = out[(out["config_ref"] == "tuned") & (out["method"] == "m")]
    one = one[one["budget"] == "1m"].set_index("floor")
    assert one.loc[1, "rate"] == 0.5
    mixed = out[(out["config_ref"] == "simhash")].set_index("floor")
    assert mixed.loc[1, "rate"] == 0.0


def test_floor_reach_mode_parameter_and_empty_input() -> None:
    out = floor_reach(floor_frame(), mode="greedy", n=N).set_index("floor")
    assert out.loc[8, "rate"] == 1.0
    empty = floor_reach(floor_frame().iloc[:0], n=N)
    assert empty.empty
    assert "rate" in empty.columns


# ---------------------------------------------------------------- achievements


def wood(i: int) -> dict[str, int]:
    """collect_wood in the first two of 20 episodes."""
    return {"collect_wood": 50} if i < 2 else {}


def achievement_frame() -> pd.DataFrame:
    """Three seeds x (0, 122, 244) with 20 stochastic episodes each, and extras."""
    rows = []
    for seed in range(3):
        for update in (0, 122, 244):
            eps: list[dict[str, Any]] = [{} for _ in range(20)]
            if update == 122:  # collect_wood 40%, back to 10% at the end
                eps[:8] = [{"unlocked": {"collect_wood": 50}}] * 8
            if update == 244:
                eps[:2] = [{"unlocked": {"collect_wood": 50}}] * 2
                if seed == 1:  # place_table: one episode of one seed
                    eps[2] = {"unlocked": {"place_table": 90}}
                if seed == 0:  # enter_sewers: every episode of one seed
                    eps = [
                        {"unlocked": {"enter_sewers": 700} | wood(i)} for i in range(20)
                    ]
            rows += run_rows(seed, eps, update=update)
        # defeat_zombie only in greedy eval, seed 0 at the final checkpoint
        zombie = [{"unlocked": {"defeat_zombie": 30}}] * 5 if seed == 0 else [{}] * 5
        rows += run_rows(seed, zombie, mode="greedy")
    return frame(rows)


def curves(**logged: list[float | None]) -> pd.DataFrame:
    """Curves of the three runs: ``<name>=[seed 0, seed 1, seed 2]`` peak values.

    None leaves the metric unlogged for that seed; NaN is a series with no finished
    episode.
    """
    rows = []
    for name, peaks in logged.items():
        for seed, peak in enumerate(peaks):
            if peak is None:
                continue
            start = peak if peak != peak else 0.0  # NaN: no episode ever finished
            for step, value in enumerate([start, peak]):
                rows.append({
                    "run_id": f"m-tuned-{seed}",
                    "env_steps": step * 1000,
                    "metric": f"train/ach/{name}",
                    "value": value,
                })
    return pd.DataFrame(rows)


def status() -> pd.DataFrame:
    """Statuses with training logs: collect_diamond all zeros, make_arrow in seed 2."""
    logged = curves(
        collect_diamond=[0.0, 0.0, 0.0],
        make_arrow=[0.0, 0.0, 12.5],
        collect_wood=[1.0, 1.0, 1.0],
    )
    return achievement_status(achievement_frame(), logged).set_index("achievement")


def test_achievement_final_and_best_rate() -> None:
    out = status()
    wood = out.loc["collect_wood"]
    assert (wood["final_rate"], wood["best_rate"]) == pytest.approx((0.1, 0.4))
    assert wood["best_update"] == 122
    assert wood["seeds_ever"] == 3
    assert wood["n_seeds"] == 3
    # one episode of one seed in 20: seed rates 0, .05, 0
    assert out.loc["place_table", "final_rate"] == pytest.approx(0.05 / 3)
    assert out.loc["enter_sewers", "final_rate"] == pytest.approx(1 / 3)


def test_achievement_status_never_rare_common() -> None:
    out = status()
    assert out.loc["collect_wood", "status"] == "common"
    assert out.loc["enter_sewers", "status"] == "common"
    assert out.loc["place_table", "status"] == "rare"
    assert out.loc["place_table", "seeds_ever"] == 1
    # eval-only greedy unlock: a seed unlocked it, the stochastic final rate is 0
    assert out.loc["defeat_zombie", "status"] == "rare"
    assert out.loc["defeat_zombie", "seeds_ever"] == 1
    assert out.loc["defeat_zombie", "final_rate"] == 0
    # training-only unlock counts as an unlock of that seed
    assert out.loc["make_arrow", "seeds_ever"] == 1
    assert out.loc["make_arrow", "train_seeds"] == 1
    assert out.loc["make_arrow", "status"] == "rare"
    never = out.loc["collect_diamond"]
    assert never["status"] == "never"
    assert (never["seeds_ever"], never["train_seeds"]) == (0, 0)
    assert never["train_logged"] == 3


def test_achievement_never_needs_the_training_episodes_of_every_seed() -> None:
    episodes = achievement_frame()
    without = achievement_status(episodes).set_index("achievement")
    row = without.loc["collect_diamond"]
    assert row["status"] == "never in eval"
    assert np.isnan(row["train_seeds"])  # not logged, never read as 0
    assert row["train_logged"] == 0
    # a series with no finished episode (NaN) and a missing seed are not logged
    partial = curves(collect_diamond=[0.0, float("nan"), None])
    out = achievement_status(episodes, partial).set_index("achievement")
    assert out.loc["collect_diamond", "status"] == "never in eval"
    assert out.loc["collect_diamond", "train_logged"] == 1
    assert out.loc["collect_diamond", "train_seeds"] == 0


def test_achievement_status_with_curves_lacking_every_achievement_series() -> None:
    other = pd.DataFrame({
        "run_id": ["m-tuned-0"], "env_steps": [0], "metric": ["train/score_pct"],
        "value": [1.0],
    })  # fmt: skip
    out = achievement_status(achievement_frame(), other).set_index("achievement")
    assert (out["train_logged"] == 0).all()
    assert out["train_seeds"].isna().all()
    assert out.loc["collect_diamond", "status"] == "never in eval"


def test_achievement_rows_are_grouped_by_tier_floor_and_value() -> None:
    out = achievement_status(achievement_frame())
    assert list(out["achievement"]) == [
        "collect_wood", "place_table", "defeat_zombie", "collect_diamond",
        "make_arrow", "enter_dungeon", "enter_sewers",
    ]  # fmt: skip
    got = out.set_index("achievement")[["tier", "floor"]]
    assert got.loc["collect_wood"].tolist() == ["basic", 0]
    assert got.loc["enter_dungeon"].tolist() == ["intermediate", 1]
    assert got.loc["enter_sewers"].tolist() == ["advanced", 3]


def test_achievement_status_units_apart() -> None:
    a = achievement_frame()
    b = a.copy()
    b["config_ref"] = "simhash"
    b["run_id"] = b["run_id"].str.replace("tuned", "simhash")
    out = achievement_status(pd.concat([a, b], ignore_index=True))
    assert out.groupby("config_ref").size().tolist() == [len(NAMES)] * 2
    assert set(out["n_seeds"]) == {3}


def test_achievements_table_matches_the_game() -> None:
    from wut_murl_craftax.evaluation.milestones import (
        achievement_names,
        reward_weights,
    )

    names = achievement_names("craftax")
    assert tuple(ACHIEVEMENTS) == names  # value order, all 67
    tier = {1: "basic", 3: "intermediate", 5: "advanced", 8: "very_advanced"}
    weights = reward_weights("craftax")
    assert [ACHIEVEMENTS[n].tier for n in names] == [tier[int(w)] for w in weights]
    assert all(0 <= a.floor <= 8 for a in ACHIEVEMENTS.values())
    # the enter_* achievements sit on the floor they enter
    assert ACHIEVEMENTS["enter_dungeon"].floor == 1
    assert ACHIEVEMENTS["enter_graveyard"].floor == 8


def test_unknown_achievement_raises() -> None:
    episodes = achievement_frame()
    episodes["ach/not_a_craftax_achievement"] = False
    with pytest.raises(ValueError, match="not_a_craftax_achievement"):
        achievement_status(episodes)


# ---------------------------------------------------------------------- deaths


def death_frame() -> pd.DataFrame:
    """Two seeds, three checkpoints; the shares are chosen to be exact."""

    def eps(causes: list[str]) -> list[dict[str, Any]]:
        return [
            {"end_type": "death", "cause_of_death": c}
            if c != "timeout"
            else {"end_type": "timeout"}
            for c in causes
        ]

    rows = []
    for seed in range(2):
        rows += run_rows(seed, eps(["timeout"] * 3), update=0)
        rows += run_rows(
            seed, eps(["zombie", "zombie", "skeleton", "starvation", "timeout"]),
            update=122,
        )  # fmt: skip
        rows += run_rows(seed, eps(["lava", "lava", "", "timeout"]), update=244)
        rows += run_rows(seed, eps(["zombie"] * 4), update=244, mode="greedy")
    return frame(rows)


def test_death_causes_shares_sum_to_one_per_checkpoint() -> None:
    out = death_causes(death_frame())
    deaths = out[out["n_deaths"] > 0]
    assert deaths.groupby("update")["share"].sum().tolist() == pytest.approx([1, 1])
    mid = out[out["update"] == 122].set_index("cause")
    assert mid["share"].to_dict() == pytest.approx({
        "zombie": 0.5, "skeleton": 0.25, "starvation": 0.25
    })  # fmt: skip
    assert mid["n_deaths"].to_dict() == {"zombie": 4, "skeleton": 2, "starvation": 2}
    assert set(mid["n_episodes"]) == {10}
    assert set(mid["n_seeds"]) == {2}


def test_death_causes_final_flag_modes_and_unknown_cause() -> None:
    out = death_causes(death_frame())
    assert out.groupby("update")["is_final"].first().to_dict() == {
        0: False, 122: False, 244: True
    }  # fmt: skip
    last = out[out["is_final"]].set_index("cause")
    # greedy zombie deaths are out of the mode; "" is a death of unknown cause
    assert last["share"].to_dict() == pytest.approx({"lava": 2 / 3, "unknown": 1 / 3})
    greedy = death_causes(death_frame(), mode="greedy")
    assert greedy[greedy["is_final"]]["share"].tolist() == [1.0]


def test_death_causes_checkpoint_without_deaths_is_not_available() -> None:
    first = death_causes(death_frame()).query("update == 0")
    assert first["cause"].tolist() == [NO_DEATHS]
    assert first["n_deaths"].tolist() == [0]
    assert first["share"].isna().all()  # undefined, not 0
    assert first["n_episodes"].tolist() == [6]


def test_death_causes_env_steps_and_units_apart() -> None:
    episodes = death_frame()
    other = episodes.copy()
    other["method"] = "n"
    other["run_id"] = other["run_id"].str.replace("m-", "n-")
    out = death_causes(pd.concat([episodes, other], ignore_index=True))
    assert out.groupby("method").size().tolist() == [len(death_causes(episodes))] * 2
    assert set(out.loc[out["update"] == 122, "env_steps"]) == {122 * 4096}


# ---------------------------------------------------------------------- stalls


def stall_frame() -> pd.DataFrame:
    """Two seeds with the stalls of the module note; an earlier checkpoint on top."""
    a = {
        "unlocked": {"collect_wood": 400},
        "stalls": [
            stall(0, 400, 0, "unlock", "collect_wood"),
            stall(400, 900, 1, "timeout"),
        ],
    }
    c = {"end_type": "death", "stalls": [stall(0, 300, 0, "death")]}
    d = {
        "unlocked": {"collect_wood": 20, "place_table": 700},
        "stalls": [stall(20, 700, 2, "unlock", "place_table")],
    }
    rows = run_rows(0, [a, {}]) + run_rows(1, [c, d, {}])
    rows += run_rows(0, [{"stalls": [stall(0, 5000, 0, "timeout")]}], update=0)
    rows += run_rows(0, [{"stalls": [stall(0, 4000, 0, "timeout")]}], mode="greedy")
    return frame(rows)


def test_stall_table_rows_final_checkpoint_and_started_by() -> None:
    out = stall_table(stall_frame())
    frames = ["start", "end", "length", "floor", "ending", "ended_by"]
    assert out[frames].to_numpy().tolist() == [
        [0, 400, 400, 0, "unlock", "collect_wood"],
        [400, 900, 500, 1, "timeout", ""],
        [0, 300, 300, 0, "death", ""],
        [20, 700, 680, 2, "unlock", "place_table"],
    ]  # fmt: skip
    assert out["started_by"].tolist() == [
        "(episode start)", "collect_wood", "(episode start)", "collect_wood"
    ]  # fmt: skip
    assert out["seed"].tolist() == [0, 0, 1, 1]
    assert {"method", "config_ref", "budget", "run_id", "update", "eval_seed"} <= set(
        out
    )
    everything = stall_table(stall_frame(), final=False)
    assert 5000 in everything["length"].tolist()
    assert 4000 not in everything["length"].tolist()  # still the stochastic mode


def test_stall_table_unmatched_start_and_json_cells() -> None:
    episodes = stall_frame()
    assert (
        stall_table(episodes.drop(columns=["unlock_step/collect_wood"]))[
            "started_by"
        ].iloc[3]
        == "(unknown)"
    )
    as_csv = episodes.copy()
    as_csv["stalls"] = [
        json.dumps([
            {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in s.items()}
            for s in cell
        ])
        for cell in episodes["stalls"]
    ]
    pd.testing.assert_frame_equal(stall_table(as_csv), stall_table(episodes))


def test_stall_stats_scalars() -> None:
    out = stall_stats(stall_frame(), n=N).iloc[0]
    assert out["n_seeds"] == 2
    assert out["n_episodes"] == 5
    assert out["n_stalls"] == 4
    # seed means 2 / 2 and 2 / 3
    assert out["stalls_per_episode"] == pytest.approx((1.0 + 2 / 3) / 2)
    assert out["stalls_lo"] <= out["stalls_per_episode"] <= out["stalls_hi"]
    assert out["median_length"] == 450
    assert out["escape_rate"] == 0.5
    hist = {f: out[f"start_floor/{f}"] for f in range(9)}
    assert hist == {0: 2, 1: 1, 2: 1} | dict.fromkeys(range(3, 9), 0)


def test_stall_stats_interval_is_the_shared_bootstrap() -> None:
    out = stall_stats(stall_frame(), n=N).iloc[0]
    want = stratified_bootstrap_ci(
        {"seeds": [1.0, 2 / 3]}, lambda d: float(d["seeds"].mean()), n=N
    )
    assert (out["stalls_lo"], out["stalls_hi"]) == pytest.approx((want.lo, want.hi))


def test_stall_stats_without_stalls_is_not_available_rather_than_zero() -> None:
    quiet = frame(run_rows(0, [{}, {}]) + run_rows(1, [{}]))
    out = stall_stats(quiet, n=N).iloc[0]
    assert out["n_stalls"] == 0
    assert out["stalls_per_episode"] == 0
    assert np.isnan(out["median_length"])
    assert np.isnan(out["escape_rate"])
    assert out["start_floor/0"] == 0


def test_stall_stats_units_apart() -> None:
    episodes = stall_frame()
    other = episodes.copy()
    other["config_ref"] = "simhash"
    other["run_id"] = other["run_id"].str.replace("tuned", "simhash")
    out = stall_stats(pd.concat([episodes, other], ignore_index=True), n=N)
    assert out["config_ref"].tolist() == ["simhash", "tuned"]
    assert out["n_stalls"].tolist() == [4, 4]


def test_stall_starts_floor_and_achievement() -> None:
    out = stall_starts(stall_table(stall_frame()))
    got = {(r.floor, r.started_by): (r.n_stalls, r.share) for r in out.itertuples()}
    assert got == {
        (0, "(episode start)"): (2, 0.5),
        (1, "collect_wood"): (1, 0.25),
        (2, "collect_wood"): (1, 0.25),
    }


def test_stall_ccdf() -> None:
    table = pd.DataFrame({
        "method": "m", "config_ref": "tuned", "budget": "1m",
        "length": [300, 300, 500, 900],
    })  # fmt: skip
    out = stall_ccdf(table)
    assert out["length"].tolist() == [300, 500, 900]
    assert out["ccdf"].tolist() == [1.0, 0.5, 0.25]
    assert stall_ccdf(table.iloc[:0]).empty


# --------------------------------------------------------- replay-only columns


def test_replay_only_fields_are_not_read() -> None:
    """NaN gathered, coverage and survival columns change nothing."""
    wide = stall_frame()
    replay = [c for c in wide if c.startswith(("coverage", "gathered", "survival"))]
    assert replay
    plain = wide.drop(columns=replay)
    pd.testing.assert_frame_equal(floor_reach(plain, n=N), floor_reach(wide, n=N))
    pd.testing.assert_frame_equal(achievement_status(plain), achievement_status(wide))
    pd.testing.assert_frame_equal(death_causes(plain), death_causes(wide))
    pd.testing.assert_frame_equal(stall_stats(plain, n=N), stall_stats(wide, n=N))
    pd.testing.assert_frame_equal(stall_table(plain), stall_table(wide))
    assert not stall_table(wide).filter(regex="intrinsics|inventory").columns.size


# ----------------------------------------------------------------------- plots


def assert_png(path: Path) -> None:
    assert path.is_file()
    data = path.read_bytes()
    assert data[:8] == PNG
    assert len(data) > 2_000


def test_plots_write_pngs(tmp_path: Path) -> None:
    episodes = pd.concat([achievement_frame(), stall_frame(), death_frame()])
    tuned = floor_reach(floor_frame(), n=N)
    ablation = tuned.assign(config_ref="simhash", rate=tuned["rate"] / 2)
    both = pd.concat([tuned, ablation], ignore_index=True)
    paths = [
        plot_floor_reach(both, tmp_path),
        plot_achievement_status(achievement_status(achievement_frame()), tmp_path),
        plot_death_causes(death_causes(death_frame()), tmp_path),
        plot_stall_lengths(stall_ccdf(stall_table(episodes)), tmp_path),
    ]
    assert [p.name for p in paths] == [
        "floor_reach.png",
        "achievement_status.png",
        "death_causes_over_training.png",
        "stall_lengths.png",
    ]
    for path in paths:
        assert_png(path)


def test_plots_of_empty_tables_say_so(tmp_path: Path) -> None:
    for fn, table in (
        (plot_floor_reach, floor_reach(floor_frame().iloc[:0])),
        (plot_achievement_status, achievement_status(achievement_frame().iloc[:0])),
        (plot_death_causes, death_causes(death_frame().iloc[:0])),
        (plot_stall_lengths, stall_ccdf(stall_table(stall_frame()).iloc[:0])),
    ):
        assert_png(fn(table, tmp_path))


def test_plots_refuse_several_budgets(tmp_path: Path) -> None:
    both = pd.concat([
        floor_reach(floor_frame(), n=N),
        floor_reach(floor_frame().assign(budget="1b"), n=N),
    ])
    with pytest.raises(ValueError, match="budget"):
        plot_floor_reach(both, tmp_path)


def test_plot_images_differ_by_data(tmp_path: Path) -> None:
    """A plot is drawn from its table: another table gives other pixels."""
    one = plot_floor_reach(floor_reach(floor_frame(), n=N), tmp_path / "a")
    flat = floor_reach(floor_frame(), n=N).assign(rate=0.2, lo=0.1, hi=0.3)
    two = plot_floor_reach(flat, tmp_path / "b")
    assert one.read_bytes() != two.read_bytes()


# ------------------------------------------------------------- export fixture


def test_summaries_on_the_export_fixture(export_root: Path, tmp_path: Path) -> None:
    frames = load_phases(["final-1m"], export_root)
    episodes = frames["eval_episodes"]
    n_units = len(list(units(frames)))
    assert n_units > 2
    reach = floor_reach(episodes, n=N)
    assert len(reach) == 9 * n_units
    floors_of = reach.groupby(["method", "config_ref"])["floor"].apply(list)
    assert floors_of.map(lambda f: f == list(range(9))).all()
    stalls = stall_table(episodes)
    assert {"method", "config_ref"} <= set(stalls)
    stats = stall_stats(episodes, n=N)
    assert len(stats) == n_units
    deaths = death_causes(episodes)
    shares = (
        deaths[deaths["n_deaths"] > 0]
        .groupby(["method", "config_ref", "update"])["share"]
        .sum()
    )
    assert shares.to_numpy() == pytest.approx(1.0)
    status = achievement_status(episodes)
    assert set(status["achievement"]) == {
        "collect_wood",
        "place_table",
        "defeat_zombie",
    }
    sil = episodes[episodes["method"] == "sil"]
    assert not sil.empty
    assert_png(
        plot_stall_lengths(stall_ccdf(stalls[stalls["method"] == "sil"]), tmp_path)
    )
    assert_png(plot_death_causes(death_causes(sil), tmp_path))


def test_import_is_light() -> None:
    code = (
        "import sys, wut_murl_craftax.analysis.hard; "
        "print('mlflow' in sys.modules, 'jax' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.split() == ["False", "False"]
