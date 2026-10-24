"""Tests for the comparison tables: blocks, Holm families, 1B rows, gate, T2, T6."""

import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from _compare_fixture import (
    ABLATIONS,
    COLLAPSING,
    MAIN_PAIR,
    NOT_LOGGED,
    SEEDS,
    SKILL_PAIR,
    episodes_frame,
    make_frames,
)

from wut_murl_craftax.analysis import compare
from wut_murl_craftax.analysis.compare import (
    ABLATION_COLUMNS,
    CONTEXT_NOTE,
    GAME_COLUMNS,
    GATE_COLUMNS,
    HARDWARE_COLUMNS,
    LITERATURE_TABLE_COLUMNS,
    POI_COLUMNS,
    SENSITIVITY_TABLE_COLUMNS,
    SKILL_METHODS,
    STABILITY_COLUMNS,
    ablation_table,
    block_of,
    game_summary_table,
    hardware_table,
    literature_table,
    main_columns,
    main_table,
    poi_matrix,
    race_table,
    reproduction_gate,
    sensitivity_table,
    stability_table,
)
from wut_murl_craftax.analysis.literature import COMPARABLE_BLOCKS
from wut_murl_craftax.analysis.load import CONTROLS, load_phases
from wut_murl_craftax.evaluation import select_pair
from wut_murl_craftax.evaluation.reproduction import band
from wut_murl_craftax.evaluation.stability import (
    config_scores,
    robust_share,
    sensitivity,
)
from wut_murl_craftax.evaluation.stats import holm, seed_spread

N_BOOT = 200
"""Bootstrap resamples in tests (the protocol's 10,000 only slow them down)."""


@pytest.fixture(scope="module")
def frames() -> dict[str, pd.DataFrame]:
    """1M (10 seeds) and 1B (3 seeds); combo pairs two main-block methods."""
    return make_frames()


@pytest.fixture(scope="module")
def table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return main_table(frames, n_boot=N_BOOT)


def _block_methods(t: pd.DataFrame, budget: str, block: str) -> list[str]:
    rows = t[(t["benchmark"] == budget) & (t["block"] == block)]
    return rows.loc[rows["role"] == "method", "method"].tolist()


# --- blocks --------------------------------------------------------------------


def test_skill_methods_are_select_pairs() -> None:
    assert SKILL_METHODS is select_pair.SKILL_METHODS


@pytest.mark.parametrize(
    ("method", "pair", "block"),
    [
        ("smdp-skills", None, "skills"),
        ("dyna-q", None, "skills"),
        ("sil", None, "main"),
        ("count-bonus-pqn-rnn", None, "main"),
        ("ppo-rnn", None, "main"),
        ("combo", {"base": "ppo-rnn", "methods": list(MAIN_PAIR)}, "main"),
        ("combo", {"base": "ppo-rnn", "methods": list(SKILL_PAIR)}, "skills"),
        ("combo", {"base": "ppo", "methods": ["smdp-skills", "sil"]}, "skills"),
    ],
)
def test_block_of(method: str, pair: dict | None, block: str) -> None:
    assert block_of(method, pair) == block


def test_block_of_a_combo_needs_its_pair() -> None:
    with pytest.raises(ValueError, match="pair"):
        block_of("combo", None)


# --- T1 --------------------------------------------------------------------------


def test_every_table_reads_the_shared_export(export_root: Path) -> None:
    frames = load_phases(["final-1m", "combo-1m"], export_root)
    tables = {
        "stability": stability_table(frames),
        "hardware": hardware_table(frames),
        "game": game_summary_table(frames),
    }
    for t in tables.values():
        for block in ("main", "skills"):
            rows = t[(t["block"] == block) & (t["role"] == "control")]
            assert rows["method"].tolist() == list(CONTROLS)
    assert race_table(frames).empty  # no run of the fixture logged first_unlock/*
    assert tables["game"][["race_wins_steps", "race_wins_wall"]].isna().all().all()


def test_t1_on_the_shared_export(export_root: Path) -> None:
    frames = load_phases(["final-1m", "combo-1m"], export_root)
    t = main_table(frames, n_boot=N_BOOT)
    assert list(t.columns) == main_columns()
    assert not t.duplicated(["benchmark", "block", "method"]).any()
    for block in ("main", "skills"):
        rows = t[t["block"] == block]
        assert rows.loc[rows["role"] == "control", "method"].tolist() == list(CONTROLS)
    assert set(_block_methods(t, "1m", "skills")) == set(SKILL_METHODS)
    assert "combo" in _block_methods(t, "1m", "main")  # pair count-bonus + sil
    assert t["seeds"].eq(3).all()  # tuned runs only: the ablations are left out


def test_t1_rows_and_columns(table: pd.DataFrame) -> None:
    assert list(table.columns) == main_columns()
    assert not table.duplicated(["benchmark", "block", "method"]).any()
    assert table["benchmark"].unique().tolist() == ["1m", "1b"]
    for budget, block in itertools.product(("1m", "1b"), ("main", "skills")):
        rows = table[(table["benchmark"] == budget) & (table["block"] == block)]
        assert rows["method"].tolist()[:3] == list(CONTROLS)
    assert set(_block_methods(table, "1m", "skills")) == {"smdp-skills", "dyna-q"}
    one_m = table[table["benchmark"] == "1m"]
    assert one_m["seeds"].eq(SEEDS["1m"]).all()
    assert one_m[["iqm", "iqm_lo", "iqm_hi", "mean_lo", "mean_hi"]].notna().all().all()
    assert (one_m["iqm_lo"] <= one_m["iqm"]).all()
    assert (one_m["iqm"] <= one_m["iqm_hi"]).all()


def test_t1_holm_families_are_the_1m_pairs_of_one_block(
    frames: dict[str, pd.DataFrame], monkeypatch: pytest.MonkeyPatch
) -> None:
    families: list[set[tuple[str, str]]] = []
    real = compare.mann_whitney_holm

    def spy(comparisons, alpha=0.05):
        families.append(set(comparisons))
        return real(comparisons, alpha)

    monkeypatch.setattr(compare, "mann_whitney_holm", spy)
    t = main_table(frames, n_boot=N_BOOT)
    expected = [
        {(m, c) for m in _block_methods(t, "1m", block) for c in CONTROLS}
        for block in ("main", "skills")
    ]
    assert families == expected  # one call per block, 1M only
    one_m = t[(t["benchmark"] == "1m") & (t["role"] == "method")]
    for _, rows in one_m.groupby("block"):
        raw = rows[[f"p_{c}" for c in CONTROLS]].to_numpy().ravel()
        adjusted = rows[[f"p_holm_{c}" for c in CONTROLS]].to_numpy().ravel()
        np.testing.assert_allclose(adjusted, holm(raw))
    controls = t[t["role"] == "control"]
    assert controls[[f"p_holm_{c}" for c in CONTROLS]].isna().all().all()


def test_t1_1b_rows_are_indicative(table: pd.DataFrame) -> None:
    one_b = table[table["benchmark"] == "1b"]
    assert one_b["indicative"].all()
    assert not table.loc[table["benchmark"] == "1m", "indicative"].any()
    shown = ["mean", "se", "mean_lo", "mean_hi", "median", "median_lo", "median_hi"]
    assert one_b[[*shown, "t_lo", "t_hi"]].notna().all().all()
    absent = ["iqm", "iqm_lo", "iqm_hi"] + [
        f"{q}_{c}" for c in CONTROLS for q in ("poi", "poi_lo", "poi_hi", "p", "p_holm")
    ]
    assert one_b[absent].isna().all().all()
    methods = one_b[one_b["role"] == "method"]
    assert methods[[f"diff_{c}" for c in CONTROLS]].notna().all().all()


def test_t1_poi_against_itself_is_empty(table: pd.DataFrame) -> None:
    for c in CONTROLS:
        assert table.loc[table["method"] == c, f"poi_{c}"].isna().all()


# --- PoI matrix ------------------------------------------------------------------


def test_poi_matrix_has_no_cross_block_cell(
    frames: dict[str, pd.DataFrame], table: pd.DataFrame
) -> None:
    m = poi_matrix(frames, n_boot=N_BOOT)
    assert list(m.columns) == POI_COLUMNS
    assert m["benchmark"].unique().tolist() == ["1m"]
    for block, rows in m.groupby("block"):
        members = set(table.loc[table["block"] == block, "method"])
        assert set(rows["method"]) | set(rows["versus"]) == members
        n = len(members)
        assert len(rows) == n * (n - 1)
    skills = m[m["block"] == "skills"]
    assert not ({"sil", "combo"} & (set(skills["method"]) | set(skills["versus"])))
    pairs = m.set_index(["block", "method", "versus"])["poi"]
    for (block, a, b), p in pairs.items():
        assert p + pairs[(block, b, a)] == pytest.approx(1.0)


# --- ablations (R77) ---------------------------------------------------------------


def test_ablation_table_compares_each_ablation_with_its_tuned_config(
    frames: dict[str, pd.DataFrame],
) -> None:
    t = ablation_table(frames, n_boot=N_BOOT)
    assert list(t.columns) == ABLATION_COLUMNS
    assert set(zip(t["method"], t["config_ref"], strict=True)) == set(ABLATIONS)
    assert len(t) == 2 * len(ABLATIONS)
    runs = frames["runs"]
    for row in t.itertuples():
        mine = runs[(runs["budget"] == row.benchmark) & (runs["method"] == row.method)]
        tuned = mine.loc[mine["config_ref"] == "tuned", "final_score"].mean()
        ablated = mine.loc[mine["config_ref"] == row.config_ref, "final_score"].mean()
        assert row.tuned_mean == pytest.approx(tuned)
        assert row.diff == pytest.approx(ablated - tuned)
    one_m = t[t["benchmark"] == "1m"]
    assert set(one_m.loc[one_m["block"] == "skills", "method"]) == set(SKILL_METHODS)
    for _, rows in one_m.groupby("block"):
        np.testing.assert_allclose(rows["p_holm"], holm(rows["p"].tolist()))
    one_b = t[t["benchmark"] == "1b"]
    assert one_b[["poi_tuned", "p", "p_holm", "iqm"]].isna().all().all()


# --- reproduction gate -------------------------------------------------------------


def _published(tmp_path: Path, runs: pd.DataFrame) -> Path:
    """ppo-rnn at its own 1M mean (pass), ppo 3 points off (fail), 1B for both."""
    tuned = runs[(runs["config_ref"] == "tuned") & (runs["budget"] == "1m")]
    mean = tuned.groupby("method")["final_score"].mean()
    ref = {
        "ppo": {
            "1m": {"score": float(mean["ppo"]) + 3, "s_pub": 0.1},
            "1b": {"score": 11.9},
        },
        "ppo-rnn": {
            "1m": {"score": float(mean["ppo-rnn"]), "s_pub": 0.1},
            "1b": {"score": 15.3, "s_pub": 0.35},
        },
        "pqn-rnn": {"1b": {"score": 16.0, "s_pub": 1.31}},
    }
    ref["ppo"]["1b"]["s_pub"] = 0.91
    path = tmp_path / "published.yaml"
    path.write_text(yaml.safe_dump(ref))
    return path


def test_gate_passes_fails_and_notes_without_raising(tmp_path: Path) -> None:
    frames = make_frames(("1m",))
    gate = reproduction_gate(frames, _published(tmp_path, frames["runs"]))
    assert list(gate.columns) == GATE_COLUMNS
    assert gate["benchmark"].unique().tolist() == ["1m"]  # no 1B runs: band skipped
    row = gate.set_index("control")
    runs = frames["runs"]
    for control in ("ppo", "ppo-rnn"):
        x = runs.loc[
            (runs["method"] == control) & (runs["config_ref"] == "tuned"),
            "final_score",
        ]
        h, passed = band(x.tolist(), row.loc[control, "published"], 0.1)
        assert row.loc[control, "h"] == pytest.approx(h)
        assert row.loc[control, "gap"] == pytest.approx(
            x.mean() - row.loc[control, "published"]
        )
        assert row.loc[control, "status"] == ("pass" if passed else "fail")
    assert row.loc["ppo", "status"] == "fail"
    assert row.loc["ppo-rnn", "status"] == "pass"
    pqn = row.loc["pqn-rnn"]
    assert pqn["check"] == "reference code"
    assert pqn["status"] == "pass"
    assert "1.509" in pqn["note"]
    assert math.isnan(pqn["h"])


def test_gate_reads_1b_and_survives_a_single_seed(tmp_path: Path) -> None:
    frames = make_frames()
    runs = frames["runs"]
    lone = (runs["method"] == "pqn-rnn") & (runs["budget"] == "1b") & (runs["seed"] > 0)
    frames["runs"] = runs[~lone]
    gate = reproduction_gate(frames, _published(tmp_path, runs))
    one_b = gate[gate["benchmark"] == "1b"].set_index("control")
    assert one_b.loc["pqn-rnn", "status"] == "too few seeds"
    assert one_b.loc["ppo-rnn", "status"] in {"pass", "fail"}
    assert one_b.loc["ppo-rnn", "h"] > 0


# --- T2 ----------------------------------------------------------------------------


def test_t2_never_mixes_literature_blocks(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    t = literature_table(
        frames, published=_published(tmp_path, frames["runs"]), n_boot=N_BOOT
    )
    assert list(t.columns) == LITERATURE_TABLE_COLUMNS
    groups = t.groupby(["score_kind", "benchmark"])
    assert set(groups.groups) == set(COMPARABLE_BLOCKS)
    for key, rows in groups:
        lit = rows[rows["source"] == "literature"]
        assert set(lit["block"]) == {COMPARABLE_BLOCKS[key]}
        assert not lit["name"].str.contains(r"\[\[").any()
        ours = rows[rows["source"] == "ours"]
        assert set(ours["block"]) == {"main", "skills"}
        runs = frames["runs"]
        tuned = runs[(runs["budget"] == key[1]) & (runs["config_ref"] == "tuned")]
        assert set(ours["name"]) == set(tuned["method"])


def test_t2_places_our_ci_against_each_published_value(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    t = literature_table(
        frames, published=_published(tmp_path, frames["runs"]), n_boot=N_BOOT
    )
    for _, rows in t.groupby(["score_kind", "benchmark"]):
        ours = rows[rows["source"] == "ours"].drop_duplicates("name")
        for lit in rows[rows["source"] == "literature"].itertuples():
            v = lit.score_pct
            for col, mask in (
                ("ours_above", ours["ci_lo"] > v),
                ("ours_within", (ours["ci_lo"] <= v) & (v <= ours["ci_hi"])),
                ("ours_below", ours["ci_hi"] < v),
            ):
                listed = [s for s in getattr(lit, col).split(", ") if s]
                assert listed == ours.loc[mask, "name"].tolist()
    training = t[(t["score_kind"] == "training") & (t["source"] == "ours")]
    gated = training[training["role"] == "control"]
    assert gated["gate"].isin({"pass", "fail", "too few seeds"}).all()
    assert gated.loc[gated["gate"] == "fail", ["gap", "h"]].notna().all().all()
    assert training.loc[training["role"] == "method", "gate"].eq("").all()


# --- T3, T4 ------------------------------------------------------------------------


def test_t3_counts_collapses_and_matches_seed_spread(
    frames: dict[str, pd.DataFrame],
) -> None:
    t = stability_table(frames)
    assert list(t.columns) == STABILITY_COLUMNS
    one_m = t[(t["benchmark"] == "1m") & (t["block"] == "main")].set_index("method")
    assert one_m.loc[COLLAPSING, "runs_with_collapse"] == 1
    assert one_m.loc[COLLAPSING, "runs_with_drops"] == 1
    assert one_m.drop(COLLAPSING)["runs_with_collapse"].eq(0).all()
    runs = frames["runs"]
    x = runs.loc[
        (runs["method"] == "sil")
        & (runs["config_ref"] == "tuned")
        & (runs["budget"] == "1m"),
        "final_score",
    ]
    spread = seed_spread(x.tolist())
    assert one_m.loc["sil", "sd"] == pytest.approx(spread["sd"])
    assert one_m.loc["sil", "cv"] == pytest.approx(spread["cv"])
    assert one_m.loc["sil", "max_drawdown_median"] == pytest.approx(0.0)
    assert set(t.loc[t["block"] == "skills", "method"]) >= set(CONTROLS)


def test_t4_is_per_profile_with_the_context_note(
    frames: dict[str, pd.DataFrame],
) -> None:
    runs = frames["runs"].copy()
    cuda = runs["method"].isin(["ppo", "sil"]) & (runs["budget"] == "1m")
    runs.loc[cuda, "profile"] = "cuda"
    t = hardware_table({**frames, "runs": runs})
    assert list(t.columns) == HARDWARE_COLUMNS
    one_m = t[t["benchmark"] == "1m"]
    assert set(one_m.loc[one_m["profile"] == "cuda", "method"]) == {"ppo", "sil"}
    assert one_m.loc[one_m["profile"] == "laptop", "note"].eq(CONTEXT_NOTE).all()
    assert one_m.loc[one_m["profile"] == "cuda", "note"].eq("").all()
    row = one_m[(one_m["profile"] == "laptop") & (one_m["method"] == "ppo-rnn")]
    assert row["reused_runs"].tolist() == [SEEDS["1m"]] * 2  # main and skill block
    assert row["steady_sps"].iloc[0] == pytest.approx(1045.0)  # median of 1000..1090
    assert row["steps_per_joule"].isna().all()


# --- T6 ----------------------------------------------------------------------------


def _race_frames() -> dict[str, pd.DataFrame]:
    """Three main-block methods of 3 seeds racing to collect_wood.

    count-bonus-ppo-rnn: 2 of 3 unlock (median 100); sil: 1 of 3 (50, too few);
    count-bonus-pqn-rnn: 3 of 3 (median 200). Wall-clock reverses the order.
    """
    first = {
        "count-bonus-ppo-rnn": [100.0, 120.0, None],
        "sil": [50.0, None, None],
        "count-bonus-pqn-rnn": [180.0, 200.0, 220.0],
    }
    wall = {"count-bonus-ppo-rnn": 9.0, "sil": 1.0, "count-bonus-pqn-rnn": 2.0}
    runs, curves = [], []
    for method, steps in first.items():
        for seed, step in enumerate(steps):
            run_id = f"{method}-{seed}"
            runs.append({
                "run_id": run_id,
                "method": method,
                "budget": "1m",
                "seed": seed,
                "config_ref": "tuned",
                "profile": "laptop",
                "config_json": "{}",
            })
            names = ["first_unlock/place_table", "first_unlock_wall/place_table"]
            values = [5.0, 5.0]  # every seed unlocks place_table early
            if step is not None:
                names += ["first_unlock/collect_wood", "first_unlock_wall/collect_wood"]
                values += [step, wall[method] + seed]
            curves.append(
                pd.DataFrame({
                    "run_id": run_id,
                    "env_steps": [int(v) for v in values],
                    "metric": names,
                    "value": values,
                })
            )
    runs_frame = pd.DataFrame(runs)
    return {
        "runs": runs_frame,
        "curves": pd.concat(curves, ignore_index=True),
        "eval_episodes": episodes_frame(runs_frame),
    }


def test_race_winner_is_the_lowest_median_of_half_the_seeds() -> None:
    race = race_table(_race_frames())
    wood = race[race["achievement"] == "collect_wood"].set_index("method")
    assert wood.loc["sil", "k"] == 1
    assert not wood.loc["sil", "eligible"]
    assert wood.loc["count-bonus-ppo-rnn", "median_steps"] == pytest.approx(110.0)
    assert wood["winner_steps"].to_dict() == {
        "count-bonus-ppo-rnn": True,
        "sil": False,
        "count-bonus-pqn-rnn": False,
    }
    assert wood.loc["count-bonus-pqn-rnn", "winner_wall"]
    table = race[race["achievement"] == "place_table"]
    assert table["winner_steps"].all()  # a tie: every tied method wins


def test_t6_one_row_per_method_with_race_wins_and_headlines() -> None:
    t = game_summary_table(_race_frames())
    assert list(t.columns) == GAME_COLUMNS
    row = t.set_index("method")
    assert row.loc["count-bonus-ppo-rnn", "race_wins_steps"] == 2  # wood + table tie
    assert row.loc["sil", "race_wins_steps"] == 1
    assert row.loc["count-bonus-pqn-rnn", "race_wins_wall"] == 2
    assert row.loc["sil", "top_death_cause"] == "starvation"
    assert row.loc["sil", "top_death_share"] == pytest.approx(1.0)
    assert row.loc["sil", "deepest_floor"] == 1
    # floor 1 only in eval seed 0 of even seeds: 1 of 4 episodes in 2 of 3 runs
    assert row.loc["sil", "deepest_floor_share"] == pytest.approx(1 / 6)
    assert row.loc["sil", "adoption"] == pytest.approx(0.125)  # 1 of 4, 1 of 2 kinds
    assert row.loc["sil", "wasted_share"] == pytest.approx(0.25)


def test_t6_reports_not_logged_race_as_missing(
    frames: dict[str, pd.DataFrame],
) -> None:
    t = game_summary_table(frames)
    assert not t.duplicated(["benchmark", "block", "method"]).any()
    not_logged = t[t["method"].isin(NOT_LOGGED)]
    assert not_logged[["race_wins_steps", "race_wins_wall"]].isna().all().all()
    logged = t[~t["method"].isin(NOT_LOGGED)]
    assert logged["race_wins_steps"].notna().all()
    assert set(t.loc[t["block"] == "skills", "method"]) == {
        *CONTROLS,
        *SKILL_METHODS,
    }
    smdp = t[(t["method"] == "smdp-skills") & (t["benchmark"] == "1m")].iloc[0]
    assert smdp["top_death_cause"] == "zombie"


# --- T7 ----------------------------------------------------------------------------


def _tuning_runs() -> pd.DataFrame:
    """sil coarse grid: lr sets the score, clip barely matters; 3 seeds a config."""
    rows = []
    for i, (lr, clip) in enumerate(itertools.product((1e-4, 3e-4), (0.1, 0.2))):
        for seed in range(3):
            score = (1.0 if lr == 3e-4 else 0.5) + 0.01 * clip + 0.01 * seed
            rows.append({
                "run_id": f"sil-{i}-{seed}",
                "phase": "tune-coarse",
                "phases": "tune-coarse",
                "method": "sil",
                "budget": "1m",
                "config_hash": f"h{i}",
                "config_ref": f"h{i}",
                "seed": seed,
                "best": i == 3,
                "profile": "laptop",
                "final_score": score,
                "config_json": f'{{"lr": {lr}, "clip": {clip}, "seed": {seed}}}',
            })
    return pd.DataFrame(rows)


def test_t7_reports_the_most_sensitive_hyperparameters() -> None:
    runs = _tuning_runs()
    t = sensitivity_table({"runs": runs})
    assert list(t.columns) == SENSITIVITY_TABLE_COLUMNS
    row = t.iloc[0]
    assert (row["phase"], row["benchmark"], row["block"], row["method"]) == (
        "tune-coarse",
        "1m",
        "main",
        "sil",
    )
    expected = sensitivity(runs)
    assert row["top1"] == "lr" == expected["hyperparameter"].iloc[0]
    assert row["top1_s"] == pytest.approx(expected["sensitivity"].iloc[0])
    assert row["top2"] == "clip"
    assert (row["configs"], row["runs"]) == (4, 12)
    assert row["best_config"] == "h3"
    assert row["selected_config"] == "h3"
    assert row["robust_share"] == pytest.approx(robust_share(config_scores(runs)))
