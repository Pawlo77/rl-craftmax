"""Tests for the export loaders: schema, run membership, controls and missing phases."""

import json
import logging
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest
from _analysis_export import (
    COMBO_PAIR,
    FINAL_GROUPS,
    SEEDS,
    UPDATES,
)

from wut_murl_craftax.analysis.load import (
    _COMBO,
    BASE_CONTROL,
    CONTROLS,
    EPISODE_REQUIRED,
    FINAL_PHASES,
    RUNS_COLUMNS,
    UNIT_COLUMNS,
    MissingPhaseError,
    SchemaError,
    _control_maps,
    atomic_path,
    base_control,
    final_checkpoint,
    in_phase,
    load_phase,
    load_phases,
    pair_of,
    select,
    units,
)
from wut_murl_craftax.evaluation import select_pair
from wut_murl_craftax.evaluation.select_pair import SKILL_METHODS, controls_of
from wut_murl_craftax.training import interface
from wut_murl_craftax.training.interface import METHODS

COMBO_1M = json.loads(
    (Path(__file__).parents[1] / "training/fixtures/combo-1m-config.json").read_text()
)
"""``config.json`` of the real combo-1m runs: no ``combine`` block."""
N_FINAL = len(FINAL_GROUPS) * len(SEEDS)
"""Runs of the fixture's ``final-1m``."""


# --- the fixture itself ---------------------------------------------------------


def test_the_fixture_covers_what_the_comparison_track_needs(export_root: Path) -> None:
    runs = load_phases(["final-1m", "combo-1m"], export_root)["runs"]
    methods = set(runs["method"])
    assert set(CONTROLS) <= methods
    assert methods & set(SKILL_METHODS)
    assert methods - set(CONTROLS) - set(SKILL_METHODS) - {"combo"}  # a hook method
    assert "combo" in methods
    assert (runs["config_ref"] != "tuned").any()
    assert set(runs.groupby(["method", "config_ref"]).size()) == {len(SEEDS)}


# --- load_phase -----------------------------------------------------------------


def test_load_phase_returns_the_three_tables_with_the_contract_columns(
    export_root: Path,
) -> None:
    frames = load_phase(export_root / "final-1m")
    assert set(frames) == {"runs", "curves", "eval_episodes"}
    assert set(RUNS_COLUMNS) <= set(frames["runs"].columns)
    assert set(EPISODE_REQUIRED) <= set(frames["eval_episodes"].columns)
    assert list(frames["curves"].columns) == ["run_id", "env_steps", "metric", "value"]
    assert len(frames["runs"]) == N_FINAL
    assert frames["runs"]["run_id"].is_unique
    assert set(frames["runs"]["phase"]) == {"final-1m"}
    assert len(frames["eval_episodes"]) == N_FINAL * len(UPDATES) * 2 * 2


def test_the_list_struct_columns_come_back_as_records(export_root: Path) -> None:
    episodes = load_phase(export_root / "final-1m")["eval_episodes"]
    stalls = [s for rows in episodes["stalls"] for s in rows]
    assert stalls
    assert {"start", "end", "length", "floor", "ending", "ended_by"} == set(stalls[0])
    assert (episodes["n_stalls"] == episodes["stalls"].map(len)).all()


def test_a_missing_column_names_the_column_the_table_and_the_phase(
    export_root: Path, tmp_path: Path, export_writer: Callable[..., Path]
) -> None:
    root = export_writer(tmp_path, ("final-1m",))
    path = root / "final-1m" / "runs.parquet"
    pd.read_parquet(path).drop(columns=["config_ref", "final_score"]).to_parquet(path)
    with pytest.raises(SchemaError, match=r"runs.*final-1m.*config_ref, final_score"):
        load_phase(root / "final-1m")


def test_a_run_outside_the_phase_tag_is_refused(
    tmp_path: Path, export_writer: Callable[..., Path]
) -> None:
    """An exporter that picked runs by experiment name would put them here (R77)."""
    root = export_writer(tmp_path, ("final-1m",))
    path = root / "final-1m" / "runs.parquet"
    runs = pd.read_parquet(path)
    runs.loc[0, "phases"] = "tune-coarse"
    runs.to_parquet(path)
    with pytest.raises(SchemaError, match=r"final-1m.*phases"):
        load_phase(root / "final-1m")


def test_curves_of_an_unknown_run_are_refused(
    tmp_path: Path, export_writer: Callable[..., Path]
) -> None:
    root = export_writer(tmp_path, ("final-1m",))
    path = root / "final-1m" / "curves.parquet"
    curves = pd.read_parquet(path)
    curves.loc[0, "run_id"] = "0" * 32
    curves.to_parquet(path)
    with pytest.raises(SchemaError, match=r"curves.*not in runs"):
        load_phase(root / "final-1m")


def test_episode_columns_that_disagree_with_runs_are_refused(
    tmp_path: Path, export_writer: Callable[..., Path]
) -> None:
    root = export_writer(tmp_path, ("final-1m",))
    path = root / "final-1m" / "eval_episodes.parquet"
    episodes = pd.read_parquet(path)
    episodes.loc[0, "config_ref"] = "simhash"
    episodes.to_parquet(path)
    with pytest.raises(SchemaError, match=r"eval_episodes.*disagree"):
        load_phase(root / "final-1m")


# --- load_phases ----------------------------------------------------------------


def test_load_phases_concatenates_phases(export_root: Path) -> None:
    frames = load_phases(["final-1m", "combo-1m"], export_root)
    runs = frames["runs"]
    assert len(runs) == N_FINAL + len(SEEDS)
    assert set(runs["phase"]) == {"final-1m", "combo-1m"}
    assert frames["eval_episodes"]["run_id"].isin(runs["run_id"]).all()
    assert frames["curves"]["run_id"].isin(runs["run_id"]).all()


def test_a_run_of_two_phases_is_kept_once_from_the_first(export_root: Path) -> None:
    both = load_phases(["final-1m", "tune-coarse"], export_root)
    controls = both["runs"].query("method in ['ppo', 'ppo-rnn']")
    assert len(both["runs"]) == N_FINAL
    assert set(controls["phase"]) == {"final-1m"}
    assert set(controls["config_ref"]) == {"tuned"}
    flipped = load_phases(["tune-coarse", "final-1m"], export_root)["runs"]
    assert flipped["run_id"].is_unique
    assert set(flipped.query("phase == 'tune-coarse'")["method"]) == {"ppo", "ppo-rnn"}
    for frames in (both, load_phases(["tune-coarse", "final-1m"], export_root)):
        keys = ["run_id", "update", "mode", "eval_seed"]
        assert not frames["eval_episodes"].duplicated(keys).any()
        assert not frames["curves"].duplicated(["run_id", "env_steps", "metric"]).any()
        owner = frames["runs"].set_index("run_id")["config_ref"]
        episodes = frames["eval_episodes"]
        assert (episodes["config_ref"] == episodes["run_id"].map(owner)).all()


def test_a_phase_listed_twice_loads_once(export_root: Path) -> None:
    once = load_phases(["final-1m"], export_root)
    twice = load_phases(["final-1m", "final-1m"], export_root)
    for name, frame in once.items():
        pd.testing.assert_frame_equal(frame, twice[name])


def test_a_missing_phases_tag_is_a_schema_error(
    tmp_path: Path, export_writer: Callable[..., Path]
) -> None:
    root = export_writer(tmp_path, ("final-1m",))
    path = root / "final-1m" / "runs.parquet"
    runs = pd.read_parquet(path)
    runs.loc[runs.index[0], "phases"] = None
    runs.to_parquet(path)
    with pytest.raises(SchemaError, match="tag `phases`"):
        load_phase(root / "final-1m")


def test_load_phases_selects_by_the_phases_tag_not_the_experiment(
    export_root: Path,
) -> None:
    runs = load_phase(export_root / "final-1m")["runs"]
    controls = runs.query("method in ['ppo', 'ppo-rnn']")
    assert set(controls["experiment"]) == {
        "tune-coarse"
    }  # lives there, not in final-1m
    assert set(controls["phase"]) == {"final-1m"}
    assert in_phase(runs, "final-1m").all()
    assert set(runs[in_phase(runs, "tune-coarse")]["method"]) == {"ppo", "ppo-rnn"}


# --- atomic writes --------------------------------------------------------------


def test_atomic_path_replaces_on_success_and_leaves_no_temporary(
    tmp_path: Path,
) -> None:
    target = tmp_path / "f.txt"
    target.write_text("old")
    with atomic_path(target) as tmp:
        assert tmp.parent == tmp_path
        tmp.write_text("new")
        assert target.read_text() == "old"  # until the block ends
    assert target.read_text() == "new"
    assert [p.name for p in tmp_path.iterdir()] == ["f.txt"]


def test_atomic_path_keeps_the_old_file_and_removes_the_temporary_on_an_error(
    tmp_path: Path,
) -> None:
    target = tmp_path / "f.txt"
    target.write_text("old")

    def write_and_fail() -> None:
        with atomic_path(target) as tmp:
            tmp.write_text("half")
            msg = "disk full"
            raise OSError(msg)

    with pytest.raises(OSError, match="disk full"):
        write_and_fail()
    assert target.read_text() == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["f.txt"]


# --- missing phases (R86) -------------------------------------------------------


def test_a_phase_without_an_export_is_named(export_root: Path) -> None:
    with pytest.raises(MissingPhaseError, match="final-1b") as err:
        load_phase(export_root / "final-1b")
    assert isinstance(err.value, FileNotFoundError)
    with pytest.raises(MissingPhaseError, match="final-1b") as err:
        load_phases(["final-1m", "final-1b", "combo-1b"], export_root)
    assert err.value.phases == ["final-1b", "combo-1b"]  # not final-1m


def test_a_phase_missing_a_file_names_the_file(
    tmp_path: Path, export_writer: Callable[..., Path]
) -> None:
    root = export_writer(tmp_path, ("final-1m",))
    (root / "final-1m" / "curves.parquet").unlink()
    with pytest.raises(MissingPhaseError, match=r"final-1m.*curves.parquet"):
        load_phase(root / "final-1m")


def test_skip_missing_loads_only_the_phases_that_exist(export_root: Path) -> None:
    runs = load_phases(FINAL_PHASES, export_root, skip_missing=True)["runs"]
    assert set(runs["phase"]) == {"final-1m", "combo-1m"}
    assert set(runs["budget"]) == {"1m"}


def test_skip_missing_does_not_hide_an_incomplete_export(
    tmp_path: Path, export_writer: Callable[..., Path]
) -> None:
    root = export_writer(tmp_path, ("final-1m", "combo-1m"))
    (root / "combo-1m" / "runs.parquet").unlink()
    with pytest.raises(MissingPhaseError, match=r"combo-1m.*runs.parquet") as err:
        load_phases(FINAL_PHASES, root, skip_missing=True)
    assert err.value.phases == ["combo-1m"]


def test_skip_missing_still_fails_when_nothing_exists(tmp_path: Path) -> None:
    with pytest.raises(MissingPhaseError, match="final-1b"):
        load_phases(["final-1b"], tmp_path, skip_missing=True)


def test_a_later_budget_loads_beside_the_first(
    tmp_path: Path, export_writer: Callable[..., Path]
) -> None:
    root = export_writer(tmp_path, ("final-1m", "final-1b"))
    frames = load_phases(["final-1m", "final-1b"], root)
    assert set(frames["runs"]["budget"]) == {"1m", "1b"}
    assert set(select(frames, budget="1b", pool=True)["runs"]["phase"]) == {"final-1b"}


# --- select ---------------------------------------------------------------------


def test_select_filters_every_frame_consistently(export_root: Path) -> None:
    frames = load_phases(["final-1m", "combo-1m"], export_root)
    chosen = select(frames, method="smdp-skills", config_ref="monte-carlo")
    runs = chosen["runs"]
    assert len(runs) == len(SEEDS)
    assert set(runs["method"]) == {"smdp-skills"}
    for name in ("curves", "eval_episodes"):
        assert set(chosen[name]["run_id"]) == set(runs["run_id"])
    assert set(chosen["eval_episodes"]["config_ref"]) == {"monte-carlo"}
    assert len(frames["runs"]) == N_FINAL + len(SEEDS)  # the input is left alone


def test_select_refuses_to_pool_the_config_refs_of_a_method(export_root: Path) -> None:
    """R77: tuned and ablation runs of one method are separate statistics units."""
    frames = load_phase(export_root / "final-1m")
    with pytest.raises(ValueError, match=r"smdp-skills.*config_ref"):
        select(frames, method="smdp-skills")
    with pytest.raises(ValueError, match="count-bonus-ppo-rnn, smdp-skills"):
        select(frames)
    tuned = select(frames, method="smdp-skills", config_ref="tuned")["runs"]
    assert len(tuned) == len(SEEDS)
    pooled = select(frames, method="smdp-skills", pool=True)["runs"]
    assert set(pooled["config_ref"]) == {"tuned", "monte-carlo"}
    assert len(select(frames, method="sil")["runs"]) == len(SEEDS)  # one config_ref


def test_units_walks_each_method_config_ref_pair_with_its_own_frames(
    export_root: Path,
) -> None:
    frames = load_phases(["final-1m", "combo-1m"], export_root)
    walked = dict(units(frames))
    assert list(walked) == sorted(walked)
    assert ("smdp-skills", "monte-carlo") in walked
    assert ("combo", "tuned") in walked
    assert set(walked) == {tuple(x) for x in frames["runs"][list(UNIT_COLUMNS)].values}
    for (method, ref), unit in walked.items():
        assert set(unit["runs"]["method"]) == {method}
        assert set(unit["runs"]["config_ref"]) == {ref}
        assert len(unit["runs"]) == len(SEEDS)
        assert set(unit["eval_episodes"]["run_id"]) == set(unit["runs"]["run_id"])


def test_select_takes_a_budget_a_phase_tag_and_several_values(
    export_root: Path,
) -> None:
    frames = load_phases(["final-1m", "combo-1m"], export_root)
    assert len(select(frames, budget="1m", pool=True)["runs"]) == len(frames["runs"])
    by_tag = select(frames, phase="tune-coarse")["runs"]
    assert set(by_tag["method"]) == {"ppo", "ppo-rnn"}
    two = select(frames, method=["ppo", "combo"], config_ref="tuned")["runs"]
    assert set(two["method"]) == {"ppo", "combo"}


def test_select_with_no_match_returns_empty_frames_with_their_columns(
    export_root: Path,
) -> None:
    frames = load_phase(export_root / "final-1m")
    empty = select(frames, budget="1b")
    for name, frame in empty.items():
        assert frame.empty
        assert list(frame.columns) == list(frames[name].columns)


# --- controls -------------------------------------------------------------------


def test_the_controls_are_the_methods_that_are_their_own_base() -> None:
    assert set(CONTROLS) == {"ppo", "ppo-rnn", "pqn-rnn"}
    assert all(controls_of()[c] == c for c in CONTROLS)


def test_base_control_maps_every_non_combo_method_but_the_controls() -> None:
    assert BASE_CONTROL == {
        "count-bonus-ppo-rnn": "ppo-rnn",
        "count-bonus-pqn-rnn": "pqn-rnn",
        "sil": "ppo-rnn",
        "smdp-skills": "pqn-rnn",
        "dyna-q": "pqn-rnn",
    }
    assert set(BASE_CONTROL) == set(METHODS) - set(CONTROLS) - {"combo"}


def test_base_control_of_a_run_row(export_root: Path) -> None:
    runs = load_phases(["final-1m", "combo-1m"], export_root)["runs"]
    got = {
        m: base_control(r)
        for m, r in runs
        .drop_duplicates("method")
        .set_index("method", drop=False)
        .iterrows()
    }
    assert (
        got
        == {
            "ppo": "ppo",
            "ppo-rnn": "ppo-rnn",
            "pqn-rnn": "pqn-rnn",
            "combo": "ppo-rnn",
        }
        | BASE_CONTROL
    )
    assert base_control({"method": "ppo", "config_json": "{}"}) == "ppo"


def test_a_combo_run_takes_the_base_of_its_own_pair_not_of_combo_yaml() -> None:
    row = {
        "method": "combo",
        "config_json": json.dumps({
            "method": "combo",
            "combine": {
                "base": "pqn-rnn",
                "methods": ["count-bonus-pqn-rnn", "dyna-q"],
            },
        }),
    }
    assert controls_of()["combo"] == "ppo-rnn"  # what the live combo.yaml says
    assert base_control(row) == "pqn-rnn"
    assert pair_of(row) == {
        "base": "pqn-rnn",
        "methods": ["count-bonus-pqn-rnn", "dyna-q"],
    }


def test_a_combo_run_without_a_recorded_pair_infers_it_without_combo_yaml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unread() -> None:
        msg = "combo.yaml read"
        raise AssertionError(msg)

    monkeypatch.setattr(interface, "_combo", unread)
    row = {"method": "combo", "config_json": json.dumps(COMBO_1M)}
    assert pair_of(row) == {
        "base": "ppo-rnn",
        "methods": ["count-bonus-ppo-rnn", "sil"],
    }
    assert base_control(row) == "ppo-rnn"


def test_pair_of_a_non_combo_run_is_none(export_root: Path) -> None:
    runs = load_phase(export_root / "final-1m")["runs"]
    assert pair_of(runs.iloc[0]) is None
    combo = load_phase(export_root / "combo-1m")["runs"].iloc[0]
    assert pair_of(combo) == COMBO_PAIR


def test_base_control_of_an_unknown_method_is_an_error() -> None:
    with pytest.raises(ValueError, match="no-such-method"):
        base_control({"method": "no-such-method", "config_json": "{}"})


# --- final_checkpoint -----------------------------------------------------------


def test_final_checkpoint_keeps_the_last_update_of_one_run(export_root: Path) -> None:
    frames = load_phase(export_root / "final-1m")
    run = frames["runs"].iloc[0]["run_id"]
    last = final_checkpoint(frames["eval_episodes"], run)
    assert set(last["run_id"]) == {run}
    assert set(last["update"]) == {max(UPDATES)}
    assert set(last["mode"]) == {"stochastic", "greedy"}
    assert len(last) == 2 * 2


def test_final_checkpoint_without_a_run_uses_each_runs_own_last_update(
    export_root: Path,
) -> None:
    episodes = load_phase(export_root / "final-1m")["eval_episodes"]
    shortened = episodes[
        ~(
            (episodes["run_id"] == episodes["run_id"][0])
            & (episodes["update"] == max(UPDATES))
        )
    ]
    last = final_checkpoint(shortened)
    assert last["run_id"].nunique() == N_FINAL
    assert set(last[last["run_id"] == episodes["run_id"][0]]["update"]) == {
        min(UPDATES)
    }
    assert set(last[last["run_id"] != episodes["run_id"][0]]["update"]) == {
        max(UPDATES)
    }


def test_final_checkpoint_of_an_unknown_run_is_an_error(export_root: Path) -> None:
    episodes = load_phase(export_root / "final-1m")["eval_episodes"]
    with pytest.raises(ValueError, match=r"no eval episodes.*nope"):
        final_checkpoint(episodes, "nope")


def test_the_controls_never_include_combo_even_without_a_base_from(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A combo.yaml rewritten without ``base_from`` must not make combo a control."""
    real = controls_of()
    monkeypatch.setattr(select_pair, "controls_of", lambda: real | {"combo": "combo"})
    _control_maps.cache_clear()
    try:
        assert "combo" not in _control_maps()[0]
        assert "combo" not in _control_maps()[1]
    finally:
        monkeypatch.undo()
        _control_maps.cache_clear()
    assert select_pair.COMBO == _COMBO


# --- imports and duplicate runs -------------------------------------------------


def test_parquet_only_use_imports_neither_jax_nor_mlflow(export_root: Path) -> None:
    code = (
        "import sys\n"
        "from wut_murl_craftax.analysis.load import load_phases, select, "
        "final_checkpoint, units\n"
        "from pathlib import Path\n"
        f"f = load_phases(['final-1m'], Path({str(export_root)!r}))\n"
        "list(units(f)); select(f, config_ref='tuned')\n"
        "final_checkpoint(f['eval_episodes'])\n"
        "bad = [m for m in ('jax', 'mlflow') if m in sys.modules]\n"
        "assert not bad, bad\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_a_dropped_duplicate_with_another_phase_or_config_ref_is_warned_about(
    export_root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="wut_murl_craftax.analysis.load"):
        load_phases(["final-1m", "tune-coarse"], export_root)
    (record,) = caplog.records
    assert "6 runs" in record.message
    assert "final-1m" in record.message


def test_the_same_phase_twice_loads_once_without_a_warning(
    export_root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="wut_murl_craftax.analysis.load"):
        runs = load_phases(["final-1m", "final-1m"], export_root)["runs"]
    assert len(runs) == N_FINAL
    assert not caplog.records


def test_a_runs_only_load_never_reads_the_other_tables(
    export_writer: Callable[..., Path], tmp_path: Path
) -> None:
    root = export_writer(tmp_path, ("final-1m", "tune-coarse"))
    (root / "tune-coarse" / "curves.parquet").write_bytes(b"not parquet")
    (root / "tune-coarse" / "eval_episodes.parquet").unlink()
    got = load_phases(["tune-coarse"], root, tables=("runs",))
    assert len(got["runs"]) == 6
    assert list(got["curves"].columns) == ["run_id", "env_steps", "metric", "value"]
    assert got["curves"].empty
    assert got["eval_episodes"].empty
    assert set(EPISODE_REQUIRED) <= set(got["eval_episodes"].columns)


def test_a_runs_only_load_still_checks_the_runs_and_names_missing_runs(
    export_writer: Callable[..., Path], tmp_path: Path
) -> None:
    root = export_writer(tmp_path, ("final-1m",))
    (root / "final-1m" / "runs.parquet").unlink()
    with pytest.raises(MissingPhaseError, match=r"runs\.parquet"):
        load_phases(["final-1m"], root, tables=("runs",))
    with pytest.raises(ValueError, match="runs"):
        load_phases(["final-1m"], root, tables=("curves",))
