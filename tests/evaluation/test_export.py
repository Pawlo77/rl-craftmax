"""Tests for the phase export to tidy tables, on a throwaway MLflow store."""

import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any, NamedTuple

import mlflow
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from mlflow.entities import Metric

from wut_murl_craftax.analysis.load import RUNS_COLUMNS, load_phase, load_phases
from wut_murl_craftax.evaluation import export
from wut_murl_craftax.evaluation.episode_records import stored_records
from wut_murl_craftax.evaluation.events import NO_SOURCE
from wut_murl_craftax.evaluation.export import (
    ANALYSIS_EXPERIMENT,
    config_refs,
    export_phase,
    log_export,
)
from wut_murl_craftax.evaluation.milestones import CREATURES
from wut_murl_craftax.evaluation.rollout_metrics import (
    RUN_IDENTITY,
    RunRef,
    arrow_table,
    episode_columns,
    out_paths,
)
from wut_murl_craftax.orchestration.queue import PHASES_DIR
from wut_murl_craftax.training.checkpoints import mark_pruned

T0 = 1_700_000_000_000
"""Wall-clock time (ms) of the first fake sample."""
SPU = 1000
"""Env steps per update of the fake runs."""
UPDATES = (0, 2)
"""Checkpoints with eval files."""
EVAL_SEEDS = (10000, 10001)
LENGTHS = (320, 7)
"""Episode length per eval seed; the first stalls from its unlock at 3 to the end."""
WOOD = 0
"""``Achievement.COLLECT_WOOD.value``."""
HASH = {
    ("ppo", "tuned"): "e06ac6db",
    ("sil", "tuned"): "6f8690ad",
    ("sil", "single-buffer"): "57a864f9",
}
"""config_hash of each (method, config_ref) in configs/phases/final-1m.yaml."""
LAPTOP = {
    "device_kind": "Apple GPU",
    "actor_device": "cpu:0",
    "learner_device": "MetalDevice(id=0)",
}


class Run(NamedTuple):
    """A fake run: identity and the values its metrics were given."""

    run_id: str
    name: str
    method: str
    ref: str
    seed: int
    phases: str
    final_score: float


def _events(n: int) -> dict[str, np.ndarray]:
    """Stored event fields of ``n`` quiet Craftax steps with wood unlocked at step 2."""

    def z(*shape: int, dtype: Any = bool) -> np.ndarray:
        return np.zeros((n, *shape), dtype)

    ev = {
        "unlocked": z(67), "floor": z(dtype=np.int8),
        "health_delta": z(dtype=np.float32), "health_loss": z(dtype=np.float16),
        "drain": z(), "combat_damage": z(dtype=np.float16),
        "damage_source": np.full(n, NO_SOURCE, np.int8),
        "kills": z(len(CREATURES), dtype=np.int8), "passive_kills": z(dtype=np.int8),
        "damage_dealt": z(dtype=np.float16), "projectile_damage": z(dtype=np.float16),
        "ate": z(), "drank": z(), "slept": z(), "stat_low": z(3), "stat_zero": z(3),
        "forced": z(), "effective": z(), "hostiles_on_screen": z(dtype=np.int8),
        "hostiles_close": z(dtype=np.int8),
    }  # fmt: skip
    ev["unlocked"][2, WOOD] = True
    return ev


def _eval_arrays(update: int) -> dict[str, np.ndarray]:
    """An eval file of the two eval seeds; actions differ per update."""
    width = max(LENGTHS)
    per_seed = [_events(width) for _ in EVAL_SEEDS]
    arrays = {f"events/{k}": np.stack([e[k] for e in per_seed]) for k in per_seed[0]}
    actions = np.full((len(EVAL_SEEDS), width), -1, np.int8)
    for i, n in enumerate(LENGTHS):
        actions[i, :n] = update % 5
    return arrays | {
        "actions": actions,
        "length": np.array(LENGTHS),
        "seeds": np.array(EVAL_SEEDS),
    }


def _artifacts(run_dir: Path, method: str) -> None:
    """config.json, the checkpoint index and both modes' eval files of every update."""
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {"method": method, "env": "Craftax-Symbolic-v1", "lr": 0.001}
    (run_dir / "config.json").write_text(json.dumps(config))
    index = [
        {
            "update": u,
            "env_steps": u * SPU,
            "wall_s": float(u),
            "file": f"{u:05d}.msgpack",
        }
        for u in UPDATES
    ]
    (run_dir / "ckpt").mkdir()
    (run_dir / "ckpt" / "index.json").write_text(json.dumps(index))
    for mode in ("stochastic", "greedy"):
        (run_dir / "eval" / mode).mkdir(parents=True)
        for u in UPDATES:
            np.savez(run_dir / "eval" / mode / f"{u:05d}.npz", **_eval_arrays(u))


def _metrics(run: Run) -> list[Metric]:
    """Training, eval, hardware and system series; step 2000 is logged twice."""
    last = UPDATES[-1] * SPU
    pts = {
        "train/env_steps": [(SPU, 1.0 * SPU), (last, 1.0 * last)],
        "train/wall_s": [(SPU, 10.0), (last, 12.0)],
        "train/score_pct": [(SPU, 0.5), (last, 1.0)],
        "final_score": [(last, run.final_score)],
        "eval/stochastic/score_pct": [(0, 0.1), (last, 2.5 + run.seed)],
        "eval/greedy/score_pct": [(0, 0.2), (last, 1.5 + run.seed)],
        "eval/stochastic/return": [(0, 1.0), (last, 5.0 + run.seed)],
        "eval/greedy/return": [(0, 2.0), (last, 3.0 + run.seed)],
        "compile_s": [(SPU, 3.0)],
        "peak_rss_mb": [(last, 500.0)],
    }
    if run.method == "sil":
        pts["train/action_entropy"] = [(SPU, 2.0), (last, 1.9)]
    out = [
        Metric(k, v, T0 + i * 1000, s)
        for k, series in pts.items()
        for i, (s, v) in enumerate(series)
    ]
    # a resume re-logs the last update later: the curve keeps the later value
    out.append(Metric("train/score_pct", 1.5, T0 + 60_000, last))
    out += [
        Metric("system/cpu_utilization_percentage", v, T0 + i * 10_000, i)
        for i, v in enumerate((10.0, 20.0, 30.0))
    ]
    out += [
        Metric("system/system_memory_usage_megabytes", v, T0 + i * 10_000, i)
        for i, v in enumerate((100.0, 300.0))
    ]
    return out


class Store(NamedTuple):
    """The fake store and its runs."""

    uri: str
    path: Path
    client: mlflow.MlflowClient
    runs: list[Run]
    root: Path


def _add_run(
    store: Store, method: str, ref: str, seed: int, phases: str, **extra: str
) -> Run:
    client = store.client
    experiment = phases.split(",")[0]
    exp = client.get_experiment_by_name(experiment)
    exp_id = (
        exp.experiment_id
        if exp
        else client.create_experiment(
            experiment, artifact_location=(store.root / "mlruns" / experiment).as_uri()
        )
    )
    config_hash = HASH.get((method, ref), "deadbeef")
    name = f"{method}-1m-{config_hash}-s{seed}"
    tags = {
        "method": method,
        "budget": "1m",
        "seed": str(seed),
        "config_hash": config_hash,
        "phases": phases,
        **LAPTOP,
        **extra,
    }
    info = client.create_run(exp_id, run_name=name, tags=tags).info
    run = Run(info.run_id, name, method, ref, seed, phases, 1.25 + seed + len(ref))
    client.log_batch(run.run_id, metrics=_metrics(run))
    _artifacts(Path(info.artifact_uri.removeprefix("file://")), method)
    client.set_terminated(run.run_id)
    store.runs.append(run)
    return run


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> Store:
    """2 methods x 2 seeds of final-1m (ppo reused from tune-coarse), a sil ablation
    run, a tune-coarse-only ppo run and a smoke run."""
    root = tmp_path_factory.mktemp("store")
    path = root / "mlruns.db"
    uri = f"sqlite:///{path}"
    s = Store(uri, path, mlflow.MlflowClient(uri), [], root)
    for seed in (0, 1):
        _add_run(s, "ppo", "tuned", seed, "tune-coarse,final-1m", best="true")
        _add_run(s, "sil", "tuned", seed, "final-1m")
    _add_run(s, "sil", "single-buffer", 0, "final-1m")
    _add_run(s, "ppo", "other", 5, "tune-coarse")
    _add_run(s, "ppo", "tuned", 9, "smoke")
    return s


def _final(store: Store) -> list[Run]:
    return [r for r in store.runs if "final-1m" in r.phases.split(",")]


def _replay_file(store: Store, run: Run, replay: Path, update: int, mode: str) -> None:
    """What the replay writes for one eval file: its rows with coverage filled."""
    run_dir = Path(
        store.client.get_run(run.run_id).info.artifact_uri.removeprefix("file://")
    )
    arrays = dict(np.load(run_dir / "eval" / mode / f"{update:05d}.npz"))
    rows = []
    for seed, rec in zip(EVAL_SEEDS, stored_records(arrays, "craftax"), strict=True):
        head = {
            "run_id": run.run_id, "method": run.method, "budget": "1m",
            "seed": run.seed, "update": update, "env_steps": update * SPU,
            "mode": mode, "eval_seed": seed,
        }  # fmt: skip
        rows.append(head | rec.row())
    df = pd.DataFrame(rows)
    order = [c for c in episode_columns("craftax") if c != "config_ref"]
    for c in order:
        if c.startswith("coverage/") or c in ("new_tiles_per_100", "revisit_rate"):
            df[c] = 0.5
    df["unique_positions"] = 7
    paths = out_paths(replay, run.run_id, update, mode)
    paths["episodes"].parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(arrow_table(df[order]), paths["episodes"])
    inventory = pd.DataFrame({
        "run_id": run.run_id,
        "update": update,
        "mode": mode,
        "eval_seed": list(EVAL_SEEDS),
        "steps": list(LENGTHS),
        "capped_steps": [0, 3],
    })
    pq.write_table(arrow_table(inventory), paths["inventory"])


class Exported(NamedTuple):
    out: Path
    replay: Path
    replayed: Run
    """The run whose final stochastic file has a replay table."""


@pytest.fixture(scope="module")
def exported(store: Store, tmp_path_factory: pytest.TempPathFactory) -> Exported:
    """``final-1m`` and ``tune-coarse`` exported; one eval file has a replay table."""
    base = tmp_path_factory.mktemp("metrics")
    replay = base / "replay"
    replayed = _final(store)[0]
    _replay_file(store, replayed, replay, UPDATES[-1], "stochastic")
    for phase in ("final-1m", "tune-coarse"):
        export_phase(phase, base / phase, tracking_uri=store.uri, replay_root=replay)
    return Exported(base, replay, replayed)


def _runs(exported: Exported, phase: str = "final-1m") -> pd.DataFrame:
    return pd.read_parquet(exported.out / phase / "runs.parquet").set_index("run_id")


# --- config_ref ----------------------------------------------------------------------


def test_config_refs_of_final_1m_follow_the_contract_table() -> None:
    refs = config_refs(PHASES_DIR / "final-1m.yaml")
    assert refs[("ppo", "e06ac6db")] == "tuned"
    assert refs[("count-bonus-ppo-rnn", "b0b1c535")] == "tuned"
    ablations = {k: v for k, v in refs.items() if v != "tuned"}
    assert ablations == {
        ("count-bonus-ppo-rnn", "9403d21d"): "simhash",
        ("sil", "57a864f9"): "single-buffer",
        ("smdp-skills", "1f559f2c"): "random-top",
        ("smdp-skills", "20ec0807"): "monte-carlo",
        ("smdp-skills", "89f3c98f"): "no-flags",
        ("dyna-q", "3fd02b70"): "kappa0",
    }
    assert len(refs) == 14


def test_the_combo_phase_has_one_tuned_config() -> None:
    assert config_refs(PHASES_DIR / "combo-1m.yaml") == {("combo", "fd6c4333"): "tuned"}


def test_a_1b_phase_maps_its_own_hashes_by_the_same_rule() -> None:
    refs = config_refs(PHASES_DIR / "final-1b.yaml")
    assert set(refs.values()) == {"tuned"}
    assert {m for m, _ in refs} >= {"ppo", "ppo-rnn", "pqn-rnn"}
    assert ("ppo", "e06ac6db") not in refs


def test_an_entry_led_by_an_ablation_is_refused(tmp_path: Path) -> None:
    phase = tmp_path / "final-1m.yaml"
    text = (PHASES_DIR / "final-1m.yaml").read_text()
    phase.write_text(
        text.replace(
            "sil: {configs: [tuned, {ref: tuned, set: {layout: single}}]}",
            "sil: {configs: [{ref: tuned, set: {layout: single}}, tuned]}",
        )
    )
    with pytest.raises(ValueError, match="sil starts with"):
        config_refs(phase)


def test_an_ablation_without_a_name_is_refused(tmp_path: Path) -> None:
    phase = tmp_path / "final-1m.yaml"
    text = (PHASES_DIR / "final-1m.yaml").read_text()
    phase.write_text(text.replace("plan_kappa: 0.0", "plan_kappa: 0.001"))
    with pytest.raises(ValueError, match="dyna-q"):
        config_refs(phase)


# --- runs ----------------------------------------------------------------------------


def test_runs_has_one_row_per_run_of_the_phase_tag_with_the_contract_columns(
    store: Store, exported: Exported
) -> None:
    runs = pd.read_parquet(exported.out / "final-1m" / "runs.parquet")
    assert list(runs.columns[: len(RUNS_COLUMNS)]) == RUNS_COLUMNS
    assert sorted(runs["run_id"]) == sorted(r.run_id for r in _final(store))
    assert set(runs["phase"]) == {"final-1m"}
    by = runs.set_index("run_id")
    ppo = next(r for r in store.runs if r.method == "ppo" and r.seed == 0)
    row = by.loc[ppo.run_id]
    assert row["experiment"] == "tune-coarse"
    assert row["phases"] == "tune-coarse,final-1m"
    assert (row["method"], row["budget"], row["seed"]) == ("ppo", "1m", 0)
    assert row["config_hash"] == "e06ac6db"
    assert row["run_name"] == ppo.name
    assert row["status"] == "FINISHED"
    assert json.loads(row["config_json"])["lr"] == 0.001


def test_runs_carry_config_ref_and_best_by_phase_kind(
    store: Store, exported: Exported
) -> None:
    final, tune = _runs(exported), _runs(exported, "tune-coarse")
    for r in _final(store):
        assert final.loc[r.run_id, "config_ref"] == r.ref
    assert not final["best"].any()
    assert sorted(tune.index) == sorted(
        r.run_id for r in store.runs if "tune-coarse" in r.phases
    )
    assert (tune["config_ref"] == tune["config_hash"]).all()
    assert tune["best"].sum() == 2  # the two ppo runs tagged best


def test_runs_take_the_stored_final_score_and_the_last_eval_point(
    store: Store, exported: Exported
) -> None:
    runs = _runs(exported)
    for r in _final(store):
        row = runs.loc[r.run_id]
        assert row["final_score"] == pytest.approx(r.final_score)
        assert row["eval_stochastic_score"] == pytest.approx(2.5 + r.seed)
        assert row["eval_greedy_score"] == pytest.approx(1.5 + r.seed)
        assert row["eval_stochastic_return"] == pytest.approx(5.0 + r.seed)
        assert row["eval_greedy_return"] == pytest.approx(3.0 + r.seed)
        assert row["env_steps"] == UPDATES[-1] * SPU
        assert row["wall_clock_s"] == pytest.approx(12.0)


def test_runs_carry_the_hardware_summary_and_the_profile(exported: Exported) -> None:
    runs = _runs(exported)
    assert set(runs["profile"]) == {"laptop"}
    assert set(runs["learner_device"]) == {"MetalDevice(id=0)"}
    assert (runs["compile_s"] == 3.0).all()
    assert np.allclose(runs["cpu_util_mean"], 20.0)
    assert (runs["peak_host_mem_mb"] == 300.0).all()
    assert (runs["peak_rss_mb"] == 500.0).all()
    assert (
        runs[["eval_s", "gpu_util_mean", "energy_j", "steps_per_joule"]]
        .isna()
        .all(axis=None)
    )
    assert np.allclose(runs["overall_sps"], 1e6 / 12.0)


def test_online_families_is_false_without_action_entropy(
    store: Store, exported: Exported
) -> None:
    runs = _runs(exported)
    for r in _final(store):
        assert runs.loc[r.run_id, "online_families"] == (r.method == "sil")


def test_a_run_hash_the_phase_file_does_not_produce_is_refused(
    store: Store, tmp_path: Path
) -> None:
    db = tmp_path / "mlruns.db"
    shutil.copy(store.path, db)
    with sqlite3.connect(db) as con:
        con.execute(
            "update tags set value = 'abcdef12' where key = 'config_hash' "
            "and run_uuid = ?",
            (_final(store)[0].run_id,),
        )
    with pytest.raises(ValueError, match="abcdef12"):
        export_phase("final-1m", tmp_path / "out", tracking_uri=f"sqlite:///{db}")


def test_a_phase_without_runs_is_refused(store: Store, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="final-1b"):
        export_phase("final-1b", tmp_path / "out", tracking_uri=store.uri)


# --- curves --------------------------------------------------------------------------


def test_curves_are_long_without_system_series_and_keep_a_relogged_update_once(
    store: Store, exported: Exported
) -> None:
    curves = pd.read_parquet(exported.out / "final-1m" / "curves.parquet")
    assert list(curves.columns) == ["run_id", "env_steps", "metric", "value"]
    assert curves["env_steps"].dtype == np.int64
    assert curves["value"].dtype == np.float64
    assert not curves["metric"].str.startswith("system/").any()
    assert set(curves["run_id"]) == {r.run_id for r in _final(store)}
    assert not curves.duplicated(["run_id", "metric", "env_steps"]).any()
    one = curves[
        (curves["run_id"] == _final(store)[0].run_id)
        & (curves["metric"] == "train/score_pct")
    ]
    assert one.sort_values("env_steps")["value"].tolist() == [0.5, 1.5]
    assert "eval/stochastic/score_pct" in set(curves["metric"])


# --- eval episodes -------------------------------------------------------------------


def _episodes(exported: Exported, phase: str = "final-1m") -> pd.DataFrame:
    return pd.read_parquet(exported.out / phase / "eval_episodes.parquet")


def test_eval_episodes_have_a_row_per_run_update_mode_and_eval_seed(
    store: Store, exported: Exported
) -> None:
    ep = _episodes(exported)
    assert len(ep) == len(_final(store)) * len(UPDATES) * 2 * len(EVAL_SEEDS)
    assert not ep.duplicated(["run_id", "update", "mode", "eval_seed"]).any()
    expected = list(episode_columns("craftax"))
    assert list(ep.columns) == expected
    assert expected[:3] == ["run_id", "method", "config_ref"]
    runs = _runs(exported)
    assert (ep["config_ref"] == ep["run_id"].map(runs["config_ref"])).all()
    assert (ep["env_steps"] == ep["update"] * SPU).all()
    assert ep["length"].tolist()[:2] == list(LENGTHS)
    assert (ep["unlock_step/collect_wood"] == 3).all()
    assert ep["unlock_step/collect_wood"].dtype == np.int32


def test_replay_columns_are_filled_where_the_replay_wrote_them_and_nan_elsewhere(
    exported: Exported,
) -> None:
    ep = _episodes(exported)
    r = exported.replayed
    have = (
        (ep["run_id"] == r.run_id)
        & (ep["update"] == UPDATES[-1])
        & (ep["mode"] == "stochastic")
    )
    assert have.sum() == len(EVAL_SEEDS)
    assert (ep.loc[have, "coverage/overworld"] == 0.5).all()
    assert (ep.loc[have, "unique_positions"] == 7).all()
    assert ep.loc[~have, ["coverage/overworld", "revisit_rate"]].isna().all(axis=None)
    assert ep.loc[~have, "unique_positions"].isna().all()
    schema = pq.read_schema(exported.out / "final-1m" / "eval_episodes.parquet")
    assert str(schema.field("unique_positions").type) == "int64"


def test_the_export_warns_how_many_files_lack_a_replay_table(
    store: Store,
    exported: Exported,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING"):
        export_phase(
            "final-1m",
            tmp_path / "x",
            tracking_uri=store.uri,
            replay_root=exported.replay,
        )
    files = len(_final(store)) * len(UPDATES) * 2
    assert f"final-1m: {files - 1} of {files} eval files have no replay table" in (
        caplog.text
    )


def test_any_zero_share_is_exported_for_stored_and_replayed_files(
    exported: Exported,
) -> None:
    ep = _episodes(exported)
    zero = ep[[c for c in ep.columns if c.startswith("zero_share/")]]
    assert ep["any_zero_share"].notna().all()  # not a replay-only column
    assert (ep["any_zero_share"] >= zero.max(axis=1)).all()
    assert (ep["any_zero_share"] <= zero.sum(axis=1)).all()


def test_a_tuning_phase_gives_eval_episodes_the_hash_as_config_ref(
    exported: Exported,
) -> None:
    ep, runs = _episodes(exported, "tune-coarse"), _runs(exported, "tune-coarse")
    assert set(ep["run_id"]) == set(runs.index)
    assert (ep["config_ref"] == ep["run_id"].map(runs["config_hash"])).all()


def test_replay_identity_that_disagrees_with_the_tags_is_refused(
    store: Store, tmp_path: Path
) -> None:
    replay = tmp_path / "replay"
    run = _final(store)[1]
    _replay_file(store, run._replace(seed=7), replay, UPDATES[-1], "greedy")
    with pytest.raises(ValueError, match="seed"):
        export_phase(
            "final-1m", tmp_path / "out", tracking_uri=store.uri, replay_root=replay
        )


def test_the_replay_inventory_frames_are_concatenated_per_phase(
    exported: Exported,
) -> None:
    inv = pd.read_parquet(exported.out / "final-1m" / "eval_inventory.parquet")
    assert inv["run_id"].tolist() == [exported.replayed.run_id] * len(EVAL_SEEDS)
    assert inv["capped_steps"].tolist() == [0, 3]
    assert not (exported.out / "final-1m" / "skill_ends.parquet").exists()


# --- CSV copies and the loaders ------------------------------------------------------


def test_every_parquet_file_has_a_csv_copy_with_json_list_columns(
    exported: Exported,
) -> None:
    phase = exported.out / "final-1m"
    parquets = {p.stem for p in phase.glob("*.parquet")}
    assert parquets - {"curves"} == {p.stem for p in phase.glob("*.csv")}
    assert not (phase / "curves.csv").exists()  # R104: gzipped

    ep = _episodes(exported)
    csv = pd.read_csv(phase / "eval_episodes.csv")
    assert list(csv.columns) == list(ep.columns)
    assert len(csv) == len(ep)
    stalls = csv["stalls"].map(json.loads)
    assert stalls.map(len).tolist() == ep["n_stalls"].tolist()
    first = next(s for s in stalls if s)[0]
    assert set(first) == {
        "start", "end", "length", "floor", "ending", "ended_by", "actions",
        "intrinsics", "inventory",
    }  # fmt: skip
    assert csv["fights"].map(json.loads).map(len).sum() == ep["fights"].map(len).sum()
    runs = pd.read_csv(phase / "runs.csv")
    assert runs["run_id"].tolist() == _runs(exported).index.tolist()
    curves = pd.read_csv(phase / "curves.csv.gz")
    pd.testing.assert_frame_equal(curves, pd.read_parquet(phase / "curves.parquet"))


def test_the_analysis_loaders_read_the_export(exported: Exported) -> None:
    frames = load_phase(exported.out / "final-1m")
    assert set(frames) == {"runs", "curves", "eval_episodes"}
    stalls = frames["eval_episodes"]["stalls"]
    assert stalls.map(len).sum() == frames["eval_episodes"]["n_stalls"].sum()
    assert frames["eval_episodes"]["any_zero_share"].notna().all()
    both = load_phases(["final-1m", "tune-coarse"], exported.out)
    assert both["runs"]["run_id"].is_unique


def test_a_run_of_two_phases_is_exported_with_each(
    store: Store, exported: Exported
) -> None:
    final, tune = _runs(exported), _runs(exported, "tune-coarse")
    shared = set(final.index) & set(tune.index)
    assert shared == {
        r.run_id for r in store.runs if r.phases == "tune-coarse,final-1m"
    }
    for rid in shared:
        assert final.loc[rid, "config_ref"] == "tuned"
        assert tune.loc[rid, "config_ref"] == "e06ac6db"


# --- MLflow and CLI ------------------------------------------------------------------


def _analysis_runs(client: mlflow.MlflowClient, name: str) -> list[Any]:
    exp = client.get_experiment_by_name(ANALYSIS_EXPERIMENT)
    assert exp is not None
    return client.search_runs(
        [exp.experiment_id], filter_string=f"attributes.run_name = '{name}'"
    )


def test_the_export_is_logged_as_artifacts_of_one_analysis_run_per_phase(
    store: Store, exported: Exported, tmp_path: Path
) -> None:
    db = tmp_path / "mlruns.db"
    shutil.copy(store.path, db)
    uri, client = f"sqlite:///{db}", mlflow.MlflowClient(f"sqlite:///{db}")
    folder = exported.out / "final-1m"
    first = log_export(folder, "final-1m", uri)
    (run,) = _analysis_runs(client, "export-final-1m")
    assert run.info.run_id == first
    names = {a.path for a in client.list_artifacts(first)}
    assert names == {p.name for p in folder.glob("*.parquet")}  # R104
    tags = run.data.tags
    assert tags["analysis/kind"] == "export"
    assert tags["analysis/phases"] == "final-1m"
    assert tags["analysis/budget"] == "1m"
    assert not {"phases", "method", "budget", "seed", "config_hash"} & set(tags)
    assert run.info.status == "FINISHED"
    # a rerun reuses the run and replaces its artifacts
    smaller = tmp_path / "again"
    shutil.copytree(folder, smaller)
    (smaller / "eval_inventory.parquet").unlink()
    assert log_export(smaller, "final-1m", uri) == first
    assert len(_analysis_runs(client, "export-final-1m")) == 1
    gone = {"eval_inventory.parquet"}
    assert {a.path for a in client.list_artifacts(first)} == names - gone
    assert client.get_run(first).info.status == "FINISHED"


def test_a_failed_relogging_keeps_the_old_artifacts(
    store: Store, exported: Exported, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "mlruns.db"
    shutil.copy(store.path, db)
    uri, client = f"sqlite:///{db}", mlflow.MlflowClient(f"sqlite:///{db}")
    folder = exported.out / "final-1m"
    run_id = log_export(folder, "final-1m", uri)
    names = {a.path for a in client.list_artifacts(run_id)}
    smaller = tmp_path / "again"
    shutil.copytree(folder, smaller)
    (smaller / "eval_inventory.parquet").unlink()

    def broken(*args: Any, **kw: Any) -> None:
        msg = "disk full"
        raise OSError(msg)

    monkeypatch.setattr(mlflow.MlflowClient, "log_artifact", broken)
    with pytest.raises(OSError, match="disk full"):
        log_export(smaller, "final-1m", uri)
    assert {a.path for a in client.list_artifacts(run_id)} == names


def test_the_csv_copies_are_logged_only_on_request(
    store: Store, exported: Exported, tmp_path: Path
) -> None:
    db = tmp_path / "mlruns.db"
    shutil.copy(store.path, db)
    uri = f"sqlite:///{db}"
    folder = exported.out / "final-1m"
    run_id = log_export(folder, "final-1m", uri, csv=True)
    logged = {a.path for a in mlflow.MlflowClient(uri).list_artifacts(run_id)}
    assert logged == {p.name for p in folder.iterdir()}
    assert "curves.csv.gz" in logged


def test_cli_exports_a_phase_without_logging(store: Store, tmp_path: Path) -> None:
    out = tmp_path / "tune"
    export.main([
        "--phase", "tune-coarse", "--out", str(out), "--tracking-uri", store.uri,
        "--replay-root", str(tmp_path / "none"), "--no-log",
    ])  # fmt: skip
    assert {p.name for p in out.iterdir()} == {
        "runs.parquet", "runs.csv", "curves.parquet", "curves.csv.gz",
        "eval_episodes.parquet", "eval_episodes.csv",
    }  # fmt: skip
    exp = mlflow.MlflowClient(store.uri).get_experiment_by_name(ANALYSIS_EXPERIMENT)
    assert exp is None


def test_cli_logs_by_default(store: Store, tmp_path: Path) -> None:
    db = tmp_path / "mlruns.db"
    shutil.copy(store.path, db)
    uri = f"sqlite:///{db}"
    export.main([
        "--phase", "tune-coarse", "--out", str(tmp_path / "out"), "--tracking-uri", uri,
        "--replay-root", str(tmp_path / "none"),
    ])  # fmt: skip
    client = mlflow.MlflowClient(uri)
    (run,) = _analysis_runs(client, "export-tune-coarse")
    assert run.data.tags["analysis/phases"] == "tune-coarse"
    logged = {a.path for a in client.list_artifacts(run.info.run_id)}
    assert logged == {"runs.parquet", "curves.parquet", "eval_episodes.parquet"}


def test_cli_refuses_a_partial_export_into_the_phase_folder(
    store: Store, tmp_path: Path
) -> None:
    rid = _final(store)[0].run_id
    with pytest.raises(SystemExit):
        export.main([
            "--phase", "final-1m", "--tracking-uri", store.uri, "--no-log",
            "--run", rid,
        ])  # fmt: skip


def test_a_run_name_that_matches_no_run_of_the_phase_is_named(
    store: Store, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="nope"):
        export_phase(
            "final-1m", tmp_path / "out", tracking_uri=store.uri, runs=["nope"]
        )
    part = export_phase(
        "final-1m", tmp_path / "out", tracking_uri=store.uri,
        runs=[_final(store)[0].name],
    )  # fmt: skip
    assert len(pd.read_parquet(part["runs"])) == 1


# --- atomic per phase ----------------------------------------------------------------


def _old_export(store: Store, tmp_path: Path) -> tuple[Path, dict[str, bytes]]:
    out = tmp_path / "final-1m"
    export_phase("final-1m", out, tracking_uri=store.uri, runs=[_final(store)[0].name])
    return out, {p.name: p.read_bytes() for p in out.iterdir()}


def test_a_failure_mid_write_leaves_the_previous_export_intact(
    store: Store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out, before = _old_export(store, tmp_path)
    real = export.write_table

    def fail_on_episodes(table: Any, folder: Path, stem: str) -> None:
        if stem == "eval_episodes":
            msg = "no space left on device"
            raise OSError(msg)
        real(table, folder, stem)

    monkeypatch.setattr(export, "write_table", fail_on_episodes)
    with pytest.raises(OSError, match="no space"):
        export_phase("final-1m", out, tracking_uri=store.uri)
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before
    assert load_phase(out)["runs"]["run_id"].tolist() == [_final(store)[0].run_id]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["final-1m"]  # no leftovers


def test_an_export_left_aside_by_a_crash_between_the_moves_is_restored(
    store: Store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out, before = _old_export(store, tmp_path)
    out.rename(out.with_name("final-1m.previous"))  # as if killed between the moves

    def fail(*args: Any) -> None:
        msg = "no space left on device"
        raise OSError(msg)

    monkeypatch.setattr(export, "write_table", fail)
    with pytest.raises(OSError, match="no space"):
        export_phase("final-1m", out, tracking_uri=store.uri)
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["final-1m"]


def test_a_bad_run_fails_before_any_work(
    store: Store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out, before = _old_export(store, tmp_path)
    db = tmp_path / "bad.db"
    shutil.copy(store.path, db)
    with sqlite3.connect(db) as con:
        con.execute(
            "update tags set value = 'abcdef12' where key = 'config_hash' "
            "and run_uuid = ?",
            (_final(store)[-1].run_id,),
        )
    called = []
    monkeypatch.setattr(export, "run_curves", lambda *a: called.append(a))
    with pytest.raises(ValueError, match="abcdef12"):
        export_phase("final-1m", out, tracking_uri=f"sqlite:///{db}")
    assert not called
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before


def test_a_run_without_its_seed_tag_is_refused(store: Store, tmp_path: Path) -> None:
    db = tmp_path / "mlruns.db"
    shutil.copy(store.path, db)
    with sqlite3.connect(db) as con:
        con.execute(
            "delete from tags where key = 'seed' and run_uuid = ?",
            (_final(store)[0].run_id,),
        )
    with pytest.raises(ValueError, match="seed"):
        export_phase("tune-coarse", tmp_path / "out", tracking_uri=f"sqlite:///{db}")


def test_a_stored_record_without_a_non_replay_column_is_refused(
    store: Store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = export.stored_records

    class Short:
        def __init__(self, rec: Any) -> None:
            self.rec = rec

        def row(self) -> dict[str, Any]:
            return {k: v for k, v in self.rec.row().items() if k != "max_floor"}

    monkeypatch.setattr(
        export, "stored_records", lambda *a: [Short(r) for r in real(*a)]
    )
    with pytest.raises(ValueError, match="max_floor"):
        export_phase("tune-coarse", tmp_path / "out", tracking_uri=store.uri)


def test_the_store_is_read_only_during_the_export(store: Store, tmp_path: Path) -> None:
    before = store.path.stat().st_mtime_ns
    export_phase("tune-coarse", tmp_path / "out", tracking_uri=store.uri)
    assert store.path.stat().st_mtime_ns == before
    assert set(RUN_IDENTITY) < set(_episodes_of(tmp_path / "out").columns)


def test_a_writer_can_commit_between_the_runs_of_an_export(
    store: Store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The store is in rollback-journal mode: a read held across runs locks writers."""
    db = tmp_path / "mlruns.db"
    shutil.copy(store.path, db)
    commits = []
    real = export.run_episodes

    def write_then_go_on(*args: Any) -> Any:
        with sqlite3.connect(db, timeout=0) as con:  # fails at once if locked
            con.execute("update runs set status = status")
        commits.append(1)
        return real(*args)

    monkeypatch.setattr(export, "run_episodes", write_then_go_on)
    uri = f"sqlite:///{db}?timeout=300"
    export_phase("tune-coarse", tmp_path / "out", tracking_uri=uri)
    assert len(commits) == 3


def test_a_timeout_in_the_uri_reaches_the_mlflow_logging(
    store: Store, exported: Exported, tmp_path: Path
) -> None:
    db = tmp_path / "mlruns.db"
    shutil.copy(store.path, db)
    run_id = log_export(
        exported.out / "tune-coarse", "tune-coarse", f"sqlite:///{db}?timeout=300"
    )
    client = mlflow.MlflowClient(f"sqlite:///{db}")
    assert client.get_run(run_id).info.run_name == "export-tune-coarse"
    art = Path(client.get_run(run_id).info.artifact_uri.removeprefix("file://"))
    assert art.is_relative_to(tmp_path / "mlruns" / ANALYSIS_EXPERIMENT)


def _episodes_of(folder: Path) -> pd.DataFrame:
    return pd.read_parquet(folder / "eval_episodes.parquet")


def test_a_pruned_run_keeps_the_env_steps_of_every_eval_file(tmp_path: Path) -> None:
    """Storage retention: ``index.json`` stays whole when the checkpoints go."""
    run_dir = tmp_path / "run"
    _artifacts(run_dir, "ppo")
    mark_pruned(run_dir, list(UPDATES), "test")
    ref = RunRef("r" * 32, "ppo-1m-aaaaaaaa-s0", "ppo", "1m", 0, run_dir)
    tables = export.run_episodes(ref, "tuned", "craftax", tmp_path / "replay")
    steps = {s for t in tables.episodes for s in t["env_steps"].to_pylist()}
    assert steps == {u * SPU for u in UPDATES}
    assert len(tables.episodes) == 2 * len(UPDATES)
