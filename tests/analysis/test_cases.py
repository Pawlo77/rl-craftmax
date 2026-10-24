"""Tests for the case finder, the case export and the case facts.

`episode` builds eval-episode rows with chosen stalls, floors, unlocks and deaths for
`find_cases`; the export and fact tests replay the tiny Classic PPO run of the
visualisation tests (`_learning_run`), whose stored episodes are known:
update 4, stochastic, seed 10000 (76 steps) and 10001 (229 steps, wood at 25, wake up
at 116), both deaths.
"""

import json
import sqlite3
import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from _learning_run import needs_ffmpeg, tiny_run

from wut_murl_craftax.analysis.cases import (
    CASE_COLUMNS,
    GIF_SECONDS,
    OVERLAY_STEPS,
    describe_case,
    export_case,
    find_cases,
    run_dirs,
    write_cases,
)
from wut_murl_craftax.evaluation.episode_records import stored_records
from wut_murl_craftax.evaluation.milestones import action_names
from wut_murl_craftax.visualisation import ffmpeg
from wut_murl_craftax.visualisation.eval_rollouts import EvalRun, load_eval
from wut_murl_craftax.visualisation.rollout import Rollout, replay

CRAFTAX = "Craftax-Symbolic-v1"
NAMES = (
    "collect_wood", "place_table", "collect_diamond", "enter_dungeon",
    "enter_gnomish_mines",
)  # fmt: skip
"""Achievement columns of the built frames: two common, one rare, two floors."""
CONTRACT = (
    "id", "method", "config_ref", "budget", "run_id", "run_name", "update", "mode",
    "seed", "kind", "start", "end", "peak", "unlock_step", "unlocked", "speed",
    "game", "caption", "rollout", "master", "gif", "overlay", "video",
)  # fmt: skip
"""The `cases.csv` columns of contracts.md section 4."""
FINAL, EARLIER = 50, (10, 20, 30, 40)
"""Checkpoints of the built frames."""


def stall(start: int, end: int, ending: str = "unlock", by: str = "") -> dict:
    """A stored stall struct, no replay trace."""
    by = by or ("collect_diamond" if ending == "unlock" else "")
    return {
        "start": start,
        "end": end,
        "length": end - start,
        "floor": 0,
        "ending": ending,
        "ended_by": by,
    }


def episode(
    run: int = 0,
    update: int = FINAL,
    eval_seed: int = 10000,
    *,
    length: int = 2000,
    stalls: tuple[dict, ...] = (),
    unlocks: dict[str, int] | None = None,
    cause: str = "",
    max_floor: int = 0,
    mode: str = "stochastic",
    ref: str = "tuned",
) -> dict[str, Any]:
    """One eval-episode row of method ``ppo-rnn`` at 1M (a timeout unless ``cause``)."""
    steps = {"collect_wood": 10, "place_table": 40} | (unlocks or {})
    steps = {a: steps.get(a, -1) for a in NAMES}
    return {
        "run_id": f"{ref}-r{run}",
        "method": "ppo-rnn",
        "config_ref": ref,
        "budget": "1m",
        "seed": run,
        "update": update,
        "env_steps": update * 20_000,
        "mode": mode,
        "eval_seed": eval_seed,
        "length": length,
        "end_type": "death" if cause else "timeout",
        "cause_of_death": cause,
        "max_floor": max_floor,
        "n_achievements": sum(s >= 0 for s in steps.values()),
        **{f"ach/{a}": s >= 0 for a, s in steps.items()},
        **{f"unlock_step/{a}": np.int32(s) for a, s in steps.items()},
        "stalls": list(stalls),
        "n_stalls": len(stalls),
    }


def frames(rows: list[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    """``runs``, empty ``curves`` and ``eval_episodes`` of the rows' runs."""
    eps = pd.DataFrame(rows)
    runs = eps.drop_duplicates("run_id")[["run_id", "method", "config_ref", "budget"]]
    runs = runs.assign(
        run_name=lambda r: "name-" + r["run_id"],
        phases="final-1m",
        seed=eps.drop_duplicates("run_id")["seed"].to_numpy(),
        config_json=json.dumps({"env": CRAFTAX}),
    )
    curves = pd.DataFrame(columns=["run_id", "env_steps", "metric", "value"])
    return {"runs": runs, "curves": curves, "eval_episodes": eps}


def of_kind(cases: pd.DataFrame, kind: str) -> pd.DataFrame:
    return cases[cases["kind"] == kind].reset_index(drop=True)


# ----------------------------------------------------------------- find_cases


def test_the_longest_escaped_stall_ranks_first_then_four_more() -> None:
    long = stall(600, 1000)
    rows = [
        episode(0, stalls=(stall(0, 320), long)),
        episode(1, stalls=(stall(100, 450), stall(500, 1400, "death"))),
        episode(2, stalls=(stall(0, 310), stall(400, 730), stall(800, 1140))),
        episode(2, eval_seed=10001, stalls=(stall(0, 305),)),
    ]
    stalls = of_kind(find_cases(frames(rows), "ppo-rnn", "1m"), "stall")
    first = stalls.iloc[0]
    assert (first["run_id"], first["seed"], first["start"]) == ("tuned-r0", 10000, 600)
    assert (first["peak"], first["unlock_step"], first["unlocked"]) == (
        1000,
        1000,
        "collect_diamond",
    )
    assert first["end"] == 1000 + 20  # 20 steps after the unlock (VIS:72)
    assert first["id"] == "1m-ppo-rnn-stall-1"
    # the 900-step stall ended in a death: not escaped, never a case
    lengths = (stalls["peak"] - stalls["start"]).tolist()
    assert lengths == [400, 350, 340, 330, 320]


def test_the_longest_stall_of_three_earlier_checkpoints_follows() -> None:
    rows = [episode(0, stalls=(stall(0, 400),))]
    for k, u in enumerate(EARLIER):  # two stalls per earlier checkpoint
        rows.append(episode(0, u, stalls=(stall(0, 300 + k), stall(500, 900 + k))))
        rows.append(episode(1, u, stalls=(stall(0, 350),)))
    rows.append(episode(1, FINAL))
    stalls = of_kind(find_cases(frames(rows), "ppo-rnn", "1m"), "stall")
    assert stalls["update"].tolist() == [FINAL, 10, 30, 40]
    assert (stalls["peak"] - stalls["start"]).tolist() == [400, 400, 402, 403]
    assert stalls["id"].tolist() == [f"1m-ppo-rnn-stall-{n}" for n in range(1, 5)]


def test_deep_floor_cases_are_the_three_deepest_ties_by_achievements() -> None:
    deep = {"enter_dungeon": 300, "enter_gnomish_mines": 700}
    rows = [
        episode(0, max_floor=2, unlocks=deep),
        episode(1, max_floor=2, unlocks=deep | {"collect_diamond": 900}),
        episode(2, max_floor=1, unlocks={"enter_dungeon": 500}),
        episode(2, 10, max_floor=1, unlocks={"enter_dungeon": 200}),
        episode(0, eval_seed=10001, max_floor=0),
    ]
    deep_cases = of_kind(find_cases(frames(rows), "ppo-rnn", "1m"), "deep_floor")
    assert deep_cases["run_id"].tolist() == ["tuned-r1", "tuned-r0", "tuned-r2"]
    # the peak is the first step on the deepest floor
    assert deep_cases["peak"].tolist() == [700, 700, 500]
    assert (deep_cases["unlock_step"] == -1).all()
    assert (deep_cases["unlocked"] == "").all()
    first = deep_cases.iloc[0]
    assert first["start"] <= 700 <= first["end"]
    assert first["end"] - first["start"] + 1 <= GIF_SECONDS * 10  # 20 s at 1x


def test_rare_unlock_cases_cover_each_rare_achievement_once() -> None:
    rows = [
        episode(r, eval_seed=s, cause="", unlocks={"enter_dungeon": 30})
        for r in range(3)
        for s in range(10000, 10010)
    ]
    rows[3] = episode(0, eval_seed=10003, unlocks={"collect_diamond": 812})
    rows.append(episode(1, 10, unlocks={"collect_diamond": 400}))
    rare = of_kind(find_cases(frames(rows), "ppo-rnn", "1m"), "rare_unlock")
    # 1 of 30 final episodes: rare; the others are common or never unlocked
    assert len(rare) == 1
    case = rare.iloc[0]
    assert (case["run_id"], case["update"], case["seed"]) == ("tuned-r0", FINAL, 10003)
    assert case["peak"] == 812
    assert "collect diamond" in case["caption"]


def test_death_cases_are_the_median_episode_of_the_three_top_causes() -> None:
    lengths = {"zombie": (100, 300, 900), "food": (400, 500), "lava": (50,)}
    lengths |= {"arrow": (60,)}
    rows = [
        episode(k, eval_seed=10000 + i, length=n, cause=c)
        for k, (c, ns) in enumerate(lengths.items())
        for i, n in enumerate(ns)
    ]
    deaths = of_kind(find_cases(frames(rows), "ppo-rnn", "1m"), "death")
    assert [c.split(" dies of ")[1].split(" after")[0] for c in deaths["caption"]] == [
        "zombie",
        "food",
        "arrow",
    ]
    assert deaths["length"].tolist() == [300, 400, 60]
    assert (deaths["peak"] == deaths["length"]).all()
    assert (deaths["end"] == deaths["length"]).all()


def test_the_case_table_fits_the_contract() -> None:
    rows = [
        episode(0, stalls=(stall(0, 400),), cause="zombie", length=900),
        episode(1, max_floor=1, unlocks={"enter_dungeon": 600}),
        episode(2, mode="greedy", stalls=(stall(0, 800),)),
    ]
    cases = find_cases(frames(rows), "ppo-rnn", "1m")
    assert tuple(cases.columns[: len(CONTRACT)]) == CONTRACT
    assert tuple(cases.columns) == CASE_COLUMNS
    assert cases["id"].is_unique
    assert set(cases["mode"]) == {"stochastic"}  # greedy episodes are not searched
    assert set(cases["kind"]) == {"stall", "deep_floor", "death"}
    assert (cases["speed"] == np.where(cases["kind"] == "stall", 8.0, 1.0)).all()
    assert set(cases["game"]) == {"craftax"}
    assert cases["run_name"].str.startswith("name-").all()
    assert not cases["caption"].str.contains("\n").any()
    assert (cases[["rollout", "master", "gif", "overlay", "video"]] == "").all().all()
    stall_row = of_kind(cases, "stall").iloc[0]
    assert stall_row["caption"].startswith("ppo-rnn breaks a 400-step stall")
    assert (cases["start"] <= cases["peak"]).all()
    assert (cases["peak"] <= cases["end"]).all()


def test_cases_come_from_one_config_ref_tuned_by_default() -> None:
    rows = [
        episode(0, stalls=(stall(0, 400),)),
        episode(0, stalls=(stall(0, 900),), ref="simhash"),
    ]
    tuned = find_cases(frames(rows), "ppo-rnn", "1m")
    assert set(tuned["config_ref"]) == {"tuned"}
    assert of_kind(tuned, "stall")["start"].tolist() == [0]
    assert (of_kind(tuned, "stall")["peak"] == 400).all()
    ablation = find_cases(frames(rows), "ppo-rnn", "1m", config_ref="simhash")
    assert ablation["id"].iloc[0] == "1m-ppo-rnn-simhash-stall-1"
    assert "(simhash)" in ablation["caption"].iloc[0]


def test_a_unit_without_episodes_has_no_cases() -> None:
    cases = find_cases(frames([episode(0)]), "sil", "1m")
    assert cases.empty
    assert tuple(cases.columns) == CASE_COLUMNS


def test_finding_cases_imports_no_jax() -> None:
    code = (
        "import sys; from wut_murl_craftax.analysis import cases; "
        "print('mlflow' in sys.modules, 'jax' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.split() == ["False", "False"]


def test_write_cases_writes_the_budget_table_and_one_per_method(
    tmp_path: Path,
) -> None:
    rows = [episode(0, stalls=(stall(0, 400),), cause="zombie")]
    cases = find_cases(frames(rows), "ppo-rnn", "1m")
    other = cases.assign(method="sil", id=cases["id"].str.replace("ppo-rnn", "sil"))
    paths = write_cases(pd.concat([cases, other]), tmp_path)
    budget = tmp_path / "analysis" / "1m" / "cases.csv"
    assert paths == [
        budget,
        tmp_path / "analysis/1m/ppo-rnn/cases.csv",
        tmp_path / "analysis/1m/sil/cases.csv",
    ]
    table = pd.read_csv(budget, keep_default_na=False)
    assert tuple(table.columns) == CASE_COLUMNS
    assert len(table) == 2 * len(cases)
    mine = pd.read_csv(paths[1], keep_default_na=False)
    assert set(mine["method"]) == {"ppo-rnn"}


def test_run_dirs_read_artifact_locations_from_the_store(tmp_path: Path) -> None:
    db = tmp_path / "mlruns.db"
    with closing(sqlite3.connect(db)) as con:
        con.execute("create table runs (run_uuid text, artifact_uri text)")
        con.execute("insert into runs values ('a1', ?)", ((tmp_path / "x").as_uri(),))
        con.commit()
    assert run_dirs(["a1"], f"sqlite:///{db}") == {"a1": tmp_path / "x"}
    with pytest.raises(KeyError, match="b2"):
        run_dirs(["a1", "b2"], f"sqlite:///{db}")
    with pytest.raises(ValueError, match="sqlite"):
        run_dirs(["a1"], "http://localhost:5000")


# ------------------------------------------------------- export and description


def fixture_case(kind: str, seed: int, start: int, peak: int, end: int) -> dict:
    """A case of the tiny run's update 4, stochastic."""
    stall_case = kind == "stall"
    return {
        "id": f"1m-ppo-{kind}-1",
        "method": "ppo",
        "config_ref": "tuned",
        "budget": "1m",
        "run_id": "tiny",
        "run_name": "tiny",
        "update": 4,
        "mode": "stochastic",
        "seed": seed,
        "kind": kind,
        "start": start,
        "end": end,
        "peak": peak,
        "unlock_step": peak if stall_case else -1,
        "unlocked": "wake_up" if stall_case else "",
        "speed": 8.0 if stall_case else 1.0,
        "game": "classic",
        "caption": f"ppo {kind} after 0.0M steps, eval seed {seed}",
    }


@pytest.fixture(scope="module")
def run() -> EvalRun:
    return EvalRun.load(tiny_run())


@needs_ffmpeg
def test_an_exported_stall_replays_to_the_stored_episode(
    run: EvalRun, tmp_path: Path
) -> None:
    root = tmp_path / "outputs"
    out_dir = root / "analysis" / "1m" / "ppo"
    case = fixture_case("stall", 10001, 25, 116, 136)
    row = export_case(
        case,
        out_dir,
        run,
        root=root,
        overlay_kw={"game_px": 144, "panel": (200, 144), "fps": 5},
    )
    assert row["rollout"] == "visualisation/runs/case-1m-ppo-stall-1.npz"
    assert row["master"] == "analysis/1m/ppo/masters/case-1m-ppo-stall-1.mkv"
    assert row["gif"] == "analysis/1m/ppo/gifs/1m-ppo-stall-1.gif"
    assert row["overlay"] == "analysis/1m/ppo/overlays/1m-ppo-stall-1.mp4"
    assert (out_dir / "rollouts" / "case-1m-ppo-stall-1.npz").is_file()
    rollout = Rollout.load(root / row["rollout"])
    record = stored_records(load_eval(run.dir, 4), "classic")[1]
    np.testing.assert_array_equal(rollout.achievements[-1], record.unlock_steps >= 0)
    assert rollout.steps == record.length
    feats = np.concatenate([c.features for c in replay(rollout)])
    np.testing.assert_array_equal(feats, rollout.features)
    assert ffmpeg.count_frames(root / row["master"]) == record.length + 1
    # the overlay covers the last 100 steps before the unlock
    assert ffmpeg.count_frames(root / row["overlay"]) == OVERLAY_STEPS
    assert ffmpeg.duration(root / row["gif"]) <= GIF_SECONDS


@needs_ffmpeg
def test_a_gif_lasts_at_most_20_seconds_and_ends_at_the_window_end(
    run: EvalRun, tmp_path: Path
) -> None:
    case = fixture_case("death", 10001, 0, 229, 229)  # 230 frames = 23 s at 1x
    row = export_case(case, tmp_path / "out", run, root=tmp_path)
    gif = tmp_path / row["gif"]
    assert ffmpeg.duration(gif) <= GIF_SECONDS
    assert ffmpeg.count_frames(gif) == GIF_SECONDS * 10
    assert row["overlay"] == ""  # internals clips are for stall cases only


def test_describe_case_gives_the_facts_at_the_start_and_the_peak(
    run: EvalRun,
) -> None:
    # seed 10000: wood collected in action 12, the unlock shows on frame 13
    facts = describe_case(fixture_case("rare_unlock", 10000, 0, 13, 62), run)
    assert facts["floor"] == {"start": 0, "peak": 0}
    assert facts["inventory"]["start"]["wood"] == 0
    assert facts["inventory"]["peak"]["wood"] == 1
    assert facts["stats"]["start"] == {
        "health": 9.0,
        "food": 9.0,
        "drink": 9.0,
        "energy": 9.0,
    }
    ev = load_eval(run.dir, 4)
    actions = ev["actions"][0, : ev["length"][0]]
    names = action_names("classic")
    assert facts["last_actions"] == [names[a] for a in actions[:13]]
    assert sum(facts["actions"]["window"].values()) <= 13
    assert sum(facts["actions"]["episode"].values()) <= len(actions)
    damage = ev["events/combat_damage"][0, :13].sum(dtype=np.float64)
    assert facts["damage_taken"] == pytest.approx(damage)
    assert facts["close_calls"] == 0
    gained = facts["coverage_gained"]
    assert set(gained) == {"overworld"}
    assert 0 < gained["overworld"] < 1


def test_describe_case_counts_the_window_of_a_stall_only(run: EvalRun) -> None:
    facts = describe_case(fixture_case("stall", 10001, 25, 116, 136), run)
    ev = load_eval(run.dir, 4)
    actions = ev["actions"][1, : ev["length"][1]]
    forced = ev["events/forced"][1, : ev["length"][1]]
    names = action_names("classic")
    window = np.bincount(actions[25:116][~forced[25:116]], minlength=len(names))
    assert facts["actions"]["window"] == {
        names[a]: int(k) for a, k in enumerate(window) if k
    }
    assert len(facts["last_actions"]) == 50
    assert facts["last_actions"] == [names[a] for a in actions[66:116]]
    assert facts["min_health"] <= 9.0


def fixture_frames(run_dir: Path) -> dict[str, pd.DataFrame]:
    """The tiny run's eval files as the export tables of one ``ppo`` run."""
    opened = EvalRun.load(run_dir)
    rows = []
    for update in opened.updates:
        for mode in ("stochastic", "greedy"):
            ev = load_eval(run_dir, update, mode == "greedy")
            records = stored_records(ev, "classic")
            for seed, rec in zip(ev["seeds"], records, strict=True):
                rows.append({
                    "run_id": "tiny",
                    "method": "ppo",
                    "config_ref": "tuned",
                    "budget": "1m",
                    "seed": 0,
                    "update": update,
                    "env_steps": opened.env_steps(update),
                    "mode": mode,
                    "eval_seed": int(seed),
                    **rec.row(),
                })
    runs = pd.DataFrame([
        {
            "run_id": "tiny",
            "run_name": "ppo-1m-tiny",
            "method": "ppo",
            "config_ref": "tuned",
            "budget": "1m",
            "phases": "final-1m",
            "config_json": json.dumps(opened.cfg),
        }
    ])
    curves = pd.DataFrame(columns=["run_id", "env_steps", "metric", "value"])
    return {"runs": runs, "curves": curves, "eval_episodes": pd.DataFrame(rows)}


@needs_ffmpeg
def test_a_case_found_on_a_run_exports_with_its_files(tmp_path: Path) -> None:
    run_dir = tiny_run()
    t0 = time.perf_counter()
    cases = find_cases(fixture_frames(run_dir), "ppo", "1m")
    found = time.perf_counter() - t0
    assert set(cases["game"]) == {"classic"}
    assert {"deep_floor", "death"} <= set(cases["kind"])
    case = cases.iloc[0]
    t0 = time.perf_counter()
    row = export_case(
        case, tmp_path / "analysis" / "1m" / "ppo", run_dir, root=tmp_path
    )
    exported = time.perf_counter() - t0
    print(f"find {found:.2f} s, export {exported:.2f} s")
    for key in ("rollout", "master", "gif"):
        assert (tmp_path / row[key]).is_file()
    assert row["id"] == case["id"]
