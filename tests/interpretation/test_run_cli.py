"""Tests for the distillation CLI: files per policy and method, complexity, logging."""

import json
import math
import pickle
import shutil
from pathlib import Path
from typing import Any

import jax
import mlflow
import numpy as np
import pandas as pd
import pytest
from _toys import CLASSIC

from wut_murl_craftax.analysis.load import (
    CURVES_COLUMNS,
    EPISODE_REQUIRED,
    RUNS_COLUMNS,
)
from wut_murl_craftax.interpretation import FittedTree
from wut_murl_craftax.interpretation import run as cli
from wut_murl_craftax.orchestration.tracking import artifact_path
from wut_murl_craftax.training.checkpoints import save_checkpoint
from wut_murl_craftax.training.interface import METHODS
from wut_murl_craftax.training.smdp import GREEDY_TABLE_FILE, field_names

SMALL = {"env": CLASSIC, "envs": 16, "num_steps": 8, "minibatches": 2}
CONFIGS: dict[str, dict[str, Any]] = {
    "ppo": {"width": 32},
    "smdp-skills": {"hidden": 16, "epochs": 1, "log2_rows": 12, "extra_obs_dim": 26},
}
"""The two methods of the fixture: feed-forward and two-level (recurrent skills)."""
UPDATES = (0, 1)
"""A 2-update run: the final checkpoint is 1."""
TOP_SKILLS = ("collect_wood", "place_table", "collect_stone")
CONTRACT = [
    "budget", "method", "config_ref", "variant", "n_seeds",
    "l80_median", "l90_median", "l95_median", "l90_lo", "l90_hi", "l90_spread",
    "depth_at_l90_median", "fidelity_16", "fidelity_64", "fidelity_256",
    "expected_fidelity_256", "closed_loop_256", "spearman_score_l90", "spearman_p",
    "top_keys", "top_skills", "top_l90",
]  # fmt: skip
"""`complexity.csv` columns of contracts.md section 3, in order."""
METHOD_FILES = {
    "curve.csv",
    "complexity.json",
    "rules_16.txt",
    "importance.csv",
    "closed_loop.csv",
    "fidelity_vs_leaves.png",
}
"""Files of every method folder (plan Task 6 Step 1)."""
SKILL_FILES = {"top_level_rules.txt", "top_level.json"}


def _greedy_table(update: int) -> pd.DataFrame:
    """Keys of one checkpoint whose greedy skill depends on wood; one optimistic key."""
    names = field_names(16)
    wood = np.arange(12) % 6
    skill = np.array(TOP_SKILLS, object)[np.minimum(wood // 2, 2)]
    tried = skill.copy()
    tried[0] = TOP_SKILLS[2]  # the default argmax is an untried optimistic entry
    frame = pd.DataFrame(0, index=range(12), columns=names, dtype=np.int8)
    frame["wood"] = wood
    frame["pickaxe"] = update  # an earlier checkpoint must not be read
    return pd.DataFrame({
        "update": np.int32(update),
        "key": np.arange(12, dtype=np.int32) + 100 * update,
        **{c: frame[c] for c in names},
        "greedy_skill": skill,
        "value": np.linspace(1, 2, 12, dtype=np.float32),
        "tried": np.int32(1),
        "greedy_tried_skill": tried,
        "greedy_tried_value": np.linspace(0, 1, 12, dtype=np.float32),
    })


def _write_run(root: Path, method: str) -> None:
    """A 2-update Classic run: config, both checkpoints and, for skills, its table."""
    root.mkdir(parents=True, exist_ok=True)
    cfg = {"method": method, **SMALL, **CONFIGS[method]}
    (root / "config.json").write_text(json.dumps(cfg))
    trainer = METHODS[method]
    for update in UPDATES:
        key = jax.random.key(update)  # other params at each checkpoint
        state = trainer.init(cfg | {"steps": trainer.steps_per_update(cfg)}, key)
        save_checkpoint(
            root, update, 128 * update, float(update), trainer.params(state)
        )
    if method == "smdp-skills":
        (root / "eval").mkdir(exist_ok=True)
        table = pd.concat([_greedy_table(u) for u in UPDATES], ignore_index=True)
        table.to_parquet(root / "eval" / GREEDY_TABLE_FILE)


def _export(metrics: Path, rows: list[dict[str, Any]]) -> None:
    """A ``final-1m`` export of ``rows`` with the contract columns (empty curves)."""
    folder = metrics / "final-1m"
    folder.mkdir(parents=True)
    runs = pd.DataFrame([dict.fromkeys(RUNS_COLUMNS, np.nan) | r for r in rows])
    runs.to_parquet(folder / "runs.parquet")
    pd.DataFrame(columns=CURVES_COLUMNS).to_parquet(folder / "curves.parquet")
    pd.DataFrame(columns=EPISODE_REQUIRED).to_parquet(folder / "eval_episodes.parquet")


class World:
    """One CLI run over a temporary store and export; paths and the run ids."""

    def __init__(self, root: Path) -> None:
        self.store = f"sqlite:///{root / 'store' / 'mlruns.db'}?timeout=300"
        self.out, self.figures = root / "interpretation", root / "figures"
        self.metrics = root / "metrics"
        client = mlflow.MlflowClient(self.store)
        exp = client.create_experiment("final-1m", (root / "artifacts").as_uri())
        self.ids: dict[str, str] = {}
        rows = []
        for i, method in enumerate(CONFIGS):
            run = client.create_run(exp)
            self.ids[method] = run.info.run_id
            _write_run(artifact_path(run.info.artifact_uri), method)
            rows.append(self._row(run.info.run_id, method, "tuned", 1.0 + i))
        # an ablation of ppo: never distilled, so it needs no run folder
        rows.append(self._row("0" * 32, "ppo", "wide", 3.0))
        _export(self.metrics, rows)

    @staticmethod
    def _row(run_id: str, method: str, ref: str, score: float) -> dict[str, Any]:
        return {
            "run_id": run_id, "run_name": f"{method}-{ref}", "phase": "final-1m",
            "phases": "final-1m", "method": method, "budget": "1m", "seed": 0,
            "config_ref": ref, "final_score": score, "status": "FINISHED",
        }  # fmt: skip

    def main(self, *extra: str) -> None:
        cli.main([
            "--phases", "final-1m", "combo-1m", "--skip-missing",
            "--method", "all", "--out", str(self.out),
            "--metrics-root", str(self.metrics), "--tracking-uri", self.store,
            "--figures", str(self.figures), "--max-seeds", "2", "--workers", "2",
            *extra,
        ])  # fmt: skip

    def method(self, method: str) -> Path:
        return self.out / "1m" / method

    def policy(self, method: str) -> Path:
        return self.method(method) / self.ids[method]


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> World:
    """Both methods distilled once, with a DAgger round, at 2 seeds per split."""
    w = World(tmp_path_factory.mktemp("distill"))
    w.main("--dagger")
    return w


def test_every_method_folder_holds_the_plan_files(world: World) -> None:
    for method in CONFIGS:
        names = {p.name for p in world.method(method).iterdir()}
        assert names >= METHOD_FILES, method
    assert {p.name for p in world.method("smdp-skills").iterdir()} >= SKILL_FILES
    assert not SKILL_FILES & {p.name for p in world.method("ppo").iterdir()}
    assert {p.name for p in (world.out / "1m").iterdir()} == {
        "ppo", "smdp-skills", "complexity.csv", "policies.csv",
        "fidelity_vs_leaves_1m.png",
    }  # fmt: skip
    assert (world.figures / "fidelity_vs_leaves_1m.png").is_file()


def test_only_tuned_runs_are_distilled_at_their_final_checkpoint(world: World) -> None:
    for method in CONFIGS:
        summary = json.loads((world.method(method) / "complexity.json").read_text())
        assert [s["run_id"] for s in summary["seeds"]] == [world.ids[method]]
        assert summary["seeds"][0]["update"] == UPDATES[-1]
        assert summary["median_run"] == world.ids[method]
    table = pd.read_csv(world.out / "1m" / "complexity.csv")
    assert set(table["config_ref"]) == {"tuned"}  # the ppo ablation stays out


def test_a_recurrent_policy_is_fitted_with_and_without_memory_proxies(
    world: World,
) -> None:
    ppo = pd.read_csv(world.method("ppo") / "curve.csv")
    skills = pd.read_csv(world.method("smdp-skills") / "curve.csv")
    grid = 21  # 12 leaf sizes and 9 depths (R99)
    base = skills[skills["variant"] == "base"]
    assert len(base) == grid
    assert base["memory"].all()
    assert base["skill"].all()
    plain = skills[skills["variant"] == "no-memory"]
    assert len(plain) == grid
    assert not plain["memory"].any()
    assert plain["skill"].all()
    assert set(ppo["variant"]) == {"base", "dagger"}  # feed-forward: one fit
    assert not ppo["memory"].any()
    assert (ppo.groupby("variant").size() == grid).all()
    assert {"run_id", "seed", "kind", "size", "top1", "expected"} <= set(ppo.columns)


def test_only_the_16_and_256_leaf_trees_are_kept(world: World) -> None:
    kept = sorted(p.name for p in world.policy("smdp-skills").glob("*.pkl"))
    assert kept == [
        f"tree_{n}_{v}.pkl" for n in (16, 256) for v in ("base", "dagger", "no-memory")
    ]
    with (world.policy("ppo") / "tree_256_base.pkl").open("rb") as f:
        tree = pickle.load(f)  # written by the CLI under test
    assert isinstance(tree, FittedTree)
    assert (tree.kind, tree.size) == ("leaves", 256)


def test_rules_are_the_16_leaf_tree_of_the_median_seed(world: World) -> None:
    text = (world.method("ppo") / "rules_16.txt").read_text()
    assert world.ids["ppo"] in text
    base = text.split("## base")[1].split("## ")[0]
    lines = [line for line in base.splitlines() if line[:1].isdigit()]
    assert 1 <= len(lines) <= 16
    assert all(" THEN " in line or "ALWAYS" in line for line in lines)
    importance = pd.read_csv(world.method("ppo") / "importance.csv")
    assert {"run_id", "variant", "feature", "group", "importance"} <= set(
        importance.columns
    )


def test_closed_loop_plays_the_256_leaf_tree_against_the_network(world: World) -> None:
    loop = pd.read_csv(world.method("smdp-skills") / "closed_loop.csv")
    assert set(loop["variant"]) == {"base", "no-memory", "dagger"}
    assert (loop.groupby("variant").size() == 2).all()  # 2 test seeds each
    net = loop.pivot(index="eval_seed", columns="variant", values="network")
    np.testing.assert_array_equal(net["base"], net["no-memory"])  # one network run


def test_the_top_level_reports_the_tried_view_of_the_final_checkpoint(
    world: World,
) -> None:
    top = json.loads((world.method("smdp-skills") / "top_level.json").read_text())
    seed = top["seeds"][0]
    assert seed["run_id"] == world.ids["smdp-skills"]
    assert seed["main"] == "tried"
    tried, default = seed["tried"], seed["default"]
    assert tried["update"] == default["update"] == UPDATES[-1]
    assert tried["n_rules"] == 12
    assert set(tried["skills"]) == set(TOP_SKILLS)
    assert tried["optimistic_share"] == pytest.approx(1 / 12)
    text = (world.method("smdp-skills") / "top_level_rules.txt").read_text()
    assert text.index("tried") < text.index("default")
    assert "optimistic" in text


def test_complexity_csv_has_the_contract_columns_and_a_row_per_variant(
    world: World,
) -> None:
    table = pd.read_csv(world.out / "1m" / "complexity.csv")
    assert list(table.columns[: len(CONTRACT)]) == CONTRACT
    rows = set(zip(table["method"], table["variant"], strict=True))
    assert rows == {
        ("ppo", "base"), ("ppo", "no-memory"), ("ppo", "dagger"),
        ("smdp-skills", "base"), ("smdp-skills", "no-memory"),
        ("smdp-skills", "dagger"),
    }  # fmt: skip
    assert (table["n_seeds"] == 1).all()
    assert (table["l90_spread"] == "iqr").all()
    ppo = table[table["method"] == "ppo"].set_index("variant")
    # a feed-forward policy has no memory proxies: its no-memory fit is its base fit
    pd.testing.assert_series_equal(
        ppo.loc["base", "l80_median":"closed_loop_256"],
        ppo.loc["no-memory", "l80_median":"closed_loop_256"],
        check_names=False,
    )
    assert ppo["top_keys"].isna().all()
    skills = table[table["method"] == "smdp-skills"].iloc[0]
    assert (skills["top_keys"], skills["top_skills"]) == (12, 3)
    assert skills["top_optimistic_share"] == pytest.approx(1 / 12)


def test_each_method_is_logged_without_its_datasets(world: World) -> None:
    client = mlflow.MlflowClient(world.store)
    exp = client.get_experiment_by_name("analysis")
    assert exp is not None
    runs = {r.info.run_name: r for r in client.search_runs([exp.experiment_id])}
    assert set(runs) == {"distill-1m-ppo", "distill-1m-smdp-skills"}
    run = runs["distill-1m-smdp-skills"]
    tags = run.data.tags
    assert tags["analysis/kind"] == "distill"
    assert (tags["analysis/budget"], tags["analysis/method"]) == ("1m", "smdp-skills")
    assert tags["analysis/phases"] == "final-1m"  # combo-1m has no export
    assert not {"phases", "method", "budget", "seed", "config_hash"} & set(tags)
    root = artifact_path(run.info.artifact_uri)
    logged = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    assert logged >= METHOD_FILES | SKILL_FILES
    assert f"{world.ids['smdp-skills']}/tree_16_base.pkl" in logged
    assert not [p for p in logged if p.endswith(".npz")]


def test_a_rerun_skips_finished_policies_and_force_redoes_them(world: World) -> None:
    done = world.policy("ppo") / "complexity.json"
    before = done.stat().st_mtime_ns
    world.main("--dagger", "--no-log")
    assert done.stat().st_mtime_ns == before
    world.main("--method", "ppo", "--force", "--no-log")
    assert done.stat().st_mtime_ns != before
    assert not (world.policy("ppo") / "fit_dagger.json").exists()  # no --dagger now


def test_censored_levels_rank_above_the_grid_and_set_the_median() -> None:
    inf = math.inf
    assert cli.censored_median([None] * 6 + [16] * 4) == inf  # more than half
    assert cli.censored_median([None] * 5 + [16] * 5) == 16
    assert cli.censored_median([4, 64, None]) == 64
    assert math.isnan(cli.censored_median([]))
    assert cli.level_text(inf) == "> 4096"
    assert cli.level_text(64.0) == "64"
    rho, _ = cli.spearman([1.0, 2.0, 3.0, 4.0], [16, 64, 4096, None])
    assert rho == pytest.approx(cli.spearman([1, 2, 3, 4], [1, 2, 3, 4])[0])
    assert math.isnan(cli.spearman([1.0, 2.0], [4, 8])[0])  # too few seeds
    # one or two uncensored levels are no statistic, whatever the seed count
    assert math.isnan(cli.spearman([1.0, 2.0, 3.0, 4.0], [16, 64, None, None])[0])


def test_the_l90_depth_needs_three_uncensored_seeds() -> None:
    assert math.isnan(cli.uncensored_median([1.0]))
    assert math.isnan(cli.uncensored_median([1.0, 3.0]))
    assert cli.uncensored_median([1.0, 3.0, 8.0]) == 3.0


# --- R106 numbers --------------------------------------------------------------


def test_the_r106_columns_are_computed_from_the_policy_files(world: World) -> None:
    table = pd.read_csv(world.out / "1m" / "complexity.csv")
    assert list(table.columns[len(CONTRACT) + 2 :]) == [
        "majority", "gain_16", "gain_64", "gain_256", "peak_val_leaves",
        *(f"group_{g}" for g in cli.ACTION_GROUPS),
        "spearman_score_fid256", "spearman_fid256_p",
    ]  # fmt: skip
    for method in CONFIGS:
        with np.load(world.policy(method) / "test.npz") as z:
            label = z["label"]
        share = np.bincount(label).max() / len(label)
        rows = table[table["method"] == method]
        assert rows["majority"].eq(share).all()
        base = rows[rows["variant"] == "base"].iloc[0]
        assert base["gain_256"] == pytest.approx(base["fidelity_256"] - share)
        assert base["peak_val_leaves"] in cli.LEAF_GRID
        groups = rows[[f"group_{g}" for g in cli.ACTION_GROUPS]]
        assert ((groups >= 0) | groups.isna()).all().all()
        assert groups.notna().any().any()  # the network played some group
    assert table["spearman_score_fid256"].isna().all()  # one seed per method


def test_policies_csv_has_a_row_per_policy_and_variant(world: World) -> None:
    pol = pd.read_csv(world.out / "1m" / "policies.csv")
    # base and no-memory for each policy; dagger too unless a test redid it without
    assert not pol.duplicated(["run_id", "variant"]).any()
    for run_id, g in pol.groupby("run_id"):
        assert {"base", "no-memory"} <= set(g["variant"]), run_id
    assert {"run_id", "seed", "score", "majority", "peak_val_leaves"} <= set(
        pol.columns
    )
    assert set(pol["run_id"]) == set(world.ids.values())


def test_a_torn_curve_file_does_not_stop_the_budget_summary(
    world: World, tmp_path: Path
) -> None:
    copy = tmp_path / "1m"
    shutil.copytree(world.out / "1m", copy)
    (copy / "ppo" / "curve.csv").write_text("")  # a writer caught mid-rewrite
    cli.sum_up_budget(copy, tmp_path / "figures")
    assert (copy / "complexity.csv").is_file()
    assert not list(tmp_path.rglob("*.tmp"))


def _summary(run_id: str, score: float, fid: float) -> dict[str, Any]:
    fit = {
        "L80": 8, "L90": 16, "L95": None, "depth_L90": 3, "fidelity_16": fid - 0.2,
        "fidelity_64": fid - 0.1, "fidelity_256": fid, "expected_256": fid,
        "closed_loop_256": 0.5,
    }  # fmt: skip
    return {
        "run_id": run_id, "seed": 0, "method": "ppo", "config_ref": "tuned",
        "score": score, "recurrent": False, "fits": {"base": fit}, "top_level": None,
    }  # fmt: skip


def test_the_spearman_of_score_against_fidelity_replaces_the_censored_one() -> None:
    summaries = [_summary(f"r{i}", 1.0 + i, 0.5 + 0.1 * i) for i in range(4)]
    extras = {
        (s["run_id"], "base"): {"majority": 0.25, "peak_val_leaves": 64.0}
        for s in summaries
    }
    row = cli.complexity_table(summaries, "1m", extras).set_index("variant").loc["base"]
    assert row["spearman_score_fid256"] == pytest.approx(1.0)
    assert row["majority"] == 0.25
    assert row["gain_256"] == pytest.approx(np.median([0.25, 0.35, 0.45, 0.55]))
    assert row["peak_val_leaves"] == 64
    assert math.isnan(row["group_move"])  # no failure file for it
    bare = cli.complexity_table(summaries, "1m").set_index("variant").loc["base"]
    assert math.isnan(bare["majority"])
    assert math.isnan(bare["gain_256"])


def test_policy_extras_read_labels_the_validation_peak_and_group_recall(
    tmp_path: Path,
) -> None:
    np.savez(tmp_path / "test.npz", label=np.array([0, 0, 0, 1]))
    pd.DataFrame({
        "kind": ["leaves"] * 3,
        "leaves": [2, 4, 8],
        "val_top1": [0.5, 0.7, 0.7],
    }).to_csv(tmp_path / "curve_base.csv", index=False)
    groups = list(cli.ACTION_GROUPS)
    confusion = np.zeros((len(groups), len(groups)))
    confusion[1, 1], confusion[1, 2] = 3, 1  # network "move" (index 1): 3 of 4 right
    (tmp_path / "failure_base.json").write_text(
        json.dumps({"groups": groups, "confusion": confusion.tolist()})
    )
    got = cli.policy_extras(tmp_path, "base")
    assert got["majority"] == 0.75
    assert got["peak_val_leaves"] == 4  # the smaller tree on a tie
    assert got["group_move"] == 0.75
    assert math.isnan(got["group_noop"])  # the network never chose it
