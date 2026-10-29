"""Tests for the audit of ``enchant`` choices in stored eval episodes."""

import shutil
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import mlflow
import numpy as np
import pandas as pd
import pytest
from craftax.craftax_env import make_craftax_env_from_name

from wut_murl_craftax.evaluation.rollout_metrics import RunRef
from wut_murl_craftax.evaluation.skill_audit import (
    AUDIT_COLUMNS,
    EnchantCounts,
    audit_file,
    audit_runs,
    enchant_counts,
    main,
)
from wut_murl_craftax.training.skills import SKILL_IDS
from wut_murl_craftax.visualisation.eval_rollouts import EvalRun, load_eval

ENCHANT, FIGHT = SKILL_IDS["enchant"], SKILL_IDS["fight"]
UPDATE = 2


@pytest.fixture(scope="module")
def vault_obs(craftax_start: Any) -> dict[str, np.ndarray]:
    """Observations at a fire table on floor 4 with a sword: plain, and already fire."""
    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=False)
    s = craftax_start.replace(player_level=jnp.int32(4), player_mana=jnp.int32(9))
    s = s.replace(inventory=s.inventory.replace(sword=jnp.int32(1), ruby=jnp.int32(1)))
    plain, fire = s, s.replace(sword_enchantment=jnp.int32(1))
    get = jax.jit(env.get_obs)
    return {"plain": np.asarray(get(plain)), "fire": np.asarray(get(fire))}


def _episode(
    vault_obs: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """10 steps: ``enchant`` decided at 0 (nothing to enchant), 5 (a plain sword)
    and 8 (nothing to enchant); ``fight`` at 3."""
    where = {0: "fire", 5: "plain", 8: "fire"}
    obs = np.stack([vault_obs[where.get(t, "plain")] for t in range(10)])
    skill = np.array([ENCHANT] * 3 + [FIGHT] * 2 + [ENCHANT] * 3 + [ENCHANT] * 2)
    age = np.array([0, 1, 2, 0, 1, 0, 1, 2, 0, 1])
    return obs, skill, age


def test_enchant_counts_the_decisions_with_nothing_to_enchant(
    vault_obs: dict[str, np.ndarray],
) -> None:
    obs, skill, age = _episode(vault_obs)
    got = enchant_counts(obs, skill, age, 10)
    # decisions at 0, 5 and 8; the ones at 0 and 8 hold a fire sword on the fire floor,
    # and run 3 steps (to the decision at 3) and 2 (to the episode end)
    assert got == EnchantCounts(enchant=3, redundant=2, steps=5)


def test_enchant_counts_stop_at_the_episode_length(
    vault_obs: dict[str, np.ndarray],
) -> None:
    obs, skill, age = _episode(vault_obs)
    assert enchant_counts(obs, skill, age, 9) == EnchantCounts(3, 2, 4)
    assert enchant_counts(obs, skill, age, 8) == EnchantCounts(2, 1, 3)


def test_no_enchant_decision_is_all_zero(vault_obs: dict[str, np.ndarray]) -> None:
    obs, _, age = _episode(vault_obs)
    assert enchant_counts(obs, np.full(10, FIGHT), age, 10) == EnchantCounts(0, 0, 0)


@pytest.fixture(scope="module")
def run(skill_run: Path) -> EvalRun:
    return EvalRun.load(skill_run)


def test_audit_file_counts_decisions_and_replays_only_choosing_episodes(
    run: EvalRun, skill_run: Path
) -> None:
    arrays = load_eval(skill_run, UPDATE)
    head = dict.fromkeys(AUDIT_COLUMNS[:6], 0)
    plain = audit_file(run, arrays, head)
    assert plain is not None
    starts = arrays["extras/steps_in_skill"] == 0
    n = arrays["length"]
    want = sum(int(starts[i, : n[i]].sum()) for i in range(len(n)))
    assert (plain["episodes"], plain["decisions"]) == (len(n), want)
    assert (plain["enchant"], plain["redundant"], plain["redundant_steps"]) == (0, 0, 0)
    skill = arrays["extras/skill"].copy()
    skill[0, 0] = ENCHANT  # a (Classic) run never offers it; count what is stored
    picked = audit_file(run, arrays | {"extras/skill": skill}, head)
    assert picked is not None
    assert (picked["enchant"], picked["redundant"]) == (1, 0)  # Classic holds no gear
    assert audit_file(run, {"actions": arrays["actions"]}, head) is None


def test_audit_runs_lists_every_file_with_its_identity(
    skill_run: Path,
) -> None:
    ref = RunRef("rid", "name", "smdp-skills", "1m", 3, skill_run)
    table = audit_runs([ref], "all", ("stochastic", "greedy"))
    assert tuple(table.columns) == AUDIT_COLUMNS
    assert len(table) == 4  # checkpoints 0 and 2, both modes
    assert (table["run_id"] == "rid").all()
    assert sorted(table["mode"].unique()) == ["greedy", "stochastic"]
    assert (table["decisions"] > 0).all()
    assert (table[["enchant", "redundant", "redundant_steps"]] == 0).all(axis=None)


def test_the_cli_writes_the_csv(skill_run: Path, tmp_path: Path) -> None:
    uri = f"sqlite:///{tmp_path / 'mlruns.db'}"
    client = mlflow.MlflowClient(uri)
    exp = client.create_experiment("e", artifact_location=(tmp_path / "e").as_uri())
    rid = client.create_run(exp, run_name="r").info.run_id
    for k, v in {"phases": "final-1m", "method": "smdp-skills", "budget": "1m"}.items():
        client.set_tag(rid, k, v)
    client.set_tag(rid, "seed", "3")
    shutil.copytree(
        skill_run, Path(client.get_run(rid).info.artifact_uri.removeprefix("file://"))
    )
    client.set_terminated(rid)
    out = tmp_path / "audit" / "enchant.csv"
    main(["--phase", "final-1m", "--tracking-uri", uri, "--updates", "final",
          "--out", str(out)])  # fmt: skip
    got = pd.read_csv(out)
    assert tuple(got.columns) == AUDIT_COLUMNS
    assert got["update"].unique().tolist() == [UPDATE]
    assert len(got) == 2
