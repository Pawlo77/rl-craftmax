"""Tests for the shipped PPO paper configurations and the published reference scores."""

import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from wut_murl_craftax.orchestration.queue import (
    QueueConfigError,
    config_hash,
    expand_fine,
    expand_phase,
    load_method,
    resolve_config,
)

REPO = Path(__file__).resolve().parents[2]
METHODS = REPO / "configs" / "methods"
TABLE_11 = {
    "env": "Craftax-Symbolic-v1",
    "envs": 256,
    "num_steps": 16,
    "update_epochs": 4,
    "minibatches": 8,
    "lr": 3.0e-4,
    "anneal_lr": True,
    "gamma": 0.99,
    "gae_lambda": 0.8,
    "clip": 0.2,
    "vf": 0.5,
    "ent": 0.01,
    "max_grad_norm": 1.0,
    "width": 512,
    "resets": "optimistic",
    "reset_ratio": 16,
}
TABLE_7 = TABLE_11 | {"envs": 1024, "num_steps": 64, "lr": 2.0e-4}
STAGE_1 = {
    "published-1m": {},
    "shape-1024x64": {"envs": 1024, "num_steps": 64, "lr": 2.0e-4},
    "lr-2e-4": {"lr": 2.0e-4},
    "lr-4e-4": {"lr": 4.0e-4},
    "lambda-0.7": {"gae_lambda": 0.7},
    "lambda-0.9": {"gae_lambda": 0.9},
    "ent-0": {"ent": 0.0},
}


def ppo() -> Any:
    return load_method("ppo", METHODS)


def test_base_is_table_11_field_by_field() -> None:
    assert ppo().base == TABLE_11
    assert resolve_config(ppo(), "ppo", "1m", methods_dir=METHODS) == TABLE_11 | {
        "method": "ppo"
    }


def test_1b_budget_override_is_table_7() -> None:
    assert ppo().budgets == {"1b": {"envs": 1024, "num_steps": 64, "lr": 2.0e-4}}
    cfg = resolve_config(ppo(), "ppo", "1b", methods_dir=METHODS)
    assert cfg == TABLE_7 | {"method": "ppo"}


def test_coarse_grid_is_seven_one_knob_entries() -> None:
    coarse = {
        e["name"]: {k: v for k, v in e.items() if k not in ("name", "seeds")}
        for e in ppo().coarse
    }
    assert coarse == STAGE_1
    assert [e.get("seeds") for e in ppo().coarse if e["name"] == "published-1m"] == [10]
    assert [e["name"] for e in ppo().coarse if "seeds" in e] == ["published-1m"]
    for name, over in STAGE_1.items():
        cfg = resolve_config(ppo(), "ppo", "1m", ref=name, methods_dir=METHODS)
        diff = {k for k in cfg if cfg[k] != (TABLE_11 | {"method": "ppo"})[k]}
        assert diff <= set(over), name


def test_fine_holds_the_rule_and_knobs_and_no_entries() -> None:
    fine = ppo().fine
    assert fine is not None
    assert set(fine) == {"rule", "knobs"}
    assert fine["rule"] == {"select": 2, "combine": "each"}
    knobs = {k["key"]: k for k in fine["knobs"]}
    assert set(knobs) == {"lr", "gae_lambda", "ent"}
    assert knobs["lr"]["mul"] == [0.75, 1.5]
    assert knobs["gae_lambda"]["add"] == [-0.05, 0.05]
    assert (knobs["ent"]["mul"], knobs["ent"]["zero"]) == ([0.5, 2.0], [0.001, 0.003])
    assert "envs" not in knobs
    assert "num_steps" not in knobs


def test_fine_rule_yields_four_configs_from_two_knobs() -> None:
    scores = {}
    for e in ppo().coarse:
        cfg = resolve_config(ppo(), "ppo", "1m", ref=e["name"], methods_dir=METHODS)
        scores[config_hash(cfg)] = {"lr-4e-4": 5.0, "lambda-0.9": 4.0}.get(
            e["name"], 3.0
        )
    fine = expand_fine("ppo", scores, methods_dir=METHODS)["ppo"]
    assert len(fine) == 4
    assert set(fine) == {
        "fine-lr=0.0003",
        "fine-lr=0.0006",
        "fine-gae_lambda=0.75",
        "fine-gae_lambda=0.85",
    }
    assert fine["fine-lr=0.0006"] == {"lr": 6.0e-4}


def test_published_1m_runs_ten_seeds_and_smoke_keeps_one(tmp_path: Path) -> None:
    phase = tmp_path / "p.yaml"
    phase.write_text(
        yaml.safe_dump({
            "budget": "1m",
            "eval": "final",
            "seeds": 3,
            "runs": {"ppo": {"configs": "coarse"}},
        })
    )
    specs = expand_phase(phase, methods_dir=METHODS)
    assert len(specs) == 6 * 3 + 10
    smoke_phase = REPO / "configs/phases/smoke.yaml"
    smoke = expand_phase(smoke_phase, method="ppo", methods_dir=METHODS)
    assert len(smoke) == 2


def copy_ppo(tmp_path: Path, tuned: dict[str, Any] | None) -> Path:
    shutil.copy(METHODS / "ppo.yaml", tmp_path / "ppo.yaml")
    raw = yaml.safe_load((tmp_path / "ppo.yaml").read_text())
    raw["tuned"] = tuned
    (tmp_path / "ppo.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
    (tmp_path / "child.yaml").write_text(
        yaml.safe_dump({"base": {"base_from": "ppo.tuned", "extra_key": 1}})
    )
    return tmp_path


def test_base_from_ppo_tuned_resolves_to_tuned_plus_extra_key(tmp_path: Path) -> None:
    tuned = {"lr": 4.0e-4, "gae_lambda": 0.9}
    methods = copy_ppo(tmp_path, tuned)
    child = load_method("child", methods)
    cfg = resolve_config(child, "child", "1m", methods_dir=methods)
    assert cfg == TABLE_11 | tuned | {"extra_key": 1, "method": "child"}
    cfg_1b = resolve_config(child, "child", "1b", methods_dir=methods)
    assert (cfg_1b["envs"], cfg_1b["lr"]) == (1024, 2.0e-4)


def test_base_from_an_empty_ppo_tuned_resolves_to_the_base(tmp_path: Path) -> None:
    methods = copy_ppo(tmp_path, {})  # marked tuned, and the base won
    child = load_method("child", methods)
    cfg = resolve_config(child, "child", "1m", methods_dir=methods)
    assert cfg == TABLE_11 | {"extra_key": 1, "method": "child"}


def test_base_from_ppo_tuned_fails_loudly_while_tuned_is_null(tmp_path: Path) -> None:
    methods = copy_ppo(tmp_path, None)
    child = load_method("child", methods)
    with pytest.raises(QueueConfigError, match=r"ppo\.tuned is null"):
        resolve_config(child, "child", "1m", methods_dir=methods)


def test_published_scores() -> None:
    raw = yaml.safe_load((REPO / "configs/reference/published_scores.yaml").read_text())
    got = {(m, b): e["score"] for m, budgets in raw.items() for b, e in budgets.items()}
    assert got == {
        ("ppo", "1m"): 2.2,
        ("ppo", "1b"): 11.9,
        ("ppo-rnn", "1m"): 2.3,
        ("ppo-rnn", "1b"): 15.3,
        ("pqn-rnn", "1b"): 16.0,
    }
    for budgets in raw.values():
        for entry in budgets.values():
            assert entry["source"].startswith("https://")
            assert "s_pub" in entry
            assert entry["s_pub"] is None or entry["s_pub"] > 0
