"""Tests for the shipped PPO-RNN config: Table 11, the GRU network, Stage 1."""

from pathlib import Path
from typing import Any

from wut_murl_craftax.orchestration.queue import (
    config_hash,
    expand_phase,
    load_method,
    resolve_config,
)
from wut_murl_craftax.training.hooks import HOOK_NAMES
from wut_murl_craftax.training.ppo_rnn import PPORNNConfig

REPO = Path(__file__).resolve().parents[2]
METHODS = REPO / "configs" / "methods"
BASE = {
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
    "rnn_width": 512,
    "activation": "relu",
    "resets": "optimistic",
    "reset_ratio": 16,
}
"""PPO Table 11 (the Craftax-1M column) with the recurrent network."""
TABLE_7 = BASE | {"envs": 1024, "num_steps": 64, "lr": 2.0e-4}
STAGE_1 = {
    "published-1m": {},
    "shape-1024x64": {"envs": 1024, "num_steps": 64, "lr": 2.0e-4},
    "shape-64x64": {"envs": 64, "num_steps": 64},
    "activation-tanh": {"activation": "tanh"},
    "lr-2e-4": {"lr": 2.0e-4},
    "lr-4e-4": {"lr": 4.0e-4},
    "lambda-0.9": {"gae_lambda": 0.9},
    "ent-0": {"ent": 0.0},
}


def ppo_rnn() -> Any:
    return load_method("ppo-rnn", METHODS)


def test_base_is_table_11_with_the_relu_gru_network() -> None:
    assert ppo_rnn().base == BASE
    cfg = resolve_config(ppo_rnn(), "ppo-rnn", "1m", methods_dir=METHODS)
    assert cfg == BASE | {"method": "ppo-rnn"}


def test_base_sets_every_trainer_field_but_the_runner_and_hook_ones() -> None:
    """The code defaults are PPO's Classic debugging ones, so none may be left out."""
    runner = {"steps", "n_ckpt", "seed", "extra_steps"}
    fields = set(PPORNNConfig.__dataclass_fields__) - runner - set(HOOK_NAMES)
    assert fields == set(BASE)
    PPORNNConfig.from_config(BASE | {"method": "ppo-rnn"})


def test_1b_budget_is_table_7_with_the_published_network() -> None:
    assert ppo_rnn().budgets == {"1b": {"envs": 1024, "num_steps": 64, "lr": 2.0e-4}}
    cfg = resolve_config(ppo_rnn(), "ppo-rnn", "1b", methods_dir=METHODS)
    assert cfg == TABLE_7 | {"method": "ppo-rnn"}
    built = PPORNNConfig.from_config(cfg)
    assert (built.rnn_width, built.activation) == (512, "relu")


def test_coarse_grid_is_eight_entries_each_differing_in_its_stage_1_keys() -> None:
    coarse = {
        e["name"]: {k: v for k, v in e.items() if k not in ("name", "seeds")}
        for e in ppo_rnn().coarse
    }
    assert coarse == STAGE_1
    assert [e["name"] for e in ppo_rnn().coarse if "seeds" in e] == ["published-1m"]
    assert ppo_rnn().coarse[0]["seeds"] == 10
    for name, over in STAGE_1.items():
        cfg = resolve_config(ppo_rnn(), "ppo-rnn", "1m", ref=name, methods_dir=METHODS)
        diff = {k for k in cfg if cfg[k] != (BASE | {"method": "ppo-rnn"})[k]}
        assert diff == set(over), name
        PPORNNConfig.from_config(cfg)  # a valid trainer config


def test_fine_holds_the_rule_and_knobs_and_no_entries() -> None:
    fine = ppo_rnn().fine
    assert fine is not None
    assert set(fine) == {"rule", "knobs"}
    assert fine["rule"] == {"select": 2, "combine": "each"}
    knobs = {k["key"]: k for k in fine["knobs"]}
    assert set(knobs) == {"lr", "gae_lambda", "ent"}
    assert knobs["lr"]["mul"] == [0.75, 1.5]
    assert knobs["gae_lambda"]["add"] == [-0.05, 0.05]
    assert (knobs["ent"]["mul"], knobs["ent"]["zero"]) == ([0.5, 2.0], [0.001, 0.003])


def test_tuned_is_the_base_and_the_abort_block_is_valid() -> None:
    assert ppo_rnn().tuned == {}  # tuning kept the published config
    for budget in ("1m", "1b"):
        tuned = resolve_config(ppo_rnn(), "ppo-rnn", budget, ref="tuned")
        assert tuned == resolve_config(ppo_rnn(), "ppo-rnn", budget)
    assert ppo_rnn().abort is not None
    assert ppo_rnn().abort["finite"]


def test_abort_rules_are_ppos_plus_the_recompute_bound_and_saturation_flag() -> None:
    abort = ppo_rnn().abort
    ppo = load_method("ppo", METHODS).abort
    assert abort is not None
    assert ppo is not None
    assert abort["finite"] == ppo["finite"]
    assert abort["flags"][:3] == ppo["flags"]  # kl, entropy, value: as PPO
    assert abort["flags"][3:] == [
        {
            "name": "hidden_saturation",
            "metric": "rnn/hidden_saturation",
            "above": 0.9,
            "span": 0.05,
        }
    ]
    assert abort["limits"] == [
        {"metric": "ppo/ratio_first_mb", "min": 0.999, "max": 1.001}
    ]
    assert "permanent" not in abort  # a NaN is retried once; only the limit is final


def test_tune_coarse_phase_plans_31_runs() -> None:
    phase = REPO / "configs" / "phases" / "tune-coarse.yaml"
    specs = expand_phase(phase, method="ppo-rnn", methods_dir=METHODS)
    assert len(specs) == 7 * 3 + 10
    assert {s.budget for s in specs} == {"1m"}
    assert {s.env for s in specs} == {"Craftax-Symbolic-v1"}
    assert len({config_hash(s.config) for s in specs}) == 8
    published = [s for s in specs if s.config_slug == "ppo-rnn" and s.seed >= 3]
    assert len(published) == 7  # seeds 3 to 9 exist only for published-1m


def test_smoke_runs_published_and_tanh_on_classic_with_one_seed() -> None:
    smoke = REPO / "configs" / "phases" / "smoke.yaml"
    specs = expand_phase(smoke, method="ppo-rnn", methods_dir=METHODS)
    assert [(s.config_slug, s.seed, s.budget) for s in specs] == [
        ("ppo-rnn", 0, "200k")
    ] * 2
    assert {s.env for s in specs} == {"Craftax-Classic-Symbolic-v1"}
    assert [s.config["activation"] for s in specs] == ["relu", "tanh"]
    assert len({s.config_hash for s in specs}) == 2
