"""Tests for the shipped PQN-RNN config: the repo YAML at the 1M shape, Stage 1."""

from pathlib import Path
from typing import Any

from test_pqn_rnn import FIELDS, REPO_1B

from wut_murl_craftax.orchestration.abort import flag_hits
from wut_murl_craftax.orchestration.queue import (
    config_hash,
    expand_fine,
    expand_phase,
    load_method,
    resolve_config,
)
from wut_murl_craftax.training.hooks import HOOK_NAMES
from wut_murl_craftax.training.pqn_rnn import PQNRNNConfig

REPO = Path(__file__).resolve().parents[2]
METHODS = REPO / "configs" / "methods"
PHASES = REPO / "configs" / "phases"
BASE = {
    "env": "Craftax-Symbolic-v1",
    "envs": 256,
    "num_steps": 16,
    "minibatches": 4,
    "epochs": 4,
    "lr": 3.0e-4,
    "lr_linear_decay": True,
    "max_grad_norm": 0.5,
    "gamma": 0.99,
    "lam": 0.5,
    "eps_start": 1.0,
    "eps_finish": 0.005,
    "eps_decay": 0.1,
    "hidden": 512,
    "norm_input": True,
    "norm_type": "layer_norm",
    "input_norm": "batch_renorm",
    "add_last_action": True,
    "rew_scale": 1.0,
    "reset_ratio": 16,
}
"""The repo YAML's settings with the 1M rollout shape (256 envs x 16 steps)."""
STAGE_1 = {
    "base-1m": {},
    "shape-1024x128": {"envs": 1024, "num_steps": 128},
    "shape-64x64": {"envs": 64, "num_steps": 64},
    "lr-1.5e-4": {"lr": 1.5e-4},
    "lr-6e-4": {"lr": 6.0e-4},
    "lambda-0": {"lam": 0.0},
    "lambda-0.8": {"lam": 0.8},
    "eps-decay-0.25": {"eps_decay": 0.25},
    "norm-batch-hidden": {"norm_type": "batch_norm"},
    "norm-layer-input": {"input_norm": "layer_norm"},
}


def pqn() -> Any:
    return load_method("pqn-rnn", METHODS)


def test_base_is_the_repo_yaml_at_the_1m_shape() -> None:
    assert pqn().base == BASE
    cfg = resolve_config(pqn(), "pqn-rnn", "1m", methods_dir=METHODS)
    assert cfg == BASE | {"method": "pqn-rnn"}


def test_base_sets_every_trainer_field_but_the_runner_hook_and_caller_ones() -> None:
    """The caller-set `extra_obs_dim` stays at its code default of 0."""
    runner = {"steps", "seed", "extra_steps", "extra_obs_dim"}
    fields = set(PQNRNNConfig.__dataclass_fields__) - runner - set(HOOK_NAMES)
    assert fields == set(BASE)
    PQNRNNConfig.from_config(BASE | {"method": "pqn-rnn"})


def test_1b_budget_is_the_repo_yaml_field_by_field() -> None:
    cfg = resolve_config(pqn(), "pqn-rnn", "1b", methods_dir=METHODS)
    assert {k for k in BASE if cfg[k] != BASE[k]} == {"envs", "num_steps"}
    assert (cfg["envs"], cfg["num_steps"]) == (1024, 128)
    built = PQNRNNConfig.from_config(cfg | {"steps": 10**9})
    for key, value in REPO_1B.items():
        assert getattr(built, FIELDS[key]) == value, key
    assert built.input_norm == "batch_renorm"


def test_coarse_grid_is_ten_entries_each_differing_in_its_stage_1_keys() -> None:
    coarse = {
        e["name"]: {k: v for k, v in e.items() if k != "name"} for e in pqn().coarse
    }
    assert coarse == STAGE_1  # no entry has its own seed count: all take the phase's 3
    for name, over in STAGE_1.items():
        cfg = resolve_config(pqn(), "pqn-rnn", "1m", ref=name, methods_dir=METHODS)
        diff = {k for k in cfg if cfg[k] != (BASE | {"method": "pqn-rnn"})[k]}
        assert diff == set(over), name
        PQNRNNConfig.from_config(cfg)  # a valid trainer config


def test_fine_holds_the_rule_and_knobs_and_no_entries() -> None:
    fine = pqn().fine
    assert fine is not None
    assert set(fine) == {"rule", "knobs"}
    assert fine["rule"] == {"select": 2, "combine": "each"}
    knobs = {k["key"]: k for k in fine["knobs"]}
    assert set(knobs) == {"lr", "lam", "eps_decay"}
    assert knobs["lr"]["mul"] == [0.75, 1.5]
    assert (knobs["lam"]["add"], knobs["lam"]["clip"]) == ([-0.1, 0.1], [0.0, 0.95])
    assert knobs["eps_decay"]["mul"] == [0.5, 2.0]


def test_fine_rule_yields_four_configs_from_two_knobs() -> None:
    scores = {}
    for e in pqn().coarse:
        cfg = resolve_config(pqn(), "pqn-rnn", "1m", ref=e["name"], methods_dir=METHODS)
        scores[config_hash(cfg)] = {"lr-6e-4": 5.0, "lambda-0.8": 4.0}.get(
            e["name"], 3.0
        )
    fine = expand_fine("pqn-rnn", scores, methods_dir=METHODS)["pqn-rnn"]
    assert len(fine) == 4
    assert set(fine) == {
        "fine-lr=0.00045",
        "fine-lr=0.0009",
        "fine-lam=0.4",
        "fine-lam=0.6",
    }


def test_tuned_is_the_base_and_the_abort_block_is_valid() -> None:
    assert pqn().tuned == {}  # tuning kept the base config
    for budget in ("1m", "1b"):
        tuned = resolve_config(pqn(), "pqn-rnn", budget, ref="tuned")
        assert tuned == resolve_config(pqn(), "pqn-rnn", budget)
    assert pqn().abort is not None
    assert pqn().abort["finite"] == ["pqn/td_loss"]


def test_abort_rules_bound_q_and_flag_td_spikes_flat_gaps_and_dead_inputs() -> None:
    abort = pqn().abort
    assert abort is not None
    assert abort["limits"] == [{"metric": "pqn/q_max_mean", "max": 300}]
    assert abort["flags"] == [
        {"name": "td_spike", "metric": "pqn/td_loss", "above_median": 10, "updates": 3},
        {
            "name": "action_gap_flat",
            "metric": "pqn/action_gap",
            "below": 1.0e-3,
            "updates": 3,
            "when": {"metric": "pqn/epsilon", "below": 0.0051},
        },
        {"name": "bn_var_low", "metric": "pqn/bn_var_min", "below": 1.0e-6},
    ]
    assert "permanent" not in abort  # a NaN is retried once; only the limit is final
    assert abort["limits"][0]["max"] > 226  # above every episode return


def test_at_1b_the_dead_input_flag_reads_the_share_of_dead_inputs() -> None:
    """Every 1B run would flag a minimum variance below 1e-6 (review ctrl-metrics M1).

    Measured at 1M (39 finished tune-coarse runs): 76.5-79.5% of the 8268 inputs never
    vary in a whole run (block types of unreached floors), so one of them decays below
    1e-6 after about 13.8k Q steps of 122k at 1B. A share above 0.9 is a collapse.
    """
    at_1m, at_1b = (pqn().abort_rules("pqn-rnn", b) or {} for b in ("1m", "1b"))
    rule = {"name": "bn_var_low", "metric": "pqn/bn_dead_share", "above": 0.9}
    assert at_1b["flags"] == [*at_1m["flags"][:2], rule]
    assert {k: v for k, v in at_1b.items() if k != "flags"} == {
        k: v for k, v in at_1m.items() if k != "flags"
    }
    assert "abort" not in resolve_config(pqn(), "pqn-rnn", "1b")
    n = 7629  # updates of a 1B run
    healthy = {"pqn/bn_var_min": [0.0] * n, "pqn/bn_dead_share": [0.79] * n}
    assert flag_hits(at_1b["flags"], healthy, n) == {}
    assert "bn_var_low" in flag_hits(at_1m["flags"], healthy, n)  # the old rule fires
    collapsed = {**healthy, "pqn/bn_dead_share": [0.79] * 100 + [0.97] * (n - 100)}
    assert flag_hits(at_1b["flags"], collapsed, n) == {"bn_var_low": 101}


def test_smoke_runs_base_and_lambda_0_on_classic_with_one_seed() -> None:
    specs = expand_phase(PHASES / "smoke.yaml", method="pqn-rnn", methods_dir=METHODS)
    assert [(s.config_slug, s.seed, s.budget) for s in specs] == [
        ("pqn-rnn", 0, "200k")
    ] * 2
    assert {s.env for s in specs} == {"Craftax-Classic-Symbolic-v1"}
    assert [s.config["lam"] for s in specs] == [0.5, 0.0]
    assert len({s.config_hash for s in specs}) == 2


def test_tune_coarse_phase_plans_30_runs() -> None:
    phase = PHASES / "tune-coarse.yaml"
    specs = expand_phase(phase, method="pqn-rnn", methods_dir=METHODS)
    assert len(specs) == 10 * 3
    assert {s.budget for s in specs} == {"1m"}
    assert {s.env for s in specs} == {"Craftax-Symbolic-v1"}
    assert len({config_hash(s.config) for s in specs}) == 10
    assert {s.seed for s in specs} == {0, 1, 2}
