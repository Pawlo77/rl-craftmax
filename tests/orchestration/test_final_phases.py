"""The shipped final phases: the plan's run counts, reused control runs, 1B settings."""

import re
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import yaml

from wut_murl_craftax.orchestration import queue
from wut_murl_craftax.orchestration.queue import (
    QueueConfigError,
    RunSpec,
    expand_phase,
    load_method,
    resolve_config,
)

REPO = Path(__file__).resolve().parents[2]
PHASES = REPO / "configs" / "phases"
TUNED = {
    "count-bonus": {
        "ppo-rnn": {"beta": 0.03, "granularity": "log2"},
        "pqn-rnn": {"beta": 0.3, "decay": "linear"},
    },
    "sil": {"lambda_sil": 0.3, "slots_windows": 8, "sil_steps": 2},
    "smdp-skills": {"alpha": 0.05, "top_horizon": 200, "timeout_mult": 2.0},
    "dyna-q": {"plan_variant": "prioritized", "plan_n": 20},
}
"""Stand-ins for the methods' tuned blocks, still null while their tuning runs."""


def _kappa_item(path: Path = PHASES / "final-1m.yaml") -> float:
    """The kappa of final-1m's dyna-q ablation: 0, or the best nonzero stage-1 kappa
    once a tuned kappa of 0 made `orchestration.finals` rewrite it."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return float(raw["runs"]["dyna-q"]["configs"][1]["set"]["plan_kappa"])


KAPPA_ITEM = _kappa_item()
TUNED["dyna-q"]["plan_kappa"] = 0.003 if KAPPA_ITEM == 0.0 else 0.0
"""A tuned kappa unlike the phase file's ablation, for either state of the item."""
FINAL_1M = {
    "ppo": 10,
    "ppo-rnn": 10,
    "pqn-rnn": 10,
    "count-bonus-ppo-rnn": 20,
    "count-bonus-pqn-rnn": 10,
    "sil": 20,
    "smdp-skills": 40,
    "dyna-q": 20,
}
"""Runs per registry entry of the compute budget's `final-1m` column."""
ABLATIONS = {
    "count-bonus-ppo-rnn": [{"key": "simhash", "log2_k": 24, "max_occupancy": 0.1}],
    "sil": [{"layout": "single"}],
    "smdp-skills": [{"eps_top_end": 1.0}, {"monte_carlo": True}, {"low_flags": False}],
    "dyna-q": [{"plan_kappa": KAPPA_ITEM}],
}
"""Each method plan's final-1m ablations: the keys they change in the tuned config."""


def _copy_methods(out: Path, tuned: dict[str, Any]) -> Path:
    """configs/methods in ``out`` with the ``tuned`` block of each slug in ``tuned``."""
    for path in (REPO / "configs" / "methods").glob("*.yaml"):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        raw["tuned"] = tuned.get(path.stem, raw["tuned"])
        (out / path.name).write_text(yaml.safe_dump(raw, sort_keys=False))
    return out


@pytest.fixture(scope="module")
def methods(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _copy_methods(tmp_path_factory.mktemp("methods"), TUNED)


def _specs(phase: str, methods: Path, method: str | None = None) -> list[RunSpec]:
    return expand_phase(PHASES / f"{phase}.yaml", method=method, methods_dir=methods)


def _by_config(specs: list[RunSpec]) -> dict[str, list[int]]:
    seeds: dict[str, list[int]] = {}
    for s in specs:
        seeds.setdefault(s.config_hash, []).append(s.seed)
    return seeds


def _tuned(methods: Path, registry: str, slug: str, budget: str) -> dict[str, Any]:
    method = load_method(slug, methods)
    return resolve_config(method, registry, budget, ref="tuned", methods_dir=methods)


def test_final_1m_is_the_plans_140_runs(methods: Path) -> None:
    specs = _specs("final-1m", methods)
    assert Counter(s.registry_name for s in specs) == FINAL_1M
    assert len(specs) == sum(FINAL_1M.values()) == 140
    assert {(s.budget, s.env, s.eval_mode, s.phase) for s in specs} == {
        ("1m", "Craftax-Symbolic-v1", "log_spaced", "final-1m")
    }
    assert all(seeds == list(range(10)) for seeds in _by_config(specs).values())


def test_final_1b_is_the_plans_24_runs(methods: Path) -> None:
    specs = _specs("final-1b", methods)
    assert Counter(s.registry_name for s in specs) == dict.fromkeys(FINAL_1M, 3)
    assert {(s.budget, s.env, s.eval_mode) for s in specs} == {
        ("1b", "Craftax-Symbolic-v1", "log_spaced")
    }
    assert all(seeds == [0, 1, 2] for seeds in _by_config(specs).values())


@pytest.mark.parametrize(("phase", "runs"), [("final-1m", 140), ("final-1b", 24)])
def test_the_dry_run_lists_the_plans_runs(
    methods: Path, capsys: pytest.CaptureFixture[str], phase: str, runs: int
) -> None:
    path = str(PHASES / f"{phase}.yaml")
    queue.main(["--phase", path, "--methods-dir", str(methods), "--dry-run"])
    names = capsys.readouterr().out.split()
    assert len(names) == len(set(names)) == runs


@pytest.mark.parametrize(
    ("method", "runs_1m", "runs_1b"),
    [("count-bonus", 30, 6), ("sil", 20, 3), ("smdp-skills", 40, 3), ("ppo", 10, 3)],
)
def test_a_method_slug_selects_its_runs(
    methods: Path, method: str, runs_1m: int, runs_1b: int
) -> None:
    assert len(_specs("final-1m", methods, method)) == runs_1m
    assert len(_specs("final-1b", methods, method)) == runs_1b


@pytest.mark.parametrize(
    ("control", "entry"), [("ppo", "published-1m"), ("ppo-rnn", "published-1m")]
)
def test_published_control_finals_are_the_tune_coarse_runs(
    methods: Path, control: str, entry: str
) -> None:
    """Tuning kept the published config: its 10 tune-coarse runs are the finals."""
    final = {s.name for s in _specs("final-1m", methods, control)}
    coarse = [
        s
        for s in _specs("tune-coarse", methods, control)
        if s.config
        == resolve_config(load_method(control, methods), control, "1m", ref=entry)
    ]
    assert final == {s.name for s in coarse}
    assert len(final) == 10


def test_the_pqn_rnn_final_reuses_its_three_tuning_runs(methods: Path) -> None:
    final = {s.name for s in _specs("final-1m", methods, "pqn-rnn")}
    pqn = load_method("pqn-rnn", methods)
    base = resolve_config(pqn, "pqn-rnn", "1m", ref="base-1m")
    coarse = {
        s.name for s in _specs("tune-coarse", methods, "pqn-rnn") if s.config == base
    }
    assert len(coarse) == 3
    assert coarse < final


@pytest.mark.parametrize("registry", list(ABLATIONS))
def test_ablations_change_only_their_keys_of_the_tuned_config(
    methods: Path, registry: str
) -> None:
    slug = "count-bonus" if registry.startswith("count-bonus") else registry
    tuned = _tuned(methods, registry, slug, "1m")
    configs = {
        s.config_hash: s.config
        for s in _specs("final-1m", methods, registry)
        if s.registry_name == registry
    }
    assert tuned in configs.values()
    others = [c for c in configs.values() if c != tuned]
    changed = [{k: v for k, v in c.items() if tuned.get(k) != v} for c in others]
    assert changed == ABLATIONS[registry]


def test_1b_controls_run_their_1b_columns(methods: Path) -> None:
    configs = {s.registry_name: s.config for s in _specs("final-1b", methods)}
    for control in ("ppo", "ppo-rnn", "pqn-rnn"):
        method = load_method(control, methods)
        assert configs[control] == resolve_config(method, control, "1b")
    assert {k: configs["ppo"][k] for k in ("envs", "num_steps", "lr")} == {
        "envs": 1024,
        "num_steps": 64,
        "lr": 2.0e-4,
    }
    assert configs["ppo-rnn"]["activation"] == "relu"
    assert (configs["ppo-rnn"]["envs"], configs["ppo-rnn"]["num_steps"]) == (1024, 64)
    assert (configs["pqn-rnn"]["envs"], configs["pqn-rnn"]["num_steps"]) == (1024, 128)


def test_1b_controls_follow_their_tuned_block_like_the_donors(tmp_path: Path) -> None:
    """A control's 1B final is its `tuned` config (empty today), so it always shares
    one donor with the methods built on it."""
    methods = _copy_methods(tmp_path, TUNED | {"ppo-rnn": {"ent": 0.002}})
    configs = {s.registry_name: s.config for s in _specs("final-1b", methods)}
    assert configs["ppo-rnn"] == _tuned(methods, "ppo-rnn", "ppo-rnn", "1b")
    assert configs["ppo-rnn"]["ent"] == configs["sil"]["ent"] == 0.002


@pytest.mark.parametrize(
    ("registry", "control", "knobs"),
    [
        ("count-bonus-ppo-rnn", "ppo-rnn", TUNED["count-bonus"]["ppo-rnn"]),
        ("count-bonus-pqn-rnn", "pqn-rnn", TUNED["count-bonus"]["pqn-rnn"]),
        ("sil", "ppo-rnn", TUNED["sil"]),
        ("smdp-skills", "pqn-rnn", TUNED["smdp-skills"]),
        ("dyna-q", "smdp-skills", TUNED["dyna-q"]),
    ],
)
def test_1b_methods_take_their_1m_knobs_on_the_1b_control_settings(
    methods: Path, registry: str, control: str, knobs: dict[str, Any]
) -> None:
    """Every key of the control's 1B final (or smdp-skills' for dyna-q) is unchanged,
    but the table size dyna-q's own `budgets.1b` block sets."""
    configs = {s.registry_name: s.config for s in _specs("final-1b", methods)}
    cfg, base = configs[registry], configs[control]
    own = {"log2_rows"} if registry == "dyna-q" else set()
    keep = {k: v for k, v in base.items() if k != "method" and k not in own}
    assert {k: cfg[k] for k in keep} == keep
    assert {k: cfg[k] for k in knobs} == knobs


M1_NAMES = {
    "ppo": {"e06ac6db"},
    "ppo-rnn": {"b3ce95c4"},
    "pqn-rnn": {"10cbfc16"},
    "count-bonus-ppo-rnn": {"9403d21d", "b0b1c535"},
    "count-bonus-pqn-rnn": {"e6e1fafa"},
    "sil": {"57a864f9", "6f8690ad"},
    "smdp-skills": {"0b84c247", "1f559f2c", "20ec0807", "89f3c98f"},
    "dyna-q": {"3fd02b70", "74c22d85"},
    "combo": {"fd6c4333"},
}
"""Config hashes of the finished final-1m and combo-1m runs (the shipped methods)."""


@pytest.mark.parametrize("phase", ["final-1m", "combo-1m"])
def test_1m_finals_keep_the_names_that_ran(phase: str) -> None:
    """A change meant for 1B (a `budgets.1b` block, a donor) must not rename them."""
    hashes: dict[str, set[str]] = {}
    for s in expand_phase(PHASES / f"{phase}.yaml"):
        hashes.setdefault(s.registry_name, set()).add(s.config_hash)
    assert hashes == {k: v for k, v in M1_NAMES.items() if k in hashes}
    assert len(hashes) == (1 if phase == "combo-1m" else 8)


def test_1b_run_names_never_meet_a_1m_name() -> None:
    names_1b = {s.name for p in ("final-1b", "combo-1b") for s in _real(p)}
    names_1m = {s.name for p in ("final-1m", "combo-1m") for s in _real(p)}
    assert len(names_1b) == 27
    assert not names_1b & names_1m
    assert all(re.fullmatch(r"[a-z-]+-1b-[0-9a-f]{8}-s[0-2]", n) for n in names_1b)


def _real(phase: str) -> list[RunSpec]:
    return expand_phase(PHASES / f"{phase}.yaml")


@pytest.mark.parametrize(
    ("slug", "registry"),
    [
        ("count-bonus", "count-bonus-ppo-rnn"),
        ("count-bonus", "count-bonus-pqn-rnn"),
        ("combo", "combo"),
    ],
)
def test_the_1b_count_tables_have_four_times_the_1m_slots(
    slug: str, registry: str
) -> None:
    """The 1M tuned runs end at 16-115 keys; the largest whole-run log-log projection
    is 36,700 keys at 1B (199,000 by the last quarter's rate): 7.1 x (1.3 x) the
    guard of 2^20 slots, 28.6 x (5.3 x) that of 2^22."""
    ref = "combined" if slug == "combo" else "tuned"
    method = load_method(slug)
    for budget, log2_k in (("1m", 20), ("1b", 22)):
        cfg = resolve_config(method, registry, budget, ref=ref)
        assert (cfg["log2_k"], cfg["max_occupancy"]) == (log2_k, 0.25)


def test_the_1b_dyna_tables_are_sized_for_a_1b_run() -> None:
    """A log-log fit of the tuned 1M runs projects 1.2 million rows and 23 million
    pairs at 1B (smdp-skills 50,000 rows): the 1M shapes (2^20 rows with
    `smdp/table_full` at a quarter of them, 2^16 pairs with `dyna/model_full` at the
    last one) would end dyna-q runs for good, and 2^24 / 2^25 is what 8 GB allows."""
    shape = {
        (b, r): resolve_config(load_method(r), r, b)
        for b in ("1m", "1b")
        for r in ("smdp-skills", "dyna-q")
    }
    assert shape["1m", "smdp-skills"]["log2_rows"] == 20
    assert shape["1b", "smdp-skills"]["log2_rows"] == 20  # its projection fits 2^20
    assert shape["1m", "dyna-q"]["log2_rows"] == 20
    assert shape["1m", "dyna-q"]["model_pairs"] == 2**16
    assert shape["1b", "dyna-q"]["log2_rows"] == 24
    assert shape["1b", "dyna-q"]["model_pairs"] == 2**25


def test_a_zero_tuned_kappa_makes_the_kappa_ablation_fail_loudly(
    tmp_path: Path,
) -> None:
    """The item as shipped sets kappa 0: with a tuned kappa of 0 it must be rewritten
    (on a copy whose item is 0 again, whatever the shipped file says now)."""
    text = (PHASES / "final-1m.yaml").read_text(encoding="utf-8")
    zero = re.sub(
        r"set: \{plan_kappa: [0-9.e-]+\}\}\]\}$",
        "set: {plan_kappa: 0.0}}]}",
        text,
        flags=re.MULTILINE,
    )
    (tmp_path / "final-1m.yaml").write_text(zero, encoding="utf-8")
    assert _kappa_item(tmp_path / "final-1m.yaml") == 0.0
    tuned = TUNED | {"dyna-q": TUNED["dyna-q"] | {"plan_kappa": 0.0}}
    methods = tmp_path / "methods"
    methods.mkdir()
    with pytest.raises(QueueConfigError, match="duplicate run dyna-q-1m-"):
        expand_phase(
            tmp_path / "final-1m.yaml",
            method="dyna-q",
            methods_dir=_copy_methods(methods, tuned),
        )
