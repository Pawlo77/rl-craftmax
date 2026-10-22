"""Tests for run identity, config resolution and queue expansion."""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import mlflow
import pytest
import yaml

from wut_murl_craftax.orchestration import queue, runner
from wut_murl_craftax.orchestration.queue import (
    RunSpec,
    budget_steps,
    coarse_results,
    config_hash,
    expand_fine,
    expand_phase,
    load_method,
    mark_tuned,
    resolve_config,
)

CONFIGS = Path(__file__).parent / "fixtures" / "configs"
METHODS = CONFIGS / "methods"
PHASES = CONFIGS / "phases"
REPO = Path(__file__).resolve().parents[2]

STAGE1 = {
    "published": 10.0,
    "lr-hi": 14.0,
    "lr-lo": 9.0,
    "lam-lo": 11.0,
    "lam-hi": 10.5,
    "ent-0": 12.0,
    "shape": 8.0,
}
"""Stage-1 mean scores of the toy grid: lr spread 5, entropy 2, lambda 1."""


def phase(name: str, **kw: Any) -> list[RunSpec]:
    return expand_phase(PHASES / f"{name}.yaml", methods_dir=METHODS, **kw)


def resolved(
    slug: str,
    ref: str,
    budget: str = "1m",
    registry: str | None = None,
    methods_dir: Path = METHODS,
    **kw: Any,
) -> dict[str, Any]:
    return resolve_config(
        load_method(slug, methods_dir),
        registry or slug,
        budget,
        ref=ref,
        methods_dir=methods_dir,
        **kw,
    )


def stage1(slug: str, scores: dict[str, float]) -> dict[str, float]:
    """Stage-1 results {hash8: score} from {entry name: score}."""
    return {config_hash(resolved(slug, name)): score for name, score in scores.items()}


def phase_from(tmp_path: Path, body: dict[str, Any]) -> list[RunSpec]:
    (tmp_path / "p.yaml").write_text(yaml.safe_dump(body))
    return expand_phase(tmp_path / "p.yaml", methods_dir=METHODS)


# --- run identity ---------------------------------------------------------------


def test_two_configs_two_seeds_expand_to_four_named_specs() -> None:
    specs = phase("two_by_two")
    assert len(specs) == 4
    first = specs[0]
    assert (first.registry_name, first.config_slug, first.budget, first.seed) == (
        "toy",
        "toy",
        "1m",
        0,
    )
    assert (first.env, first.phase, first.eval_mode) == (
        "Toy-v1",
        "two_by_two",
        "final",
    )
    assert [s.seed for s in specs] == [0, 1, 0, 1]
    assert len({s.name for s in specs}) == 4
    assert all(re.fullmatch(r"toy-1m-[0-9a-f]{8}-s[01]", s.name) for s in specs)
    assert first.steps == 1_000_000
    assert first.config["method"] == "toy"
    assert first.config_hash == config_hash(first.config)


def test_run_names_are_stable_across_processes() -> None:
    code = (
        "from pathlib import Path\n"
        "from wut_murl_craftax.orchestration.queue import expand_phase\n"
        f"c = Path({str(CONFIGS)!r})\n"
        "m = c / 'methods'\n"
        "for s in expand_phase(c / 'phases/two_by_two.yaml', methods_dir=m):\n"
        "    print(s.name)\n"
    )
    outs = [
        subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO,
        ).stdout.split()
        for seed in ("1", "2")
    ]
    assert outs[0] == outs[1] == [s.name for s in phase("two_by_two")]


def test_hash_ignores_callables_and_key_order() -> None:
    a = {"lr": 1.0e-3, "env": "X", "hook": print, "nested": {"f": len, "k": 1}}
    b = {"nested": {"k": 1}, "env": "X", "lr": 1.0e-3}
    assert config_hash(a) == config_hash(b)
    assert config_hash(a) != config_hash({**b, "lr": 2.0e-3})
    assert re.fullmatch(r"[0-9a-f]{8}", config_hash(a))


def test_seeds_item_beats_entry_beats_grid_beats_phase_default() -> None:
    seeds: dict[tuple[str, str], list[int]] = {}
    for s in phase("seeds"):
        seeds.setdefault((s.registry_name, s.config_hash), []).append(s.seed)

    def key(slug: str, ref: str, **kw: Any) -> tuple[str, str]:
        return slug, config_hash(resolved(slug, ref, **kw))

    assert seeds == {
        key("toy", "published"): [0],  # the entry's seeds: 1 beats the grid entry's 3
        key("toy", "lr-hi"): [0],
        key("toy", "lr-lo"): [5, 7],  # the item's own list
        key("toy", "lam-lo", overrides={"ent": 0.02}): [0],
        key("donor", "published"): [0, 1, 2],  # the phase default
    }


def test_grid_entry_seeds_beat_the_phase_default(tmp_path: Path) -> None:
    body = {
        "budget": "1m",
        "eval": "final",
        "seeds": 2,
        "runs": {"toy": {"configs": ["published", "lr-hi"]}},
    }
    specs = phase_from(tmp_path, body)
    counts = {
        h: sum(s.config_hash == h for s in specs)
        for h in {s.config_hash for s in specs}
    }
    assert sorted(counts.values()) == [2, 3]  # published has seeds: 3 in toy.yaml


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"budget": "1m", "eval": "weird", "runs": {"toy": {}}}, "eval"),
        ({"budget": "1x", "eval": "final", "runs": {"toy": {}}}, "budget"),
        ({"budget": "1m", "eval": "final", "runs": {"toy": {}}, "extra": 1}, "extra"),
        ({"budget": "1m", "eval": "final", "runs": {"toy": {"cfgs": []}}}, "cfgs"),
        (
            {
                "budget": "1m",
                "eval": "final",
                "runs": {"toy": {"configs": ["nope"], "seeds": 1}},
            },
            "nope",
        ),
        (
            {
                "budget": "1m",
                "eval": "final",
                "runs": {"toy": {"configs": ["tuned"], "seeds": 1}},
            },
            "tuned",
        ),
        (
            {
                "budget": "1m",
                "eval": "final",
                "runs": {"toy": {"configs": "fine", "seeds": 1}},
            },
            "expand-fine",
        ),
        (
            {"budget": "1m", "eval": "final", "runs": {"toy": {"configs": ["lr-hi"]}}},
            "seeds",
        ),
        (
            {
                "budget": "1m",
                "eval": "final",
                "seeds": 1,
                "runs": {"toy": {"configs": ["lr-hi", "lr-hi"]}},
            },
            "duplicate",
        ),
    ],
)
def test_phase_yaml_fails_loudly(
    tmp_path: Path, body: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        phase_from(tmp_path, body)


def test_budget_labels() -> None:
    assert [budget_steps(b) for b in ("1m", "1b", "200k")] == [
        1_000_000,
        1_000_000_000,
        200_000,
    ]
    with pytest.raises(ValueError, match="budget"):
        budget_steps("1x")


def test_yaml_floats_without_a_dot_are_floats() -> None:
    lr = load_method("toy", METHODS).base[
        "lr"
    ]  # written `3e-4`, which plain PyYAML reads as str
    assert isinstance(lr, float)
    assert lr == 3.0e-4


# --- method schema and resolution ----------------------------------------------


@pytest.mark.parametrize(
    "edit",
    [
        {"extra": 1},
        {"grid": {"coarse": [], "oops": 1}},
        {
            "grid": {
                "coarse": [],
                "fine": {"rule": {"select": 1, "nope": 1}, "knobs": []},
            }
        },
        {
            "grid": {
                "coarse": [],
                "fine": {"rule": {}, "knobs": [{"key": "lr", "mul": [2], "x": 1}]},
            }
        },
        {
            "grid": {
                "coarse": [],
                "fine": {"rule": {}, "knobs": [{"key": "g", "top": 0}]},
            }
        },
        {
            "grid": {
                "coarse": [],
                "fine": {"rule": {}, "knobs": [{"key": "g", "top": 2, "mul": [2]}]},
            }
        },
        {
            "grid": {
                "coarse": [],
                "fine": {"rule": {}, "knobs": [{"key": "g", "top": 2, "clip": [0, 1]}]},
            }
        },
        {"grid": {"coarse": [{"lr": 1}]}},
        {"grid": {"coarse": [{"name": "a"}, {"name": "a"}]}},
        {"grid": {"coarse": [{"name": "tuned"}]}},
        {"abort": {"nan": True}},
        {"abort": {"finite": "ppo/value_loss"}},
        {"abort": {"finite": [1]}},
        {"abort": {"flags": [{"name": "a", "metric": "m"}]}},
        {"abort": {"flags": [{"name": "a", "metric": "m", "above": 1, "below": 0}]}},
        {"abort": {"flags": [{"name": "a", "metric": "m", "above": 1, "x": 0}]}},
        {"abort": {"flags": [{"name": "a", "metric": "m", "above": 1, "updates": 0}]}},
        {"abort": {"flags": [{"name": "a", "metric": "m", "above": 1, "before": 2}]}},
        {"abort": {"flags": [{"metric": "m", "above": 1}]}},
        {"abort": {"flags": [{"name": "a", "metric": "m", "above": 1}] * 2}},
        {"abort": {"flags": [{"name": "a", "metric": "m", "above": 1, "span": 0}]}},
        {"abort": {"flags": [{"name": "a", "metric": "m", "above_median": 0}]}},
        {
            "abort": {
                "flags": [{"name": "a", "metric": "m", "above_median": 2, "above": 1}]
            }
        },
        {"abort": {"flags": [{"name": "a", "metric": "m", "above": 1, "when": 3}]}},
        {"abort": {"flags": [{"name": "a", "metric": "m", "above": 1, "when": {}}]}},
        {"abort": {"limits": {"metric": "m", "max": 1}}},
        {"abort": {"limits": [{"metric": "m"}]}},
        {"abort": {"limits": [{"metric": "m", "max": "1"}]}},
        {"abort": {"limits": [{"metric": "m", "min": 2, "max": 1}]}},
        {"abort": {"limits": [{"max": 1}]}},
        {"abort": {"limits": [{"metric": "m", "max": 1, "x": 0}]}},
        {"abort": {"limits": [{"metric": "m", "max": 1}, {"metric": "m", "min": 0}]}},
    ],
)
def test_method_yaml_fails_loudly(tmp_path: Path, edit: dict[str, Any]) -> None:
    (tmp_path / "bad.yaml").write_text(yaml.safe_dump({"base": {"env": "X"}, **edit}))
    with pytest.raises(ValueError, match="bad"):
        load_method("bad", tmp_path)


@pytest.mark.parametrize("permanent", ["m", ["n"], [1]])
def test_permanent_guards_are_a_list_of_finite_metrics(
    tmp_path: Path, permanent: object
) -> None:
    """A guard aborts through ``finite``; ``permanent`` only stops its retry."""
    abort = {"finite": ["m"], "permanent": permanent}
    (tmp_path / "bad.yaml").write_text(
        yaml.safe_dump({"base": {"env": "X"}, "abort": abort})
    )
    with pytest.raises(ValueError, match=r"bad.yaml abort.permanent"):
        load_method("bad", tmp_path)
    abort["permanent"] = ["m"]
    (tmp_path / "ok.yaml").write_text(
        yaml.safe_dump({"base": {"env": "X"}, "abort": abort})
    )
    assert load_method("ok", tmp_path).abort == abort


def test_the_abort_block_reaches_every_spec_of_the_method() -> None:
    flags = load_method("toy", METHODS).abort["flags"]  # type: ignore[index]
    assert [f["name"] for f in flags] == [
        "kl_high",
        "entropy_collapse",
        "value_failure",
    ]
    specs = phase("entries")
    toy = [s for s in specs if s.registry_name == "toy"]
    assert toy
    assert all(s.abort["flags"] == flags for s in toy)  # type: ignore[index]
    assert all(s.abort["finite"] == ["ppo/value_loss"] for s in toy)  # type: ignore[index]
    assert all(s.abort is None for s in specs if s.registry_name != "toy")


def test_base_from_resolves_the_donor_tuned_block_plus_own_keys() -> None:
    assert resolved("child", "b") == {
        "env": "Toy-v1",
        "lr": 1.0e-3,
        "width": 256,
        "beta": 0.3,
        "method": "child",
    }


def test_base_from_fails_while_the_donor_tuned_is_null() -> None:
    with pytest.raises(ValueError, match=r"toy\.tuned"):
        resolved("waiting", "a")


def test_budget_1b_resolves_base_then_tuned_then_budgets_and_carries_the_donors() -> (
    None
):
    donor = resolved("donor", "tuned", "1b")
    # base lr 3e-4, tuned 1e-3 and width 256, then budgets.1b lr 2e-4 wins over tuned
    assert (donor["lr"], donor["width"]) == (2.0e-4, 256)
    child = resolved("child", "b", "1b")
    # the donor's 1b settings come along; the child's budgets.1b overrides its beta
    assert (child["lr"], child["width"], child["beta"]) == (2.0e-4, 256, 0.05)
    assert (
        resolved("toy", "lr-hi", "1b")["envs"],
        resolved("toy", "lr-hi", "1b")["lr"],
    ) == (1024, 2.0e-4)


def test_phase_set_overrides_everything() -> None:
    spec = next(s for s in phase("seeds") if s.config["ent"] == 0.02)
    assert (spec.config["gae_lambda"], spec.config["ent"]) == (0.7, 0.02)


def test_entry_forms_and_method_selection() -> None:
    specs = phase("entries")
    assert {s.registry_name for s in specs} == {"multi-ppo-rnn", "multi-pqn-rnn", "toy"}
    ppo, pqn = (
        next(s for s in specs if s.registry_name == n)
        for n in ("multi-ppo-rnn", "multi-pqn-rnn")
    )
    assert ppo.config_slug == pqn.config_slug == "multi"
    assert ppo.eval_mode == "log_spaced"
    assert (
        ppo.config["width"] == 256
    )  # base_from donor.tuned; the pqn variant has no donor
    assert "width" not in pqn.config
    assert ppo.config_hash != pqn.config_hash
    assert {s.registry_name for s in phase("entries", method="multi-ppo-rnn")} == {
        "multi-ppo-rnn"
    }
    assert {s.registry_name for s in phase("entries", method="multi")} == {
        "multi-ppo-rnn",
        "multi-pqn-rnn",
    }
    with pytest.raises(ValueError, match="nope"):
        phase("entries", method="nope")


def test_the_method_filter_skips_the_other_entries_configs() -> None:
    # `toy` has no tuned block, so its entry fails; selecting multi never touches it
    assert [s.registry_name for s in phase("partial", method="multi")] == [
        "multi-ppo-rnn"
    ]
    with pytest.raises(ValueError, match=r"toy\.tuned"):
        phase("partial")


# --- fine rule -----------------------------------------------------------------


def test_expand_fine_picks_the_best_config_and_the_two_largest_effects() -> None:
    fine = expand_fine("toy", stage1("toy", STAGE1), methods_dir=METHODS)
    assert list(fine) == ["toy"]
    # best lr-hi (lr 4e-4); lr spreads 5, entropy 2, lambda 1 (`shape` is no lr axis)
    assert list(fine["toy"].values()) == [
        {"lr": 3.0e-4},
        {"lr": 6.0e-4},
        {"lr": 4.0e-4, "ent": 0.005},
        {"lr": 4.0e-4, "ent": 0.02},
    ]
    assert expand_fine("toy", stage1("toy", STAGE1), methods_dir=METHODS) == fine


def test_expand_fine_uses_the_zero_values_when_the_best_value_is_zero() -> None:
    fine = expand_fine(
        "toy", stage1("toy", {**STAGE1, "ent-0": 20.0}), methods_dir=METHODS
    )
    assert list(fine["toy"].values()) == [
        {"ent": 0.001},
        {"ent": 0.003},
        {"ent": 0.0, "lr": 2.25e-4},
        {"ent": 0.0, "lr": 4.5e-4},
    ]


def test_a_fine_config_that_equals_the_stage_1_config_has_its_run_name() -> None:
    fine = expand_fine("toy", stage1("toy", STAGE1), methods_dir=METHODS)
    cfg = resolved("toy", "base", overrides=next(iter(fine["toy"].values())))
    assert cfg["lr"] == 3.0e-4
    assert config_hash(cfg) == config_hash(resolved("toy", "published"))


def test_expand_fine_cross_integer_and_clip() -> None:
    fine = expand_fine(
        "crossed", stage1("crossed", {"c1": 1.0, "c2": 2.0}), methods_dir=METHODS
    )
    assert list(fine["crossed"].values()) == [
        {"slots": s, "lam": lam} for s in (4, 8, 16) for lam in (0.8, 0.95)
    ]


def catfine_scores(decay_gap: float = 0.5) -> dict[str, float]:
    """Stage-1 scores of `catfine`: beta 0.3 and `recipe` best, granularity spread 3."""
    gran = {"tiers": 2.0, "recipe": 5.0, "log2": 3.0}
    dec = {"constant": decay_gap, "linear": 0.0}
    names = {
        f"b{b}-{g}-{d}": gv + dv + (1.0 if b == 0.3 else 0.0)
        for b in (0.1, 0.3)
        for g, gv in gran.items()
        for d, dv in dec.items()
    }
    return stage1("catfine", names)


def test_expand_fine_crosses_numeric_multipliers_with_the_top_categorical_values() -> (
    None
):
    fine = expand_fine("catfine", catfine_scores(), methods_dir=METHODS)
    # beta is always chosen; granularity (spread 3) beats decay (spread 0.5); top 2 by
    # mean: recipe, log2; the remaining knob (decay) stays at the best config's value
    assert list(fine["catfine"].values()) == [
        {"beta": b, "granularity": g, "decay": "constant"}
        for b in (0.15, 0.3, 0.6)
        for g in ("recipe", "log2")
    ]
    assert next(iter(fine["catfine"])) == "fine-beta=0.15-granularity=recipe"
    assert expand_fine("catfine", catfine_scores(), methods_dir=METHODS) == fine


def test_the_categorical_knob_with_the_larger_spread_is_chosen() -> None:
    fine = expand_fine("catfine", catfine_scores(decay_gap=10.0), methods_dir=METHODS)
    # decay now spreads more than granularity; its top 2 are both values
    assert {(v["granularity"], v["decay"]) for v in fine["catfine"].values()} == {
        ("recipe", "constant"),
        ("recipe", "linear"),
    }
    assert len(fine["catfine"]) == 6


def test_the_stage_1_best_config_is_among_the_top_categorical_crosses() -> None:
    fine = expand_fine("catfine", catfine_scores(), methods_dir=METHODS)
    cfg = resolved(
        "catfine", "base", overrides=fine["catfine"]["fine-beta=0.3-granularity=recipe"]
    )
    assert config_hash(cfg) == config_hash(resolved("catfine", "b0.3-recipe-constant"))


def test_the_best_configs_categorical_value_is_kept_even_below_the_top_means() -> None:
    """b0.3-tiers-constant is best alone, its tiers mean the lowest: tiers comes first,
    the rest is filled by mean (recipe), so the fine grid reuses the best config."""
    gran = {"tiers": -20.0, "recipe": 5.0, "log2": 3.0}
    names = {
        f"b{b}-{g}-{d}": gv + (0.5 if d == "constant" else 0.0) + (b == 0.3)
        for b in (0.1, 0.3)
        for g, gv in gran.items()
        for d in ("constant", "linear")
    }
    names["b0.3-tiers-constant"] = 10.0
    fine = expand_fine("catfine", stage1("catfine", names), methods_dir=METHODS)
    assert list(fine["catfine"].values()) == [
        {"beta": b, "granularity": g, "decay": "constant"}
        for b in (0.15, 0.3, 0.6)
        for g in ("tiers", "recipe")
    ]
    best = config_hash(resolved("catfine", "b0.3-tiers-constant"))
    hashes = {
        config_hash(resolved("catfine", "base", overrides=over))
        for over in fine["catfine"].values()
    }
    assert best in hashes


def test_expand_fine_per_registry_entry_and_method_filter() -> None:
    ppo, pqn = "multi-ppo-rnn", "multi-pqn-rnn"
    scores = {
        config_hash(resolved("multi", ref, registry=reg)): score
        for reg, ref, score in (
            (ppo, "a", 2.0),
            (ppo, "b", 1.0),
            (pqn, "a", 1.0),
            (pqn, "b", 2.0),
        )
    }
    fine = expand_fine("multi", scores, methods_dir=METHODS)
    assert list(fine) == [ppo, pqn]
    assert [v["beta"] for v in fine[ppo].values()] == [0.05, 0.2]  # best a: beta 0.1
    assert [v["beta"] for v in fine[pqn].values()] == [0.15, 0.6]  # best b: beta 0.3
    assert list(expand_fine("multi", scores, methods_dir=METHODS, method=pqn)) == [pqn]
    assert list(expand_fine("multi", scores, methods_dir=METHODS, method="multi")) == [
        ppo,
        pqn,
    ]


def test_expand_fine_fails_on_missing_results_or_rule() -> None:
    partial = stage1("toy", {k: v for k, v in STAGE1.items() if k != "shape"})
    with pytest.raises(ValueError, match="shape"):
        expand_fine("toy", partial, methods_dir=METHODS)
    with pytest.raises(ValueError, match=r"grid\.fine"):
        expand_fine("donor", {}, methods_dir=METHODS)


def test_a_phase_uses_the_generated_fine_configs() -> None:
    fine = expand_fine("toy", stage1("toy", STAGE1), methods_dir=METHODS)
    assert len(phase("fine", fine=fine)) == 4 * 3
    with pytest.raises(ValueError, match="expand-fine"):
        phase("fine")


# --- MLflow-backed pieces ------------------------------------------------------


class Store:
    """A throwaway MLflow store; `add` logs a finished run with the queue's tags."""

    def __init__(self, tmp_path: Path) -> None:
        self.tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
        self.client = mlflow.MlflowClient(tracking_uri=self.tracking_uri)
        self.exp = self.client.create_experiment(
            "tune-coarse", artifact_location=(tmp_path / "art").as_uri()
        )

    def add(
        self,
        method: str,
        cfg: dict[str, Any],
        score: float,
        *,
        seed: int = 0,
        phases: str = "tune-coarse",
        status: str = "FINISHED",
        budget: str = "1m",
        extra_tags: dict[str, str] | None = None,
    ) -> str:
        tags = {
            queue.TAG_METHOD: method,
            queue.TAG_CONFIG_HASH: config_hash(cfg),
            queue.TAG_PHASES: phases,
            queue.TAG_BUDGET: budget,
            "seed": str(seed),
            **(extra_tags or {}),
        }
        name = f"{method}-{budget}-{config_hash(cfg)}-s{seed}"
        rid = self.client.create_run(self.exp, tags=tags, run_name=name).info.run_id
        self.client.log_metric(rid, queue.SCORE_METRIC, score)
        self.client.log_dict(rid, cfg, queue.CONFIG_ARTIFACT)
        self.client.set_terminated(rid, status)
        return rid

    def tag(self, rid: str, key: str) -> str | None:
        return self.client.get_run(rid).data.tags.get(key)


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "s")


@pytest.fixture
def methods(tmp_path: Path) -> Path:
    """A writable copy of the fixture methods."""
    shutil.copytree(METHODS, tmp_path / "methods")
    return tmp_path / "methods"


@pytest.fixture
def coarse_phase(tmp_path: Path) -> Path:
    """A phases dir whose tune-coarse lists the toy published and lr-hi, 2 seeds."""
    phases = tmp_path / "phases"
    phases.mkdir()
    body = {"budget": "1m", "eval": "final", "seeds": 2}
    body["runs"] = {"toy": {"configs": [{"ref": "published", "seeds": 2}, "lr-hi"]}}
    (phases / "tune-coarse.yaml").write_text(yaml.safe_dump(body))
    return phases


def _coarse(store: Store, phases: Path, **kw: Any) -> dict[str, float]:
    return coarse_results(
        "toy",
        tracking_uri=store.tracking_uri,
        methods_dir=METHODS,
        phases_dir=phases,
        **kw,
    )


def test_coarse_results_are_seed_means_of_complete_configs(
    store: Store, coarse_phase: Path
) -> None:
    pub, hi = resolved("toy", "published"), resolved("toy", "lr-hi")
    store.add("toy", pub, 10.0, seed=0)
    store.add("toy", pub, 20.0, seed=1, phases="smoke,tune-coarse,final-1m")
    store.add("toy", pub, 99.0, seed=7)  # a seed the phase does not list
    store.add("other", hi, 55.0)  # another method
    for seed in (0, 1):
        store.add("toy", hi, 4.0 + seed, seed=seed)
    got = _coarse(store, coarse_phase)
    assert got == {config_hash(pub): pytest.approx(15.0), config_hash(hi): 4.5}


@pytest.mark.parametrize(
    "seed_1",
    [
        {"status": "FAILED"},  # not finished
        {"phases": "final-1m"},  # not a tune-coarse run
        {"score": float("nan")},  # no episodes
        {"extra_tags": {"abort/bug": "true"}},
        None,  # not started
    ],
)
def test_an_incomplete_config_is_refused_unless_partial_results_are_allowed(
    store: Store,
    coarse_phase: Path,
    caplog: pytest.LogCaptureFixture,
    seed_1: dict[str, Any] | None,
) -> None:
    pub, hi = resolved("toy", "published"), resolved("toy", "lr-hi")
    for seed in (0, 1):
        store.add("toy", pub, 10.0, seed=seed)
    store.add("toy", hi, 4.0, seed=0)
    if seed_1 is not None:
        store.add("toy", hi, **{"score": 5.0, **seed_1}, seed=1)
    missing = f"toy-1m-{config_hash(hi)}-s1"
    with pytest.raises(
        ValueError, match=rf"allow-partial.*{config_hash(hi)}: {missing}"
    ):
        _coarse(store, coarse_phase)
    assert _coarse(store, coarse_phase, allow_partial=True) == {config_hash(pub): 10.0}
    assert missing in caplog.text


def test_partial_results_expand_around_the_complete_configs() -> None:
    partial = stage1("toy", {k: v for k, v in STAGE1.items() if k != "lr-hi"})
    with pytest.raises(ValueError, match="lr-hi"):
        expand_fine("toy", partial, methods_dir=METHODS)
    fine = expand_fine("toy", partial, methods_dir=METHODS, allow_partial=True)
    assert fine["toy"]  # ranked over the six complete configs


def test_mark_tuned_writes_the_block_and_the_tag_together(
    store: Store, methods: Path
) -> None:
    best = resolved("toy", "lr-hi", methods_dir=methods)
    rid = store.add("toy", best, 14.0)
    other = store.add("toy", resolved("toy", "lr-lo", methods_dir=methods), 9.0)
    before = (methods / "toy.yaml").read_text()

    diff = mark_tuned(
        "toy", config_hash(best), tracking_uri=store.tracking_uri, methods_dir=methods
    )

    assert load_method("toy", methods).tuned == {"lr": 4.0e-4}
    assert "+tuned:" in diff
    assert "-tuned: null" in diff
    # everything but the tuned block is byte for byte what it was, comments included
    after = (methods / "toy.yaml").read_text()
    assert after.replace("tuned:\n  lr: 0.0004\n", "tuned: null\n") == before
    assert store.tag(rid, "best") == "true"
    assert store.tag(other, "best") is None
    assert config_hash(resolved("toy", "tuned", methods_dir=methods)) == config_hash(
        best
    )


def test_mark_tuned_again_moves_the_best_tag(store: Store, methods: Path) -> None:
    first, second = (
        resolved("toy", r, methods_dir=methods) for r in ("lr-hi", "lr-lo")
    )
    a, b = store.add("toy", first, 14.0), store.add("toy", second, 9.0)
    for cfg in (first, second):
        mark_tuned(
            "toy",
            config_hash(cfg),
            tracking_uri=store.tracking_uri,
            methods_dir=methods,
        )
    assert (store.tag(a, "best"), store.tag(b, "best")) == ("false", "true")
    assert load_method("toy", methods).tuned == {"lr": 2.0e-4}


def test_mark_tuned_fills_only_its_registry_entrys_part(
    store: Store, methods: Path
) -> None:
    cfg = resolved("multi", "b", registry="multi-pqn-rnn", methods_dir=methods)
    store.add("multi-pqn-rnn", cfg, 3.0)
    mark_tuned(
        "multi", config_hash(cfg), tracking_uri=store.tracking_uri, methods_dir=methods
    )
    assert load_method("multi", methods).tuned == {"pqn-rnn": {"beta": 0.3}}
    with pytest.raises(ValueError, match="tuned"):
        resolved("multi", "tuned", registry="multi-ppo-rnn", methods_dir=methods)
    assert (
        resolved("multi", "tuned", registry="multi-pqn-rnn", methods_dir=methods)[
            "beta"
        ]
        == 0.3
    )


def test_mark_tuned_tags_only_the_tuning_runs_of_the_hash(
    store: Store, methods: Path
) -> None:
    cfg = resolved("toy", "lr-hi", methods_dir=methods)
    tune = store.add("toy", cfg, 14.0)
    fine = store.add("toy", cfg, 13.0, seed=1, phases="tune-coarse,tune-fine")
    final_1m = store.add("toy", cfg, 15.0, phases="final-1m")
    final_1b = store.add("toy", cfg, 16.0, phases="final-1b", budget="1b")
    mark_tuned(
        "toy", config_hash(cfg), tracking_uri=store.tracking_uri, methods_dir=methods
    )
    assert (store.tag(tune, "best"), store.tag(fine, "best")) == ("true", "true")
    assert store.tag(final_1m, "best") is None
    assert store.tag(final_1b, "best") is None


def test_mark_tuned_without_a_tuning_run_fails(store: Store, methods: Path) -> None:
    cfg = resolved("toy", "lr-hi", methods_dir=methods)
    store.add("toy", cfg, 15.0, phases="final-1m")
    with pytest.raises(ValueError, match="no tuning run"):
        mark_tuned(
            "toy",
            config_hash(cfg),
            tracking_uri=store.tracking_uri,
            methods_dir=methods,
        )


def test_mark_tuned_leaves_yaml_and_tags_alone_when_tagging_fails(
    store: Store, methods: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = resolved("toy", "lr-hi", methods_dir=methods)
    rids = [store.add("toy", cfg, 14.0, seed=i) for i in range(2)]
    before = (methods / "toy.yaml").read_text()
    real = mlflow.MlflowClient.set_tag
    calls: list[str] = []

    def flaky(self: Any, run_id: str, key: str, value: str, *a: Any, **kw: Any) -> None:
        calls.append(run_id)
        if len(calls) == 2:  # the second run: the first is already tagged
            msg = "mlflow down"
            raise RuntimeError(msg)
        real(self, run_id, key, value, *a, **kw)

    monkeypatch.setattr(mlflow.MlflowClient, "set_tag", flaky)
    with pytest.raises(RuntimeError, match="mlflow down"):
        mark_tuned(
            "toy",
            config_hash(cfg),
            tracking_uri=store.tracking_uri,
            methods_dir=methods,
        )
    monkeypatch.undo()
    assert (methods / "toy.yaml").read_text() == before
    assert [store.tag(r, "best") for r in rids] == [None, None]


def test_mark_tuned_refuses_an_unknown_hash_and_writes_nothing(
    store: Store, methods: Path
) -> None:
    before = (methods / "toy.yaml").read_text()
    with pytest.raises(ValueError, match="deadbeef"):
        mark_tuned(
            "toy", "deadbeef", tracking_uri=store.tracking_uri, methods_dir=methods
        )
    assert (methods / "toy.yaml").read_text() == before


def test_mark_tuned_refuses_a_config_the_yaml_cannot_reproduce(
    store: Store, methods: Path
) -> None:
    cfg = resolved("toy", "lr-hi", methods_dir=methods)
    del cfg["ent"]  # a tuned block can add or change keys, not delete a base key
    store.add("toy", cfg, 1.0)
    before = (methods / "toy.yaml").read_text()
    with pytest.raises(ValueError, match="reproduce"):
        mark_tuned(
            "toy",
            config_hash(cfg),
            tracking_uri=store.tracking_uri,
            methods_dir=methods,
        )
    assert (methods / "toy.yaml").read_text() == before


# --- CLI -----------------------------------------------------------------------


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "wut_murl_craftax.orchestration.queue", *args],
        capture_output=True,
        text=True,
        cwd=REPO,
        check=False,
    )


def test_dry_run_prints_one_run_name_per_line_and_starts_nothing() -> None:
    out = run_cli(
        "--phase", "configs/phases/smoke.yaml", "--method", "ppo", "--dry-run"
    )
    assert out.returncode == 0, out.stderr
    lines = out.stdout.splitlines()
    assert len(lines) == 2
    assert all(re.fullmatch(r"ppo-200k-[0-9a-f]{8}-s0", line) for line in lines)


def test_without_dry_run_the_specs_go_to_the_runner_with_the_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(runner, "main", lambda *a, **kw: calls.append((*a, kw)))
    queue.main(["--phase", "configs/phases/smoke.yaml"])
    queue.main([
        "--phase",
        "configs/phases/smoke.yaml",
        "--profile",
        "p.yaml",
        "--max-runs",
        "3",
        "--worker-index",
        "1",
        "--num-workers",
        "4",
        "--tracking-uri",
        "sqlite:///node.db",
    ])
    (specs, *flags), (_, *given) = calls
    assert [s.name for s in specs] == [
        s.name for s in expand_phase(REPO / "configs/phases/smoke.yaml")
    ]
    assert flags == [
        REPO / "configs/profiles/laptop.yaml",
        0,
        1,
        None,
        {"tracking_uri": None},
    ]
    assert given == [Path("p.yaml"), 1, 4, 3, {"tracking_uri": "sqlite:///node.db"}]


def test_cli_selects_by_registry_name_and_by_slug() -> None:
    args = (
        "--phase",
        str(PHASES / "entries.yaml"),
        "--methods-dir",
        str(METHODS),
        "--dry-run",
    )
    by_slug = run_cli(*args, "--method", "multi")
    by_name = run_cli(*args, "--method", "multi-pqn-rnn")
    assert by_slug.returncode == by_name.returncode == 0
    assert [ln.split("-1m-")[0] for ln in by_slug.stdout.splitlines()] == [
        "multi-ppo-rnn",
        "multi-pqn-rnn",
    ]
    assert [ln.split("-1m-")[0] for ln in by_name.stdout.splitlines()] == [
        "multi-pqn-rnn"
    ]


def test_cli_mark_tuned_needs_no_phase(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        queue, "mark_tuned", lambda slug, h, **_: calls.append((slug, h)) or "DIFF"
    )
    queue.main(["--mark-tuned", "ppo", "abcd1234"])
    assert calls == [("ppo", "abcd1234")]
    assert "DIFF" in capsys.readouterr().out


def test_cli_expand_fine_prints_the_configs_then_the_run_names(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(queue, "coarse_results", lambda slug, **_: stage1(slug, STAGE1))
    queue.main([
        "--phase",
        str(PHASES / "fine.yaml"),
        "--methods-dir",
        str(METHODS),
        "--expand-fine",
        "toy",
        "--dry-run",
    ])
    lines = capsys.readouterr().out.splitlines()
    configs = [ln for ln in lines if ln.startswith("#")]
    names = [ln for ln in lines if not ln.startswith("#")]
    assert (len(configs), len(names)) == (4, 12)
    assert json.loads(configs[0].split(" ", 4)[4]) == {"lr": 3.0e-4}


def test_cli_allow_partial_reaches_the_coarse_results_and_the_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[bool] = []

    def results(slug: str, **kw: Any) -> dict[str, float]:
        seen.append(kw["allow_partial"])
        return stage1(slug, {k: v for k, v in STAGE1.items() if k != "shape"})

    monkeypatch.setattr(queue, "coarse_results", results)
    argv = ["--methods-dir", str(METHODS), "--expand-fine", "toy", "--dry-run"]
    with pytest.raises(ValueError, match="shape"):
        queue.main(argv)
    queue.main([*argv, "--allow-partial"])
    assert seen == [False, True]


def test_shipped_count_bonus_fine_rule_expands_on_a_fixture_store(
    tmp_path: Path, store: Store, capsys: pytest.CaptureFixture[str]
) -> None:
    # donor tuned blocks are null until tuning ends: drop `base_from`
    raw = yaml.safe_load((REPO / "configs/methods/count-bonus.yaml").read_text())
    for block in raw["base"].values():
        block.pop("base_from")
        block["env"] = "Toy-v1"  # the donor would supply it
    methods, phases = tmp_path / "methods", tmp_path / "phases"
    methods.mkdir()
    phases.mkdir()
    (methods / "count-bonus.yaml").write_text(yaml.safe_dump(raw))
    body = {"budget": "1m", "eval": "final", "seeds": 1}
    body["runs"] = {
        r: {"config": "count-bonus", "configs": "coarse"}
        for r in ("count-bonus-ppo-rnn", "count-bonus-pqn-rnn")
    }
    (phases / "tune-coarse.yaml").write_text(yaml.safe_dump(body))
    gran = {"tiers": 1.0, "recipe": 3.0, "log2": 2.0}
    beta = {0.01: 0.0, 0.03: 0.5, 0.1: 1.0, 0.3: 0.2}
    for registry in ("count-bonus-ppo-rnn", "count-bonus-pqn-rnn"):
        for entry in raw["grid"]["coarse"]:
            cfg = resolved(
                "count-bonus", entry["name"], registry=registry, methods_dir=methods
            )
            score = gran[entry["granularity"]] + beta[entry["beta"]]
            store.add(registry, cfg, score + (entry["decay"] == "linear") * 0.1)
    results = coarse_results(
        "count-bonus",
        tracking_uri=store.tracking_uri,
        methods_dir=methods,
        phases_dir=phases,
    )
    queue._print_fine(
        expand_fine("count-bonus", results, methods_dir=methods), "count-bonus", methods
    )
    configs = [ln for ln in capsys.readouterr().out.splitlines() if ln[0] == "#"]
    assert len(configs) == 12  # 6 per variant
    first = json.loads(configs[0].split(" ", 4)[4])
    # best coarse: beta 0.1, recipe, linear; granularity spreads more than decay
    assert first == {"beta": 0.05, "granularity": "recipe", "decay": "linear"}
    assert {c.split(" ")[1] for c in configs} == {
        "count-bonus-ppo-rnn",
        "count-bonus-pqn-rnn",
    }
    assert {json.loads(c.split(" ", 4)[4])["granularity"] for c in configs} == {
        "recipe",
        "log2",
    }


# --- shipped configs -----------------------------------------------------------


def test_shipped_smoke_phase_expands_to_two_classic_ppo_runs() -> None:
    specs = expand_phase(REPO / "configs/phases/smoke.yaml", method="ppo")
    assert [(s.registry_name, s.budget, s.env, s.seed, s.eval_mode) for s in specs] == [
        ("ppo", "200k", "Craftax-Classic-Symbolic-v1", 0, "final")
    ] * 2
    assert [s.config["lr"] for s in specs] == [3.0e-4, 4.0e-4]
    assert specs[0].steps == 200_000
    assert load_method("ppo", REPO / "configs/methods").tuned is None


def test_shipped_ppo_base_is_the_craftax_1m_table_11() -> None:
    assert load_method("ppo", REPO / "configs/methods").base == {
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


def test_shipped_ppo_abort_flags_are_the_plans_thresholds() -> None:
    abort = load_method("ppo", REPO / "configs/methods").abort
    assert abort == {
        "finite": ["ppo/policy_loss", "ppo/value_loss"],
        "flags": [
            {"name": "kl_high", "metric": "ppo/approx_kl", "above": 0.1, "updates": 3},
            {
                "name": "entropy_collapse",
                "metric": "ppo/entropy",
                "below": 0.2,
                "before": 0.1,
            },
            {
                "name": "value_failure",
                "metric": "ppo/explained_variance",
                "below": 0.0,
                "after": 0.2,
                "updates": 5,
            },
        ],
    }


def test_shipped_tune_coarse_phase_is_the_28_run_ppo_stage_1() -> None:
    specs = expand_phase(REPO / "configs/phases/tune-coarse.yaml", method="ppo")
    assert len(specs) == 28
    assert {(s.registry_name, s.budget, s.env) for s in specs} == {
        ("ppo", "1m", "Craftax-Symbolic-v1")
    }
    assert {s.eval_mode for s in specs} == {"final"}
    by_config = {}
    for s in specs:
        by_config.setdefault(s.config_hash, []).append(s.seed)
    assert sorted(len(v) for v in by_config.values()) == [3] * 6 + [10]
    method = load_method("ppo", REPO / "configs/methods")
    published = resolve_config(method, "ppo", "1m", ref="published-1m")
    assert by_config[config_hash(published)] == list(range(10))
