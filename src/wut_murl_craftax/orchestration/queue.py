"""Run identity and queue expansion: method and phase YAML files become RunSpecs.

A run is one (registry entry, config, budget, seed), named
``<registry>-<budget>-<confighash8>-s<seed>``; the hash covers the fully resolved
config (not the seed, the budget or the config's label), so phases that resolve to the
same config share a run. ``configs/methods/<slug>.yaml`` holds ``base``, ``budgets``,
``grid``, ``abort`` (flag rules, see `abort`) and ``tuned`` (``load_method``); a config
resolves shallowly: donor, base, tuned or grid entry, budget, overrides
(``resolve_config``). ``configs/phases/<phase>.yaml`` picks configs and seeds
(``expand_phase``). CLI: ``--phase P [--method M] [--dry-run] [--expand-fine SLUG]``
plus the runner's flags (``--profile``, ``--max-runs``, workers, node stores).
"""

import argparse
import copy
import difflib
import hashlib
import importlib
import itertools
import json
import logging
import math
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, NoReturn, cast

import yaml

from ..config import load_experiment_config
from .tracking import _resolve_tracking_uri

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
METHODS_DIR = REPO_ROOT / "configs" / "methods"
PHASES_DIR = REPO_ROOT / "configs" / "phases"
EXPERIMENT_CONFIG = REPO_ROOT / "configs" / "experiment.yaml"
DEFAULT_PROFILE = REPO_ROOT / "configs" / "profiles" / "laptop.yaml"
TUNE_BUDGET = "1m"
"""Budget of the tuning stages; `expand_fine` resolves the stage-1 configs at it."""
SCORE_METRIC = "final_score"
"""MLflow metric holding a run's final score."""
TAG_METHOD = "method"
"""Run tag: the registry name."""
TAG_BUDGET = "budget"
"""Run tag: the budget label."""
TAG_CONFIG_HASH = "config_hash"
"""Run tag: the 8-hex config hash."""
TAG_SEED = "seed"
"""Run tag: the seed."""
TAG_PHASES = "phases"
"""Run tag: comma-joined names of the phases that list the run."""
TAG_ABORT_BUG = "abort/bug"
"""Run tag: ``true`` after a second abort; the queue skips such a run."""
TAG_EVAL_PENDING = "eval/pending"
"""Run tag: ``true`` while a reused FINISHED run's evaluation is incomplete (the runner
keeps it FINISHED through a cut evaluation); results readers count it not complete."""
TAG_FAIL_ERROR = "fail/error"
"""Run tag: the error of the last failed call (type and message, truncated)."""
TAG_BEST = "best"
"""Run tag set by `mark_tuned`: ``true`` on the tuned config's tuning runs."""
CONFIG_ARTIFACT = "config.json"
"""Run artifact holding the run's `RunSpec.config`."""
COARSE_PHASE = "tune-coarse"
TUNING_PHASES = (COARSE_PHASE, "tune-fine")
"""Phases whose runs are the 1M tuning runs `mark_tuned` may tag."""

EvalMode = Literal["final", "log_spaced"]
_EVAL_MODES = ("final", "log_spaced")
_METHOD_KEYS = {"base", "budgets", "grid", "abort", "tuned", "combine"}
_RESERVED_NAMES = {"base", "tuned", "coarse", "fine"}
_KNOB_KEYS = {"key", "mul", "add", "top", "always", "zero", "clip", "integer"}
_KNOB_AXES = ("mul", "add", "top")
_FLAG_KEYS = {"name", "metric", "above", "below", "above_median", "updates", "span"}
_FLAG_KEYS |= {"before", "after", "when"}
_LIMIT_KEYS = {"metric", "min", "max"}
_BUDGET_UNITS = {"k": 1e3, "m": 1e6, "b": 1e9}
_BUDGET_LABEL = re.compile(r"(\d+(?:\.\d+)?)([kmb])")
_HASH8 = re.compile(r"[0-9a-f]{8}")
_MISSING = object()


class QueueConfigError(ValueError):
    """Raised when a method or phase file, or a request against it, is invalid."""


def _fail(message: str) -> NoReturn:
    raise QueueConfigError(message)


class _Loader(yaml.SafeLoader):
    """SafeLoader that reads ``3e-4`` as a float (YAML 1.1 makes it a string)."""


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?[0-9][0-9_]*[eE][-+]?[0-9]+$"),
    list("-+0123456789"),
)


def _load_yaml(text: str) -> Any:
    return yaml.load(text, Loader=_Loader)  # nosec B506 - SafeLoader subclass


def _check_keys(block: object, allowed: set[str], where: str) -> dict[str, Any]:
    """Return ``block`` if it is a dict with only ``allowed`` keys, else fail."""
    if not isinstance(block, dict):
        _fail(f"{where}: expected a mapping, got {type(block).__name__}")
    extra = set(block) - allowed
    if extra:
        _fail(f"{where}: unknown key(s) {sorted(extra)}; allowed {sorted(allowed)}")
    return block


def budget_steps(label: str) -> int:
    """Env steps of a budget label: ``200k``, ``1m`` (1e6) or ``1b`` (1e9)."""
    match = _BUDGET_LABEL.fullmatch(label) if isinstance(label, str) else None
    if match is None:
        _fail(f"invalid budget label {label!r}: a number and k, m or b, e.g. 1m")
    return round(float(match[1]) * _BUDGET_UNITS[match[2]])


def _plain(value: Any) -> Any:
    """Canonical JSON-ready copy of a config: callables dropped, tuples as lists."""
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items() if not callable(v)}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value if not callable(v)]
    return value


def config_hash(config: Mapping[str, Any]) -> str:
    """First 8 hex digits of the sha256 of the canonical JSON of a resolved config."""
    text = json.dumps(
        _plain(config), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(text.encode()).hexdigest()[:8]


@dataclass(frozen=True, slots=True)
class RunSpec:
    """One run to execute: what the runner trains, evaluates and logs."""

    registry_name: str
    config_slug: str
    budget: str
    env: str
    seed: int
    config: dict[str, Any]
    config_hash: str
    phase: str
    eval_mode: EvalMode
    abort: dict[str, Any] | None = None
    """The method's ``abort`` block (flag rules for the runner), not hashed."""

    @property
    def name(self) -> str:
        """The run name, unique in the MLflow store."""
        return f"{self.registry_name}-{self.budget}-{self.config_hash}-s{self.seed}"

    @property
    def steps(self) -> int:
        """Environment steps of the budget."""
        return budget_steps(self.budget)


# --- method files ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Method:
    """A loaded and validated ``configs/methods/<slug>.yaml``."""

    slug: str
    path: Path
    base: dict[str, Any]
    budgets: dict[str, dict[str, Any]]
    coarse: tuple[dict[str, Any], ...]
    fine: dict[str, Any] | None
    abort: dict[str, Any] | None
    tuned: dict[str, Any] | None
    combine: dict[str, Any] | None = None
    """``{base, methods}`` of a combination: the control and the two registry entries
    whose hook methods its trainer composes (``training.interface.combo_trainer``)."""

    @property
    def variants(self) -> list[str] | None:
        """Variant names when ``base`` holds one block per variant, else None."""
        return (
            list(self.base)
            if all(isinstance(v, dict) for v in self.base.values())
            else None
        )

    @property
    def registry_names(self) -> list[str]:
        """Registry entries this file configures: the slug, or ``<slug>-<variant>``."""
        return (
            [self.slug]
            if self.variants is None
            else [f"{self.slug}-{v}" for v in self.variants]
        )

    def abort_rules(
        self, registry: str, budget: str | None = None
    ) -> dict[str, Any] | None:
        """The ``abort`` block of a registry entry at ``budget``.

        Its variant's block, if keyed by one; ``budgets.<budget>.abort`` (same shape,
        ``flags`` only) replaces the flags of the same name and adds the others.
        """
        rules = self._for(registry, self.abort)
        extra = self._for(registry, self.budgets.get(budget or "", {}).get("abort"))
        if not extra:
            return rules
        flags = {f["name"]: f for f in (rules or {}).get("flags", [])}
        flags |= {f["name"]: f for f in extra.get("flags", [])}
        return {**(rules or {}), "flags": list(flags.values())}

    def _for(
        self, registry: str, block: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """An abort block, or its variant's part when it is keyed by variant."""
        variant = self.variant(registry)
        if variant is None or block is None or not _by_variant(block, self.variants):
            return block
        return cast(dict[str, Any] | None, block.get(variant))

    def variant(self, registry: str) -> str | None:
        """The variant block a registry entry uses; None for a single-block method."""
        if self.variants is None:
            return None
        variant = registry.removeprefix(f"{self.slug}-")
        if variant == registry or variant not in self.variants:
            _fail(f"{self.path.name}: {registry!r} is not in {self.registry_names}")
        return variant


def _check_knob(knob: object, where: str) -> None:
    knob = _check_keys(knob, _KNOB_KEYS, where)
    if not isinstance(knob.get("key"), str) or sum(k in knob for k in _KNOB_AXES) != 1:
        _fail(f"{where}: a knob needs `key` and exactly one of `mul`, `add`, `top`")
    if "top" in knob:
        top = knob["top"]
        if not isinstance(top, int) or isinstance(top, bool) or top < 1:
            _fail(f"{where}: `top` is an int >= 1")
        if {"zero", "clip", "integer"} & knob.keys():
            _fail(f"{where}: a `top` knob takes no `zero`, `clip` or `integer`")
    if not isinstance(knob.get("always", False), bool):
        _fail(f"{where}: `always` is a bool")
    for name in ("mul", "add", "zero", "clip"):
        values = knob.get(name, [])
        if not isinstance(values, list) or not all(
            isinstance(v, int | float) for v in values
        ):
            _fail(f"{where}: `{name}` must be a list of numbers")
    if "clip" in knob and len(knob["clip"]) != 2:
        _fail(f"{where}: `clip` is [low, high]")


def _check_fine(fine: object, where: str) -> dict[str, Any]:
    fine = _check_keys(fine, {"rule", "knobs"}, f"{where} grid.fine")
    rule = _check_keys(
        fine.get("rule", {}), {"select", "combine"}, f"{where} grid.fine.rule"
    )
    select = rule.get("select", 1)
    if (
        not isinstance(select, int)
        or select < 1
        or rule.get("combine", "each") not in ("each", "cross")
    ):
        _fail(f"{where} grid.fine.rule: `select` int >= 1, `combine` each or cross")
    knobs = fine.get("knobs")
    if not isinstance(knobs, list) or not knobs:
        _fail(f"{where} grid.fine.knobs: expected a non-empty list")
    for knob in knobs:
        _check_knob(knob, f"{where} grid.fine.knobs")
    return fine


def _check_coarse(entries: object, where: str) -> tuple[dict[str, Any], ...]:
    if not isinstance(entries, list):
        _fail(f"{where} grid.coarse: expected a list")
    names: set[str] = set()
    for entry in entries:
        name = entry.get("name") if isinstance(entry, dict) else None
        if not isinstance(name, str) or name in names or name in _RESERVED_NAMES:
            _fail(f"{where} grid.coarse: `name` unique, not {name!r}/{_RESERVED_NAMES}")
        names.add(name)
    return tuple(entries)


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _check_flag(flag: object, where: str) -> str:
    """Validate one ``abort.flags`` rule and return its name."""
    flag = _check_keys(flag, _FLAG_KEYS, where)
    name = flag.get("name")
    if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9_]+", name):
        _fail(f"{where}: `name` is a lower-case word (tag flag/<name>), got {name!r}")
    if not isinstance(flag.get("metric"), str):
        _fail(f"{where} {name}: `metric` is a metric name")
    limits = [k for k in ("above", "below", "above_median") if k in flag]
    if len(limits) != 1 or not _is_number(flag[limits[0]]):
        _fail(f"{where} {name}: exactly one numeric `above`, `below` or `above_median`")
    if "above_median" in flag and flag["above_median"] <= 0:
        _fail(f"{where} {name}: `above_median` is a factor > 0")
    updates = flag.get("updates", 1)
    if not isinstance(updates, int) or isinstance(updates, bool) or updates < 1:
        _fail(f"{where} {name}: `updates` is an int >= 1")
    for key in ("span", "before", "after"):
        if key in flag and not (_is_number(flag[key]) and 0 < flag[key] <= 1):
            _fail(f"{where} {name}: `{key}` is a budget share in (0, 1]")
    if "when" in flag:
        gate = _check_keys(
            flag["when"], {"metric", "above", "below"}, f"{where} {name}"
        )
        bounds = [k for k in ("above", "below") if k in gate]
        if not isinstance(gate.get("metric"), str) or len(bounds) != 1:
            _fail(f"{where} {name}: `when` is a metric with one `above` or `below`")
        if not _is_number(gate[bounds[0]]):
            _fail(f"{where} {name}: `when` bound must be a number")
    return name


def _check_limit(limit: object, where: str) -> str:
    """Validate one ``abort.limits`` rule and return its metric."""
    limit = _check_keys(limit, _LIMIT_KEYS, where)
    metric = limit.get("metric")
    bounds = [limit[k] for k in ("min", "max") if k in limit]
    if not isinstance(metric, str) or not bounds or not all(map(_is_number, bounds)):
        _fail(f"{where}: a `metric` with a numeric `min` and/or `max`, got {limit}")
    if len(bounds) == 2 and limit["min"] > limit["max"]:
        _fail(f"{where} {metric}: `min` is above `max`")
    return metric


def _check_combine(
    block: object, base: Mapping[str, Any], where: str
) -> dict[str, Any] | None:
    """Validate a ``combine`` block: a base control and two distinct registry names."""
    if block is None:
        return None
    block = _check_keys(block, {"base", "methods"}, f"{where} combine")
    methods = block.get("methods")
    if (
        not isinstance(block.get("base"), str)
        or not isinstance(methods, list)
        or len(methods) != 2
        or not all(isinstance(m, str) for m in methods)
        or methods[0] == methods[1]
    ):
        _fail(f"{where} combine: `base` a control and `methods` two registry names")
    if base.get("base_from") != f"{block['base']}.tuned":
        _fail(f"{where} combine: `base` needs `base_from: {block['base']}.tuned`")
    return block


def _by_variant(block: object, variants: Sequence[str] | None) -> bool:
    """Whether an ``abort`` block is keyed by variant (one block per variant)."""
    return (
        isinstance(block, dict)
        and bool(block)
        and variants is not None
        and set(block) <= set(variants)
    )


def _check_abort(
    block: object, where: str, variants: Sequence[str] | None = None
) -> dict[str, Any] | None:
    if block is None:
        return None
    if _by_variant(block, variants):
        for variant, rules in cast(dict[str, Any], block).items():
            _check_abort(rules, f"{where} {variant}")
        return cast(dict[str, Any], block)
    allowed = {"flags", "finite", "permanent", "limits"}
    block = _check_keys(block, allowed, f"{where} abort")
    finite = block.get("finite", [])
    if not isinstance(finite, list) or not all(isinstance(m, str) for m in finite):
        _fail(f"{where} abort.finite: expected a list of metric names")
    permanent = block.get("permanent", [])
    if not isinstance(permanent, list) or not set(permanent) <= set(finite):
        _fail(f"{where} abort.permanent: expected a list of `finite` metric names")
    flags = block.get("flags", [])
    if not isinstance(flags, list):
        _fail(f"{where} abort.flags: expected a list")
    names = [_check_flag(f, f"{where} abort.flags") for f in flags]
    if len(set(names)) != len(names):
        _fail(f"{where} abort.flags: `name` must be unique")
    limits = block.get("limits", [])
    if not isinstance(limits, list):
        _fail(f"{where} abort.limits: expected a list")
    metrics = [_check_limit(x, f"{where} abort.limits") for x in limits]
    if len(set(metrics)) != len(metrics):
        _fail(f"{where} abort.limits: one rule per metric")
    return block


def load_method(slug: str, methods_dir: Path = METHODS_DIR) -> Method:
    """Load and validate a method file; any unknown key fails loudly.

    ``base`` is one config block, or one block per variant (every value a mapping) for a
    slug with several registry entries (``<slug>-<variant>``); a block may say
    ``base_from: <slug>.tuned``. ``abort`` of such a slug may also be keyed by
    variant, and ``budgets.<label>.abort`` changes flag rules at that budget, outside
    the hashed config (`Method.abort_rules`). A ``grid.coarse`` entry is ``name``
    plus the keys it overrides (``seeds`` is its seed count at the tuning budget, see
    `expand_phase`); ``grid.fine`` is in ``expand_fine``. ``combine`` (`Method.combine`)
    names the two methods a combination composes; it is not part of the config.
    """
    path = methods_dir / f"{slug}.yaml"
    where = path.name
    raw = _check_keys(_load_yaml(path.read_text(encoding="utf-8")), _METHOD_KEYS, where)
    base = raw.get("base")
    if not isinstance(base, dict) or not base:
        _fail(f"{where}: `base` must be a non-empty mapping")
    grid = _check_keys(raw.get("grid", {}), {"coarse", "fine"}, f"{where} grid")
    budgets = raw.get("budgets") or {}
    for label, block in budgets.items():
        budget_steps(label)
        if not isinstance(block, dict):
            _fail(f"{where} budgets.{label}: expected a mapping")
    fine = grid.get("fine")
    method = Method(
        slug=slug,
        path=path,
        base=base,
        budgets=budgets,
        coarse=_check_coarse(grid.get("coarse", []), where),
        fine=None if fine is None else _check_fine(fine, where),
        abort=raw.get("abort"),
        tuned=raw.get("tuned"),
        combine=_check_combine(raw.get("combine"), base, where),
    )
    _check_abort(method.abort, where, method.variants)
    for label, block in budgets.items():  # flags change with the budget, nothing else
        extra = block.get("abort")
        parts = extra.values() if _by_variant(extra, method.variants) else [extra]
        for part in filter(None, parts):
            _check_keys(part, {"flags"}, f"{where} budgets.{label}.abort")
        _check_abort(extra, f"{where} budgets.{label}", method.variants)
    if method.tuned is not None and method.variants is not None:
        _check_keys(method.tuned, set(method.variants), f"{where} tuned")
    return method


def _tuned_block(method: Method, registry: str) -> dict[str, Any]:
    variant = method.variant(registry)
    block = (
        method.tuned
        if variant is None or method.tuned is None
        else method.tuned.get(variant)
    )
    if block is None:
        label = f"{method.slug}.tuned" + ("" if variant is None else f" ({variant})")
        _fail(f"{label} is null; tune, then `--mark-tuned {method.slug} <hash8>`")
    return block


def _overrides(entry: Mapping[str, Any]) -> dict[str, Any]:
    """The keys a grid entry overrides (its name and seed count are not config)."""
    return {k: v for k, v in entry.items() if k not in ("name", "seeds")}


def resolve_config(
    method: Method,
    registry: str,
    budget: str,
    *,
    ref: str = "base",
    overrides: Mapping[str, Any] | None = None,
    methods_dir: Path = METHODS_DIR,
    _chain: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Resolve a config: donor, base, ``ref``, ``budgets[budget]``, overrides.

    ``ref`` is ``base``, ``tuned`` or a grid entry. A ``base_from: <slug>.tuned`` donor
    is resolved for the same budget (its tuned block, then its ``budgets``) and the
    block's own keys override it. The key ``method`` is the registry name.
    """
    variant = method.variant(registry)
    block = method.base if variant is None else method.base[variant]
    cfg: dict[str, Any] = {}
    donor = block.get("base_from")
    if donor is not None:
        slug, _, attr = str(donor).rpartition(".")
        if attr != "tuned" or not slug or slug in (*_chain, method.slug):
            _fail(
                f"{method.path.name}: base_from is `<other slug>.tuned`, got {donor!r}"
            )
        parent = load_method(slug, methods_dir)
        cfg = resolve_config(
            parent,
            slug,
            budget,
            ref="tuned",
            methods_dir=methods_dir,
            _chain=(*_chain, method.slug),
        )
    cfg |= {k: v for k, v in block.items() if k != "base_from"}
    if ref == "tuned":
        cfg |= _tuned_block(method, registry)
    elif ref != "base":
        entry = next((e for e in method.coarse if e["name"] == ref), None)
        if entry is None:
            names = [e["name"] for e in method.coarse]
            _fail(
                f"{method.path.name}: unknown config {ref!r}; use base, tuned, {names}"
            )
        cfg |= _overrides(entry)
    cfg |= {k: v for k, v in method.budgets.get(budget, {}).items() if k != "abort"}
    cfg |= overrides or {}
    cfg["method"] = registry
    return copy.deepcopy(cfg)


# --- phase files ----------------------------------------------------------------


def _seed_list(spec: object, where: str) -> list[int]:
    if isinstance(spec, int) and not isinstance(spec, bool) and spec > 0:
        return list(range(spec))
    if isinstance(spec, list) and spec and all(isinstance(s, int) for s in spec):
        return list(spec)
    return _fail(f"{where}: seeds is a positive count or a list of ints, got {spec!r}")


Fine = Mapping[str, Mapping[str, Mapping[str, Any]]]
"""Generated stage-2 configs: registry name -> config name -> overrides."""


def _entry_items(
    entry: Mapping[str, Any], method: Method, registry: str, fine: Fine | None
) -> list[tuple[str, dict[str, Any], Any, Any]]:
    """Items (ref, overrides, item seeds, grid-entry seeds) of ``configs``."""
    configs = entry.get("configs", ["tuned"])
    grid_seeds = {e["name"]: e.get("seeds") for e in method.coarse}
    if configs == "coarse":
        return [(name, {}, None, seeds) for name, seeds in grid_seeds.items()]
    if configs == "fine":
        if not fine or registry not in fine:
            _fail(f"{registry}: `configs: fine` needs --expand-fine {method.slug}")
        return [("base", dict(over), None, None) for over in fine[registry].values()]
    if not isinstance(configs, list):
        _fail(f"{registry}: `configs` is a list, `coarse` or `fine`")
    items = []
    for item in configs:
        spec = {"ref": item} if isinstance(item, str) else item
        spec = _check_keys(spec, {"ref", "set", "seeds"}, f"{registry} configs")
        ref = spec.get("ref")
        if not isinstance(ref, str):
            _fail(f"{registry} configs: every item needs a `ref`")
        items.append((
            ref,
            dict(spec.get("set", {})),
            spec.get("seeds"),
            grid_seeds.get(ref),
        ))
    return items


def _entry_specs(
    raw: Mapping[str, Any], path: Path, registry: str, entry: Any, **kw: Any
) -> list[RunSpec]:
    """RunSpecs of one phase entry; ``kw``: expand_phase method, methods_dir, fine."""
    where = f"{path.name} runs.{registry}"
    entry = _check_keys(entry or {}, {"config", "configs", "seeds"}, where)
    slug: str = entry.get("config", registry)
    if kw["method"] not in (None, registry, slug):
        return []
    method = load_method(slug, kw["methods_dir"])
    env = {"env": raw["env"]} if "env" in raw else {}
    specs = []
    for ref, over, item_seeds, grid_seeds in _entry_items(
        entry, method, registry, kw["fine"]
    ):
        if raw["budget"] != TUNE_BUDGET:  # review M13: tuning seed counts only
            grid_seeds = None
        order = (item_seeds, entry.get("seeds"), grid_seeds, raw.get("seeds"))
        seeds = next((s for s in order if s is not None), None)
        cfg = resolve_config(
            method,
            registry,
            raw["budget"],
            ref=ref,
            overrides=env | over,
            methods_dir=kw["methods_dir"],
        )
        if "env" not in cfg:
            _fail(
                f"{where}: config {ref} has no `env`; set it in the method or the phase"
            )
        specs += [
            RunSpec(
                registry,
                slug,
                raw["budget"],
                cfg["env"],
                seed,
                cfg,
                config_hash(cfg),
                path.stem,
                raw["eval"],
                method.abort_rules(registry, raw["budget"]),
            )
            for seed in _seed_list(seeds, f"{where} config {ref}")
        ]
    return specs


def expand_phase(
    path: Path,
    *,
    method: str | None = None,
    methods_dir: Path = METHODS_DIR,
    fine: Fine | None = None,
    skip_fine: bool = False,
) -> list[RunSpec]:
    """Expand a phase file into RunSpecs, one per (entry, config, seed).

    Keys: ``budget`` (``200k``, ``1m``, ``1b``), ``eval`` (``final`` or ``log_spaced``),
    optional ``env`` (replaces every config's env) and ``seeds`` (a count or a list,
    the default), and ``runs``: registry name -> ``{config: <slug, default the name>,
    configs: <list | coarse | fine>, seeds}``. A ``configs`` item is ``base``, ``tuned``
    (the default list) or a grid entry name, or ``{ref, set: {overrides}, seeds}``;
    ``fine`` takes the configs of ``expand_fine``. Seeds come from the item, the entry,
    the grid entry, then the phase. A grid entry's ``seeds`` is its tuning seed count
    and counts only at the tuning budget (`TUNE_BUDGET`): a ``final-1b`` phase of
    ``published-1m`` (10 tuning seeds) runs the phase's ``seeds``. ``method`` keeps
    the entries whose registry name or config slug equals it. Without ``fine``,
    ``skip_fine`` leaves out the ``configs: fine`` entries (`fine_registries`) instead
    of failing.
    """
    keys = {"budget", "eval", "env", "seeds", "runs"}
    raw = _check_keys(_load_yaml(path.read_text(encoding="utf-8")), keys, path.name)
    raw["budget"] = str(raw.get("budget"))
    budget_steps(raw["budget"])
    if raw.get("eval") not in _EVAL_MODES:
        _fail(f"{path.name}: eval is one of {_EVAL_MODES}, got {raw.get('eval')!r}")
    runs = raw.get("runs")
    if not isinstance(runs, dict) or not runs:
        _fail(f"{path.name}: `runs` must be a non-empty mapping")
    skipped = fine_registries(path, method) if skip_fine and fine is None else []
    specs: dict[str, RunSpec] = {}
    for registry, entry in runs.items():
        if registry in skipped:
            continue
        for spec in _entry_specs(
            raw,
            path,
            registry,
            entry,
            method=method,
            methods_dir=methods_dir,
            fine=fine,
        ):
            if spec.name in specs:
                _fail(f"{path.name}: duplicate run {spec.name}")
            specs[spec.name] = spec
    if not specs and not skipped:
        _fail(f"{path.name}: no phase entry matches --method {method!r}")
    return list(specs.values())


def fine_registries(path: Path, method: str | None = None) -> list[str]:
    """Registry names of the phase's ``configs: fine`` entries that ``method`` keeps."""
    runs = _load_yaml(path.read_text(encoding="utf-8")).get("runs") or {}
    return [
        registry
        for registry, entry in runs.items()
        if isinstance(entry, dict)
        and entry.get("configs") == "fine"
        and method in (None, registry, entry.get("config", registry))
    ]


# --- fine rule ------------------------------------------------------------------


def _axis(method: Method, key: str) -> list[str]:
    """Coarse entries varying only ``key`` plus the unmodified one; all if none do."""
    pure = [e for e in method.coarse if _overrides(e).keys() == {key}]
    if not pure:
        return [e["name"] for e in method.coarse]
    return [e["name"] for e in method.coarse if _overrides(e).keys() <= {key}]


def _effect(
    method: Method, cfgs: Mapping[str, Any], scores: Mapping[str, float], key: str
) -> float:
    """Range of the mean stage-1 score over the values of ``key`` along its axis."""
    groups: dict[str, list[float]] = {}
    for name in (n for n in _axis(method, key) if n in scores):
        if key not in cfgs[name]:
            _fail(f"{method.path.name}: fine knob {key!r} is not a key of the config")
        groups.setdefault(json.dumps(cfgs[name][key]), []).append(scores[name])
    means = [sum(v) / len(v) for v in groups.values()]
    return max(means) - min(means)


def _knob_values(knob: Mapping[str, Any], best: Any) -> list[Any]:
    """Fine values of a knob around the best coarse value."""
    if isinstance(best, bool) or not isinstance(best, int | float):
        _fail(f"fine knob {knob['key']!r} is not numeric (best value {best!r})")
    if best == 0 and "zero" in knob:
        values = list(knob["zero"])
    elif "mul" in knob:
        values = [best * f for f in knob["mul"]]
    else:
        values = [best + a for a in knob["add"]]
    if "clip" in knob:
        low, high = knob["clip"]
        values = [min(max(v, low), high) for v in values]
    # 12 digits: 4e-4 x 0.75 must hash like a hand-written 3e-4
    values = [round(v) if knob.get("integer") else float(f"{v:.12g}") for v in values]
    return list(dict.fromkeys(values))


def _top_values(
    method: Method,
    cfgs: Mapping[str, Any],
    scores: Mapping[str, float],
    key: str,
    top: int,
    best: Any,
) -> list[Any]:
    """The stage-1 best config's value, then the knob's next values by mean score.

    ``best`` comes first, so the best config stays in the fine grid even when its
    value's mean ranks low; the other ``top - 1`` values rank by mean stage-1 score
    along the knob's axis, ties in the order of first appearance in ``grid.coarse``.
    """
    groups: dict[str, list[float]] = {}
    values: dict[str, Any] = {}
    for name in (n for n in _axis(method, key) if n in scores):
        if key not in cfgs[name]:
            _fail(f"{method.path.name}: fine knob {key!r} is not a key of the config")
        label = json.dumps(cfgs[name][key])
        values.setdefault(label, cfgs[name][key])
        groups.setdefault(label, []).append(scores[name])
    first = json.dumps(best)
    ranked = sorted(groups, key=lambda g: -sum(groups[g]) / len(groups[g]))
    return [best, *[values[g] for g in ranked if g != first][: top - 1]]


def _fine_for(
    method: Method,
    registry: str,
    results: Mapping[str, float],
    methods_dir: Path,
    allow_partial: bool,
) -> dict[str, dict[str, Any]]:
    cfgs = {
        e["name"]: resolve_config(
            method, registry, TUNE_BUDGET, ref=e["name"], methods_dir=methods_dir
        )
        for e in method.coarse
    }
    missing = [n for n, c in cfgs.items() if config_hash(c) not in results]
    if allow_partial:
        cfgs = {n: c for n, c in cfgs.items() if n not in missing}
    if (missing and not allow_partial) or not cfgs:
        _fail(
            f"{registry}: no stage-1 result for coarse config(s) {missing or 'at all'}"
        )
    scores = {n: results[config_hash(c)] for n, c in cfgs.items()}
    best = max(scores, key=lambda n: scores[n])
    fine = method.fine or {}
    rule = fine.get("rule", {})
    knobs = {k["key"]: k for k in fine["knobs"]}
    always = [k for k in knobs if knobs[k].get("always")]
    ranked = [k for k in knobs if k not in always]
    effects = {key: _effect(method, cfgs, scores, key) for key in ranked}
    ranked = sorted(ranked, key=lambda k: -effects[k])[
        : rule.get("select", len(ranked))
    ]
    chosen = always + ranked
    values = {
        key: (
            _top_values(method, cfgs, scores, key, knobs[key]["top"], cfgs[best][key])
            if "top" in knobs[key]
            else _knob_values(knobs[key], cfgs[best][key])
        )
        for key in chosen
    }
    if rule.get("combine", "each") == "cross":
        combos = [
            dict(zip(chosen, c, strict=True))
            for c in itertools.product(*values.values())
        ]
    else:
        combos = [{key: v} for key in chosen for v in values[key]]
    entry = next(e for e in method.coarse if e["name"] == best)
    out: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for combo in combos:
        over = _overrides(entry) | combo
        digest = config_hash(
            resolve_config(
                method, registry, TUNE_BUDGET, overrides=over, methods_dir=methods_dir
            )
        )
        if digest not in seen:
            seen.add(digest)
            out["fine-" + "-".join(f"{k}={v}" for k, v in combo.items())] = over
    return out


def expand_fine(
    slug: str,
    results: Mapping[str, float],
    *,
    method: str | None = None,
    methods_dir: Path = METHODS_DIR,
    allow_partial: bool = False,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Generate the stage-2 configs: registry name -> {config name: overrides}.

    ``results`` maps a stage-1 config hash to its mean final score over seeds. Per
    registry entry: the best coarse config; the ``rule.select`` knobs (default all) with
    the largest effect, the range of mean scores over the knob's values along its axis
    (the entries varying only that knob plus the unmodified one; all entries if none
    vary only it); each knob's ``mul`` factors or ``add`` offsets around the best value
    (``zero`` replaces them when it is 0; ``clip`` [low, high]; ``integer``) or, for a
    categorical knob, the best config's value and its ``top - 1`` next values by mean
    stage-1 score along its axis, one knob at a time (``combine: each``) or all combined
    (``cross``). A knob with ``always: true`` is always chosen and ``select`` ranks the
    others only. A config equal to the best coarse one hashes like it, so its stage-1
    run is reused. ``method`` restricts the registry entries (a registry name, or the
    slug for all). A coarse config without a result fails, unless ``allow_partial``
    ranks the others only.
    """
    loaded = load_method(slug, methods_dir)
    if loaded.fine is None:
        _fail(f"{loaded.path.name} has no grid.fine rule")
    names = [r for r in loaded.registry_names if method in (None, slug, r)]
    if not names:
        _fail(f"--method {method!r} is none of {loaded.registry_names}")
    return {r: _fine_for(loaded, r, results, methods_dir, allow_partial) for r in names}


# --- MLflow ---------------------------------------------------------------------


def _client(tracking_uri: str | None) -> Any:
    # mlflow takes seconds to import; the dry run and the expansion tests never need it
    mlflow = importlib.import_module("mlflow")
    uri = tracking_uri or load_experiment_config(EXPERIMENT_CONFIG).mlflow.tracking_uri
    return mlflow.MlflowClient(tracking_uri=_resolve_tracking_uri(uri))


def _runs(client: Any, filter_string: str) -> list[Any]:
    """All runs of every experiment that match an MLflow filter."""
    ids = [e.experiment_id for e in client.search_experiments()]
    runs: list[Any] = []
    token = None
    while True:
        page = client.search_runs(
            ids, filter_string, max_results=1000, page_token=token
        )
        runs += list(page)
        token = page.token
        if not token:
            return runs


def coarse_results(
    slug: str,
    *,
    method: str | None = None,
    tracking_uri: str | None = None,
    methods_dir: Path = METHODS_DIR,
    phases_dir: Path | None = None,
    allow_partial: bool = False,
) -> dict[str, float]:
    """Mean final score per config hash over its ``tune-coarse`` seeds, complete only.

    A config is complete when every run the phase file lists for it is FINISHED in
    the store with a finite ``final_score`` and is not marked ``abort/bug`` or
    ``eval/pending``.

    Raises:
        ValueError: Naming each incomplete config's missing runs, unless
            ``allow_partial`` leaves those configs out with a warning.
    """
    names = [
        r
        for r in load_method(slug, methods_dir).registry_names
        if method in (None, slug, r)
    ]
    expected: dict[str, list[str]] = {}
    path = (phases_dir or PHASES_DIR) / f"{COARSE_PHASE}.yaml"
    for spec in expand_phase(path, method=slug, methods_dir=methods_dir):
        if spec.registry_name in names:
            expected.setdefault(spec.config_hash, []).append(spec.name)
    done: dict[str, float] = {}
    for run in _runs(
        _client(tracking_uri),
        f"attributes.status = 'FINISHED' and tags.{TAG_PHASES} LIKE '%{COARSE_PHASE}%'",
    ):
        tags, score = run.data.tags, run.data.metrics.get(SCORE_METRIC, math.nan)
        if (
            COARSE_PHASE in tags[TAG_PHASES].split(",")
            and tags.get(TAG_ABORT_BUG) != "true"
            and tags.get(TAG_EVAL_PENDING) != "true"
            and math.isfinite(score)
        ):
            done[run.info.run_name] = score
    missing = {h: [n for n in r if n not in done] for h, r in expected.items()}
    missing = {h: runs for h, runs in missing.items() if runs}
    if missing:
        listing = "; ".join(f"{h}: {', '.join(runs)}" for h, runs in missing.items())
        if not allow_partial:
            _fail(f"incomplete {COARSE_PHASE} configs (see --allow-partial): {listing}")
        logger.warning("left out incomplete %s configs: %s", COARSE_PHASE, listing)
    return {
        h: sum(done[n] for n in runs) / len(runs)
        for h, runs in expected.items()
        if h not in missing
    }


def _replace_tuned(text: str, tuned: Mapping[str, Any]) -> str:
    """Replace the top-level ``tuned:`` block of a method file; the rest stays as is."""
    block = yaml.safe_dump(
        {"tuned": dict(tuned)}, sort_keys=False, default_flow_style=False
    )
    lines = text.splitlines(keepends=True)
    start = next((i for i, ln in enumerate(lines) if ln.startswith("tuned:")), None)
    if start is None:
        return text.rstrip("\n") + "\n" + block
    end = start + 1
    while end < len(lines) and (not lines[end].strip() or lines[end][0] in " \t"):
        end += 1
    while end > start + 1 and not lines[end - 1].strip():
        end -= 1
    return "".join(lines[:start]) + block + "".join(lines[end:])


def _set_best(
    client: Any, changes: list[tuple[Any, str]], then: Callable[[], Any]
) -> None:
    """Set ``best`` per (run, value), then call ``then``; on error restore the tags."""
    done: list[tuple[Any, str | None]] = []
    try:
        for run, value in changes:
            old = run.data.tags.get(TAG_BEST)
            client.set_tag(run.info.run_id, TAG_BEST, value)
            done.append((run, old))
        then()
    except Exception:
        for run, old in reversed(done):
            if old is None:
                client.delete_tag(run.info.run_id, TAG_BEST)
            else:
                client.set_tag(run.info.run_id, TAG_BEST, old)
        raise


def mark_tuned(
    slug: str,
    config_hash8: str,
    *,
    tracking_uri: str | None = None,
    methods_dir: Path = METHODS_DIR,
) -> str:
    """Write the ``tuned:`` block of a config, tag its runs ``best``, return the diff.

    The runs of ``config_hash8`` supply the config (artifact ``config.json``); the block
    holds the keys that differ from the resolved base and must resolve back to the same
    hash. The registry entry comes from the runs' ``method`` tag, so a slug with
    variants fills only that variant's part. Only the tuning runs (phases tune-coarse or
    tune-fine) of the hash are tagged; re-marking moves ``best`` to the new ones.
    Tags are set first; if tagging or the write fails, the tags are restored and the
    file is left as it was.
    """
    if not _HASH8.fullmatch(config_hash8):
        _fail(f"{config_hash8!r} is not a config hash (8 hex digits)")
    method = load_method(slug, methods_dir)
    client = _client(tracking_uri)
    runs = [
        r
        for r in _runs(client, f"tags.{TAG_CONFIG_HASH} = '{config_hash8}'")
        if r.data.tags.get(TAG_METHOD) in method.registry_names
        and set(r.data.tags.get(TAG_PHASES, "").split(",")) & set(TUNING_PHASES)
    ]
    if not runs:
        _fail(
            f"no tuning run ({TUNING_PHASES}) of {slug} has config hash {config_hash8}"
        )
    registry, budget = runs[0].data.tags[TAG_METHOD], runs[0].data.tags[TAG_BUDGET]
    with tempfile.TemporaryDirectory() as tmp:
        cfg = json.loads(
            Path(
                client.download_artifacts(runs[0].info.run_id, CONFIG_ARTIFACT, tmp)
            ).read_text()
        )
    base = resolve_config(method, registry, budget, methods_dir=methods_dir)
    block = {
        k: v for k, v in cfg.items() if k != "method" and base.get(k, _MISSING) != v
    }
    variant = method.variant(registry)
    tuned = block if variant is None else {**(method.tuned or {}), variant: block}
    if (
        config_hash(
            resolve_config(
                replace(method, tuned=tuned),
                registry,
                budget,
                ref="tuned",
                methods_dir=methods_dir,
            )
        )
        != config_hash8
    ):
        _fail(f"tuned block {block} would not reproduce config {config_hash8}")
    old = method.path.read_text(encoding="utf-8")
    new = _replace_tuned(old, tuned)
    if _load_yaml(new)["tuned"] != tuned:
        _fail(f"could not rewrite the tuned block of {method.path.name}")
    stale = _runs(
        client, f"tags.{TAG_BEST} = 'true' and tags.{TAG_METHOD} = '{registry}'"
    )
    changes = [
        (r, "false") for r in stale if r.data.tags[TAG_CONFIG_HASH] != config_hash8
    ]
    changes += [(r, "true") for r in runs]
    _set_best(client, changes, lambda: method.path.write_text(new, encoding="utf-8"))
    name = method.path.name
    return "".join(
        difflib.unified_diff(
            old.splitlines(True), new.splitlines(True), f"a/{name}", f"b/{name}"
        )
    )


# --- CLI ------------------------------------------------------------------------


def run_specs(specs: Sequence[RunSpec], args: argparse.Namespace) -> None:
    """Hand the specs to the runner with the execution flags."""
    # imported here: runner imports this module (a cycle), plus JAX and MLflow
    from .runner import main as run

    run(
        specs,
        args.profile,
        args.worker_index,
        args.num_workers,
        args.max_runs,
        tracking_uri=args.tracking_uri,
        seed_from=args.seed_from,
    )


def _print_fine(fine: Fine, slug: str, methods_dir: Path) -> None:
    """Print each stage-2 config as ``# <registry> <name> <hash8> <overrides json>``."""
    method = load_method(slug, methods_dir)
    for registry, configs in fine.items():
        for name, over in configs.items():
            cfg = resolve_config(
                method, registry, TUNE_BUDGET, overrides=over, methods_dir=methods_dir
            )
            print(f"# {registry} {name} {config_hash(cfg)} {json.dumps(over)}")


def phase_specs(
    args: argparse.Namespace,
    tracking_uri: str | None = None,
    *,
    skip_fine: bool = False,
) -> list[RunSpec] | None:
    """The specs of ``args.phase`` (None without one), after ``args.expand_fine``.

    ``args`` carries ``phase``, ``method``, ``expand_fine``, ``allow_partial`` and
    ``methods_dir``; the stage-2 configs come from the coarse results in
    ``tracking_uri`` (the default store when None) and are printed. ``skip_fine``:
    see `expand_phase`.
    """
    fine = None
    if args.expand_fine:
        slug = args.expand_fine
        partial = args.allow_partial
        results = coarse_results(
            slug,
            method=args.method,
            tracking_uri=tracking_uri,
            methods_dir=args.methods_dir,
            allow_partial=partial,
        )
        fine = expand_fine(
            slug,
            results,
            method=args.method,
            methods_dir=args.methods_dir,
            allow_partial=partial,
        )
        _print_fine(fine, slug, args.methods_dir)
    if args.phase is None:
        return None
    return expand_phase(
        args.phase,
        method=args.method,
        methods_dir=args.methods_dir,
        fine=fine,
        skip_fine=skip_fine,
    )


def _sqlite_uri(store: str) -> str:
    """A SQLite file given as a plain path, as an absolute ``sqlite:///`` URI."""
    if "://" in store or store.startswith("sqlite:"):
        return store
    return f"sqlite:///{Path(store).resolve()}"


def main(argv: Sequence[str] | None = None) -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--phase", type=Path, help="configs/phases/<phase>.yaml")
    parser.add_argument("--method", help="registry name or config slug to keep")
    parser.add_argument(
        "--dry-run", action="store_true", help="print run names, start nothing"
    )
    parser.add_argument(
        "--expand-fine", metavar="SLUG", help="generate the stage-2 configs of SLUG"
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="--expand-fine ranks only the coarse configs with all seeds finished",
    )
    parser.add_argument("--mark-tuned", nargs=2, metavar=("SLUG", "HASH8"))
    parser.add_argument("--methods-dir", type=Path, default=METHODS_DIR)
    parser.add_argument(
        "--profile", type=Path, default=DEFAULT_PROFILE, help="machine profile YAML"
    )
    parser.add_argument("--max-runs", type=int, help="train at most N runs")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument(
        "--num-workers", type=int, default=1, help="worker i takes runs i, i + k, ..."
    )
    parser.add_argument(
        "--tracking-uri",
        help="store of this worker's runs instead of configs/experiment.yaml's "
        "(a cluster node's own database, merged later by `make merge-mlflow`)",
    )
    parser.add_argument(
        "--seed-from",
        metavar="DB",
        help="read-only copy of the main store: this worker's runs in it are merged "
        "into --tracking-uri's store first, and --expand-fine ranks its coarse runs",
    )
    args = parser.parse_args(argv)
    if args.seed_from and not args.tracking_uri:
        parser.error("--seed-from needs --tracking-uri (the node store to seed)")
    # review p2b I2: MLflow reads a plain path as a file-store directory
    if args.seed_from:
        args.seed_from = _sqlite_uri(args.seed_from)
    if (args.tracking_uri or "").endswith(".db"):
        args.tracking_uri = _sqlite_uri(args.tracking_uri)
    if args.mark_tuned:
        tuned = mark_tuned(
            *args.mark_tuned,
            tracking_uri=args.tracking_uri,
            methods_dir=args.methods_dir,
        )
        print(tuned)
        return
    if args.phase is None and args.expand_fine is None:
        parser.error("give --phase, --expand-fine or --mark-tuned")
    # review M7: the coarse results come from the store the flags name
    specs = phase_specs(args, args.seed_from or args.tracking_uri)
    if specs is None:
        return
    if args.dry_run:
        print("\n".join(spec.name for spec in specs))
    else:
        run_specs(specs, args)


if __name__ == "__main__":
    main()
