"""Combination pair from final-1m (Phase 3 Task 1 Step 1, ruling R71) into combo.yaml.

Each method's tuned config in the phase (ablation items left out) scores the IQM of
its runs' final scores. Methods rank by IQM, ties by the probability of improvement
over their base control (the end of the ``base_from`` chain), then by name. The pair
is the compatible one of the smallest rank sum (ties: the better best rank): two hook
methods on one control that set no knob two ways. Skill methods (`SKILL_METHODS`) are
left out with a logged reason: ``combine`` composes hook methods only. Exit status 1
on unfinished tuned runs or without a compatible pair. Usage: ``python -m
wut_murl_craftax.evaluation.select_pair --phase final-1m [--dry-run]``
"""

import argparse
import itertools
import logging
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..orchestration.queue import (
    METHODS_DIR,
    PHASES_DIR,
    SCORE_METRIC,
    TAG_ABORT_BUG,
    TAG_EVAL_PENDING,
    TAG_PHASES,
    TUNE_BUDGET,
    Method,
    _client,
    _runs,
    _sqlite_uri,
    config_hash,
    expand_phase,
    load_method,
    resolve_config,
)
from .stats import iqm, probability_of_improvement

logger = logging.getLogger(__name__)

SKILL_METHODS = ("smdp-skills", "dyna-q")
"""Registry entries that are not hook methods: ``combine`` cannot compose them."""
COMBO = "combo"
"""Slug and registry name of the combination."""
HEADER = """\
# Combination of the two best compatible methods (Phase 3 Task 1); the schema is
# documented in orchestration/queue.py, the composition in training/interface.py
# (`combine`).
#
# Written by `python -m wut_murl_craftax.evaluation.select_pair --phase {phase}`
# (ruling R71): the methods ranked by the IQM of their tuned config's final scores,
# ties by the probability of improvement (PoI) over their base control; the pair is
# the best compatible one (one control, hook methods only, no knob set two ways).
# `base` is the control's tuned config plus each method's tuned knobs (its `base` keys
# with its `tuned` block applied); `abort` and `budgets` join both methods' rules. No
# retuning: one config, `combined`.
#
# Ranking (rank, registry, control, seeds, IQM, PoI over the control):
"""


class NoPairError(ValueError):
    """No two ranked methods can be combined."""


@dataclass(frozen=True, slots=True)
class Ranked:
    """A method's place in the ranking."""

    registry: str
    control: str
    n: int
    iqm: float
    poi: float
    """Probability of improvement over ``control``."""


def _entries(methods_dir: Path) -> dict[str, Method]:
    """Registry name -> its method file."""
    entries: dict[str, Method] = {}
    for path in sorted(methods_dir.glob("*.yaml")):
        method = load_method(path.stem, methods_dir)
        entries |= dict.fromkeys(method.registry_names, method)
    return entries


def _block(method: Method, registry: str) -> dict[str, Any]:
    variant = method.variant(registry)
    return method.base if variant is None else method.base[variant]


def controls_of(methods_dir: Path = METHODS_DIR) -> dict[str, str]:
    """Registry name -> the control its ``base_from`` chain ends at (itself if none)."""
    entries = _entries(methods_dir)
    out: dict[str, str] = {}
    for registry in entries:
        name = registry
        while (donor := _block(entries[name], name).get("base_from")) is not None:
            name = str(donor).rpartition(".")[0]
        out[registry] = name
    return out


def _tuned_runs(phase: Path, methods_dir: Path) -> dict[str, list[str]]:
    """Registry name -> the phase's run names of its tuned config."""
    raw = yaml.safe_load(phase.read_text(encoding="utf-8"))
    env = {"env": raw["env"]} if "env" in raw else {}
    hashes: dict[str, str] = {}
    names: dict[str, list[str]] = {}
    for spec in expand_phase(phase, methods_dir=methods_dir):
        registry = spec.registry_name
        if registry not in hashes:
            method = load_method(spec.config_slug, methods_dir)
            cfg = resolve_config(
                method,
                registry,
                spec.budget,
                ref="tuned",
                overrides=env,
                methods_dir=methods_dir,
            )
            hashes[registry] = config_hash(cfg)
        if spec.config_hash == hashes[registry]:
            names.setdefault(registry, []).append(spec.name)
    return names


def final_scores(
    phase: Path,
    *,
    tracking_uri: str | None = None,
    methods_dir: Path = METHODS_DIR,
    allow_partial: bool = False,
) -> dict[str, list[float]]:
    """Registry name -> the final scores of its tuned config's runs in ``phase``.

    A run counts when it is FINISHED in the phase (tag ``phases``) with a finite
    ``final_score``, not tagged ``abort/bug`` or ``eval/pending``.

    Raises:
        ValueError: Naming every tuned run that does not count, unless
            ``allow_partial`` leaves them out with a warning.
    """
    expected = _tuned_runs(phase, methods_dir)
    done: dict[str, float] = {}
    query = (
        f"attributes.status = 'FINISHED' and tags.{TAG_PHASES} LIKE '%{phase.stem}%'"
    )
    for run in _runs(_client(tracking_uri), query):
        tags, score = run.data.tags, run.data.metrics.get(SCORE_METRIC, math.nan)
        if (
            phase.stem in tags.get(TAG_PHASES, "").split(",")
            and tags.get(TAG_ABORT_BUG) != "true"
            and tags.get(TAG_EVAL_PENDING) != "true"
            and math.isfinite(score)
        ):
            done[run.info.run_name] = score
    gaps = []
    for registry, names in expected.items():
        if missing := [n for n in names if n not in done]:
            n_done = len(names) - len(missing)
            gaps.append(
                f"{registry}: {n_done} of {len(names)} tuned runs finished in "
                f"{phase.stem}, not {', '.join(missing)}"
            )
    if gaps and not allow_partial:
        raise ValueError("; ".join(gaps))
    if gaps:
        logger.warning("ranking without the unfinished runs: %s", "; ".join(gaps))
    scores = {r: [done[n] for n in names if n in done] for r, names in expected.items()}
    return {r: s for r, s in scores.items() if s}


def rank_methods(
    scores: Mapping[str, Sequence[float]],
    *,
    controls: Mapping[str, str] | None = None,
    methods_dir: Path = METHODS_DIR,
) -> list[Ranked]:
    """The methods among ``scores``, best first; ``controls``: method -> its control.

    Raises:
        ValueError: If a method's control has no scores.
    """
    if controls is None:
        controls = {r: c for r, c in controls_of(methods_dir).items() if r != c}
    ranked = []
    for registry, control in controls.items():
        if registry not in scores:
            continue
        if control not in scores:
            msg = f"{registry}: its control {control} has no final scores"
            raise ValueError(msg)
        x = scores[registry]
        poi = probability_of_improvement(x, scores[control])
        ranked.append(Ranked(registry, control, len(x), iqm(x), poi))
    return sorted(ranked, key=lambda r: (-r.iqm, -r.poi, r.registry))


def method_knobs(registry: str, methods_dir: Path = METHODS_DIR) -> dict[str, Any]:
    """A method's own ``base`` keys and ``tuned`` block keys, at their tuned values."""
    method = _entries(methods_dir)[registry]
    cfg = resolve_config(
        method, registry, TUNE_BUDGET, ref="tuned", methods_dir=methods_dir
    )
    variant = method.variant(registry)
    tuned = method.tuned or {}
    tuned = tuned if variant is None else tuned.get(variant) or {}
    keys = dict.fromkeys([*_block(method, registry), *tuned])
    return {k: cfg[k] for k in keys if k != "base_from"}


def _clash(a: str, b: str, methods_dir: Path) -> list[str]:
    """Knobs the two methods set to different values."""
    ka, kb = method_knobs(a, methods_dir), method_knobs(b, methods_dir)
    return sorted(k for k in ka.keys() & kb.keys() if ka[k] != kb[k])


def choose_pair(
    ranking: Sequence[Ranked], *, methods_dir: Path = METHODS_DIR
) -> tuple[Ranked, Ranked]:
    """The best compatible pair of ``ranking`` (best first), in ranking order.

    Raises:
        NoPairError: If no two hook methods share a control without a knob clash.
    """
    hooks = []
    for r in ranking:
        if r.registry in SKILL_METHODS:
            logger.info(
                "%s: skill method, left out (combine composes hook methods only)",
                r.registry,
            )
        else:
            hooks.append(r)
    place = {r.registry: i for i, r in enumerate(ranking)}
    pairs = sorted(
        itertools.combinations(hooks, 2),
        key=lambda p: (
            place[p[0].registry] + place[p[1].registry],
            place[p[0].registry],
        ),
    )
    reasons = []
    for a, b in pairs:
        if a.control != b.control:
            reasons.append(f"{a.registry} + {b.registry}: controls differ")
        elif clash := _clash(a.registry, b.registry, methods_dir):
            reasons.append(f"{a.registry} + {b.registry}: both set {clash}")
        else:
            return a, b
        logger.info("%s", reasons[-1])
    names = ", ".join(r.registry for r in hooks) or "none"
    msg = f"no compatible pair among the hook methods ({names})"
    raise NoPairError("; ".join([msg, *reasons]))


def _merged(rules: Sequence[Mapping[str, Any] | None]) -> dict[str, Any]:
    """Abort blocks joined: names once, one limit per metric, one flag per name."""
    out: dict[str, Any] = {}
    for key in ("finite", "permanent"):
        if names := list(
            dict.fromkeys(m for r in rules for m in (r or {}).get(key, []))
        ):
            out[key] = names
    for key, field in (("limits", "metric"), ("flags", "name")):
        items: dict[str, Any] = {}
        for r in rules:
            for item in (r or {}).get(key, []):
                items.setdefault(item[field], item)
        if items:
            out[key] = list(items.values())
    return out


def _budgets(pair: Sequence[str], methods_dir: Path) -> dict[str, Any]:
    """Both methods' ``budgets`` blocks: their config keys and their extra flags."""
    entries = _entries(methods_dir)
    out: dict[str, Any] = {}
    for label in sorted({b for r in pair for b in entries[r].budgets}):
        block: dict[str, Any] = {}
        extra = []
        for registry in pair:
            method = entries[registry]
            own = method.budgets.get(label, {})
            block |= {k: v for k, v in own.items() if k != "abort"}
            flags = (method.abort_rules(registry) or {}).get("flags", [])
            at = (method.abort_rules(registry, label) or {}).get("flags", [])
            extra.append({"flags": [f for f in at if f not in flags]})
        if flags := _merged(extra).get("flags"):
            block["abort"] = {"flags": flags}
        if block:
            out[label] = block
    return out


def _dump(data: Mapping[str, Any]) -> str:
    """Block YAML whose innermost collections are flow style, one item per line."""
    return yaml.safe_dump(
        dict(data), sort_keys=False, default_flow_style=None, width=1000
    )


def _flow(value: Any) -> str:
    """One value in flow style (a scalar as YAML writes it)."""
    return yaml.safe_dump([value], default_flow_style=True, width=1000).strip()[1:-1]


def _base_text(control: str, names: Sequence[str], methods_dir: Path) -> str:
    """The ``base`` block: the control's donor, then each method's knobs, by name."""
    lines = ["base:\n", f"  base_from: {control}.tuned\n"]
    seen: set[str] = set()
    for name in names:
        knobs = {
            k: v for k, v in method_knobs(name, methods_dir).items() if k not in seen
        }
        seen |= knobs.keys()
        lines.append(f"  # {name}\n")
        lines += [f"  {k}: {_flow(v)}\n" for k, v in knobs.items()]
    return "".join(lines)


def combo_text(
    pair: tuple[Ranked, Ranked],
    ranking: Sequence[Ranked],
    *,
    methods_dir: Path = METHODS_DIR,
    phase: str = "final-1m",
) -> str:
    """The text of ``combo.yaml`` for ``pair``, with the ranking as a comment."""
    entries = _entries(methods_dir)
    names = [r.registry for r in pair]
    lines = [HEADER.format(phase=phase)]
    lines += [
        f"#   {i} {r.registry} {r.control} {r.n} {r.iqm:.5f} {r.poi:.3f}\n"
        for i, r in enumerate(ranking, 1)
    ]
    lines.append(_dump({"combine": {"base": pair[0].control, "methods": names}}))
    lines.append("\n" + _base_text(pair[0].control, names, methods_dir))
    if budgets := _budgets(names, methods_dir):
        lines.append("\n" + _dump({"budgets": budgets}))
    lines.append(
        "\n" + _dump({"grid": {"coarse": [{"name": "combined"}], "fine": None}})
    )
    rules = _merged([entries[n].abort_rules(n) for n in names])
    lines.append(f"\n# Both methods' runner rules on {pair[0].control}.\n")
    lines.append(_dump({"abort": rules}))
    lines.append("\n# never tuned: the phases run `combined`\ntuned: null\n")
    return "".join(lines)


def write_combo(
    pair: tuple[Ranked, Ranked],
    ranking: Sequence[Ranked],
    *,
    methods_dir: Path = METHODS_DIR,
    phase: str = "final-1m",
) -> Path:
    """Rewrite ``combo.yaml`` of ``methods_dir`` for ``pair``; return its path.

    The new file must load and resolve its ``combined`` config, or the old one stays.
    """
    path = methods_dir / f"{COMBO}.yaml"
    old = path.read_text(encoding="utf-8") if path.exists() else None
    new = path.with_name(f".{path.name}.new")
    new.write_text(combo_text(pair, ranking, methods_dir=methods_dir, phase=phase))
    new.replace(path)
    try:
        combo = load_method(COMBO, methods_dir)
        resolve_config(
            combo, COMBO, TUNE_BUDGET, ref="combined", methods_dir=methods_dir
        )
    except Exception:
        if old is None:
            path.unlink()
        else:
            path.write_text(old, encoding="utf-8")
        raise
    return path


def _phase_path(phase: str) -> Path:
    """A phase name (``final-1m``) as its file, or a path as given."""
    return Path(phase) if phase.endswith(".yaml") else PHASES_DIR / f"{phase}.yaml"


def main(argv: Sequence[str] | None = None) -> None:
    """Command-line entry point; exits 1 without a pair or with unfinished runs."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--phase", required=True, help="phase name or file")
    parser.add_argument("--tracking-uri", help="store instead of experiment.yaml's")
    parser.add_argument("--methods-dir", type=Path, default=METHODS_DIR)
    parser.add_argument(
        "--allow-partial", action="store_true", help="rank without unfinished runs"
    )
    parser.add_argument("--dry-run", action="store_true", help="leave combo.yaml")
    args = parser.parse_args(argv)
    uri = args.tracking_uri and _sqlite_uri(args.tracking_uri)
    phase = _phase_path(args.phase)
    try:
        scores = final_scores(
            phase,
            tracking_uri=uri,
            methods_dir=args.methods_dir,
            allow_partial=args.allow_partial,
        )
        ranking = rank_methods(scores, methods_dir=args.methods_dir)
        controls = sorted({r.control for r in ranking})
        print("rank method control seeds iqm poi")
        for c in controls:
            print(f"control {c} - {len(scores[c])} {iqm(scores[c]):.5f} -")
        for i, r in enumerate(ranking, 1):
            print(f"{i} {r.registry} {r.control} {r.n} {r.iqm:.5f} {r.poi:.3f}")
        pair = choose_pair(ranking, methods_dir=args.methods_dir)
    except ValueError as err:
        sys.exit(f"select_pair: {err}")
    print(f"pair: {pair[0].registry} + {pair[1].registry} on {pair[0].control}")
    if not args.dry_run:
        path = write_combo(
            pair, ranking, methods_dir=args.methods_dir, phase=phase.stem
        )
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
