"""Reproduction check: does a control's final score match the published one?

``band`` is the acceptance band of the PPO plan (Task 5): half-width
h = 0.05 + 2 sqrt(SE_ours^2 + (s_pub / sqrt(10))^2), pass when |mean - published| <= h.
The CLI reads the final scores of the method's runs of the published config in a phase
(the ``published-<budget>`` grid entry, else the method's base at that budget, matched
by config hash) from MLflow, prints mean, SE, h and pass/fail, and tags each of those
runs ``repro/*`` (MLflow has no phase-level object). Exit status 1 on a fail.

Usage: ``python -m wut_murl_craftax.evaluation.reproduction --method ppo --budget 1m
--phase tune-coarse``
"""

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..orchestration.queue import (
    METHODS_DIR,
    REPO_ROOT,
    SCORE_METRIC,
    TAG_BUDGET,
    TAG_CONFIG_HASH,
    TAG_METHOD,
    TAG_PHASES,
    _client,
    _runs,
    config_hash,
    load_method,
    resolve_config,
)

PUBLISHED_SCORES = REPO_ROOT / "configs" / "reference" / "published_scores.yaml"
ROUNDING = 0.05
"""Half of the published score's one-decimal rounding step, added to every band."""
N_PUBLISHED = 10
"""Seeds behind the published mean (the README scoreboard)."""


def standard_error(scores: Sequence[float]) -> float:
    """Sample standard deviation over sqrt(n).

    Raises:
        ValueError: With fewer than 2 scores.
    """
    if len(scores) < 2:
        msg = f"the standard error needs at least 2 seeds, got {len(scores)}"
        raise ValueError(msg)
    return float(np.std(scores, ddof=1)) / math.sqrt(len(scores))


def band(ours: Sequence[float], published: float, s_pub: float) -> tuple[float, bool]:
    """Half-width h of the acceptance band and whether our mean is inside it.

    ``ours`` are our final scores, one per seed (SE_ours uses n = ``len(ours)``);
    ``s_pub`` is the published per-seed standard deviation (a proxy, see
    ``configs/reference/published_scores.yaml``).
    """
    se = standard_error(ours)
    h = ROUNDING + 2 * math.sqrt(se**2 + (s_pub / math.sqrt(N_PUBLISHED)) ** 2)
    return h, abs(float(np.mean(ours)) - published) <= h


@dataclass(frozen=True, slots=True)
class Reproduction:
    """Outcome of one check."""

    n: int
    mean: float
    se: float
    half_width: float
    published: float
    s_pub: float
    passed: bool


def published_config_hashes(
    method_slug: str, budget: str, methods_dir: Path
) -> dict[str, str]:
    """Registry name -> hash of the published config at ``budget``."""
    method = load_method(method_slug, methods_dir)
    ref = f"published-{budget}"
    if ref not in {e["name"] for e in method.coarse}:
        ref = "base"
    return {
        r: config_hash(
            resolve_config(method, r, budget, ref=ref, methods_dir=methods_dir)
        )
        for r in method.registry_names
    }


def check(
    method: str,
    budget: str,
    phase: str,
    *,
    tracking_uri: str | None = None,
    methods_dir: Path = METHODS_DIR,
    reference: Path = PUBLISHED_SCORES,
) -> Reproduction:
    """Compare the finished runs of the published config in ``phase`` with the paper.

    Tags every one of those runs ``repro/{n, mean, se, half_width, published, s_pub,
    passed}``.

    Raises:
        ValueError: Without a published score for ``method`` at ``budget``, or with
            fewer than 2 finished runs of the published config in ``phase``.
    """
    ref: dict[str, Any] = yaml.safe_load(reference.read_text(encoding="utf-8"))
    entry = (ref.get(method) or {}).get(budget)
    if entry is None or entry.get("s_pub") is None:
        msg = f"{reference.name} has no score and s_pub for {method} at {budget}"
        raise ValueError(msg)
    hashes = published_config_hashes(method, budget, methods_dir)
    client = _client(tracking_uri)
    runs = [
        r
        for r in _runs(client, "attributes.status = 'FINISHED'")
        if hashes.get(r.data.tags.get(TAG_METHOD)) == r.data.tags.get(TAG_CONFIG_HASH)
        and r.data.tags.get(TAG_BUDGET) == budget
        and phase in r.data.tags.get(TAG_PHASES, "").split(",")
        and SCORE_METRIC in r.data.metrics
    ]
    scores = [r.data.metrics[SCORE_METRIC] for r in runs]
    if len(scores) < 2:
        msg = (
            f"need at least 2 finished {method} runs of the published config at "
            f"{budget} in {phase}, found {len(scores)}"
        )
        raise ValueError(msg)
    half_width, passed = band(scores, entry["score"], entry["s_pub"])
    result = Reproduction(
        len(scores),
        float(np.mean(scores)),
        standard_error(scores),
        half_width,
        float(entry["score"]),
        float(entry["s_pub"]),
        passed,
    )
    tags = {
        "n": str(result.n),
        "mean": str(result.mean),
        "se": str(result.se),
        "half_width": str(result.half_width),
        "published": str(result.published),
        "s_pub": str(result.s_pub),
        "passed": str(passed).lower(),
    }
    for run in runs:
        for key, value in tags.items():
            client.set_tag(run.info.run_id, f"repro/{key}", value)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    """Command-line entry point; exits 1 when the check fails."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--method", required=True, help="slug in published_scores.yaml")
    parser.add_argument("--budget", required=True, choices=("1m", "1b"))
    parser.add_argument("--phase", required=True, help="phase whose runs are read")
    parser.add_argument(
        "--tracking-uri", help="store instead of configs/experiment.yaml's"
    )
    parser.add_argument("--methods-dir", type=Path, default=METHODS_DIR)
    parser.add_argument("--reference", type=Path, default=PUBLISHED_SCORES)
    args = parser.parse_args(argv)
    r = check(
        args.method,
        args.budget,
        args.phase,
        tracking_uri=args.tracking_uri,
        methods_dir=args.methods_dir,
        reference=args.reference,
    )
    verdict = "PASS" if r.passed else "FAIL"
    print(f"{args.method} {args.budget} in {args.phase}: published {r.published}")
    print(f"n={r.n} mean={r.mean:.4f} SE={r.se:.4f} h={r.half_width:.4f} {verdict}")
    if not r.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
