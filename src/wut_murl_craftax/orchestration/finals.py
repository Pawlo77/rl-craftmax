"""The dyna-q kappa ablation of final-1m, set once dyna-q is marked tuned.

final-1m runs dyna-q's tuned config and ``{ref: tuned, set: {plan_kappa: 0.0}}``. A
tuned kappa of 0 makes that item a copy of ``tuned`` and the queue refuses the phase;
the method plan then ablates to the best nonzero stage-1 kappa: the kappa of the
tune-coarse config with the highest mean final score among those with a kappa above 0
(`kappa_ablation`). Usage: ``python -m wut_murl_craftax.orchestration.finals [--phase
P] [--dry-run]``; it prints the new kappa, or that the ablation stays.
"""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import yaml

from .queue import (
    COARSE_PHASE,
    METHODS_DIR,
    PHASES_DIR,
    TUNE_BUDGET,
    QueueConfigError,
    _sqlite_uri,
    coarse_results,
    expand_phase,
    load_method,
    resolve_config,
)

logger = logging.getLogger(__name__)

DYNA = "dyna-q"
"""Slug and registry name of the method with the kappa ablation."""
KAPPA = "plan_kappa"
"""The Dyna-Q+ bonus knob."""
ZERO_ITEM = "{ref: tuned, set: {plan_kappa: 0.0}}"
"""The ablation item as final-1m.yaml spells it."""


def _item_kappa(phase: Path) -> float | None:
    """The kappa the phase's dyna-q ablation item sets (None without one)."""
    raw = yaml.safe_load(phase.read_text(encoding="utf-8"))
    for item in raw["runs"].get(DYNA, {}).get("configs", []):
        if isinstance(item, dict) and KAPPA in item.get("set", {}):
            return float(item["set"][KAPPA])
    return None


def kappa_ablation(
    phase: Path = PHASES_DIR / "final-1m.yaml",
    *,
    tracking_uri: str | None = None,
    methods_dir: Path = METHODS_DIR,
    phases_dir: Path | None = None,
    dry_run: bool = False,
    min_complete: float | None = None,
) -> float | None:
    """Rewrite the kappa-0 ablation of ``phase`` when dyna-q's tuned kappa is 0.

    Returns the new kappa, or None when the tuned kappa is above 0 or the item is
    already rewritten (nothing changes: a rerun is a no-op).
    The rewritten file must expand without a duplicate run, or it is left as it was.
    ``min_complete``: the coarse configs aborted for good are left out of the choice
    (`queue.coarse_results`).

    Raises:
        QueueConfigError: If the coarse stage is incomplete, no stage-1 kappa is
            above 0, or the phase file does not spell the item as `ZERO_ITEM`.
    """
    method = load_method(DYNA, methods_dir)
    tuned = resolve_config(
        method, DYNA, TUNE_BUDGET, ref="tuned", methods_dir=methods_dir
    )
    if tuned[KAPPA] != 0:
        return None
    if (item := _item_kappa(phase)) != 0:
        logger.info("the %s ablation already runs %s %s", DYNA, KAPPA, item)
        return None
    phases = phases_dir or PHASES_DIR
    means = coarse_results(
        DYNA,
        tracking_uri=tracking_uri,
        methods_dir=methods_dir,
        phases_dir=phases,
        min_complete=min_complete,
    )
    kappas = {
        s.config_hash: float(s.config[KAPPA])
        for s in expand_phase(
            phases / f"{COARSE_PHASE}.yaml", method=DYNA, methods_dir=methods_dir
        )
    }
    nonzero = [h for h in means if kappas[h] > 0]
    if not nonzero:
        msg = f"{DYNA}: no stage-1 config with {KAPPA} above 0"
        raise QueueConfigError(msg)
    best = kappas[max(nonzero, key=means.__getitem__)]
    old = phase.read_text(encoding="utf-8")
    if old.count(ZERO_ITEM) != 1:
        msg = f"{phase.name}: expected the {DYNA} item {ZERO_ITEM} once"
        raise QueueConfigError(msg)
    logger.info("tuned %s is 0: the ablation runs %s %s", KAPPA, KAPPA, best)
    if dry_run:
        return best
    new = old.replace(ZERO_ITEM, ZERO_ITEM.replace("0.0", str(best)))
    phase.write_text(new, encoding="utf-8")
    try:
        expand_phase(phase, method=DYNA, methods_dir=methods_dir)
    except Exception:
        phase.write_text(old, encoding="utf-8")
        raise
    return best


def main(argv: Sequence[str] | None = None) -> None:
    """Command-line entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--phase", type=Path, default=PHASES_DIR / "final-1m.yaml")
    parser.add_argument("--tracking-uri", help="store instead of experiment.yaml's")
    parser.add_argument("--methods-dir", type=Path, default=METHODS_DIR)
    parser.add_argument("--phases-dir", type=Path, help="phases of the tune-coarse")
    parser.add_argument("--dry-run", action="store_true", help="leave the phase file")
    parser.add_argument(
        "--min-complete", type=float, metavar="SHARE", help="as the queue's flag"
    )
    args = parser.parse_args(argv)
    kappa = kappa_ablation(
        args.phase,
        tracking_uri=args.tracking_uri and _sqlite_uri(args.tracking_uri),
        methods_dir=args.methods_dir,
        phases_dir=args.phases_dir,
        dry_run=args.dry_run,
        min_complete=args.min_complete,
    )
    if kappa is None:
        print(f"{DYNA} tuned {KAPPA} is above 0: the {KAPPA} 0 ablation stays")
    else:
        print(f"{DYNA} ablation: {KAPPA} {kappa}")


if __name__ == "__main__":
    main()
