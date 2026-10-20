"""Render rollouts to lossless RGB masters (one step per frame) or final-size windows.

``python -m wut_murl_craftax.visualisation.render RUN.npz ...`` writes 16 px masters;
``--kept review.csv --layout land`` re-renders only the kept windows at final size.
"""

import argparse
import logging
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wut_murl_craftax.visualisation.ffmpeg import FrameWriter
from wut_murl_craftax.visualisation.review import kept, read_rows
from wut_murl_craftax.visualisation.rollout import DEFAULT_RUNS, Rollout, replay

logger = logging.getLogger(__name__)

DEFAULT_MASTERS = Path("outputs/visualisation/masters")
"""Default master folder (git-ignored)."""
FINAL_BPS = {
    ("land", "craftax"): 80,
    ("vert", "craftax"): 96,
    ("land", "classic"): 112,
    ("vert", "classic"): 112,
    ("land4k", "craftax"): 160,
    ("land4k", "classic"): 224,
}
"""Tile size = 16 px x the layout's integer factor, so cuts need no further scaling."""


@dataclass(frozen=True, slots=True)
class RenderStats:
    """What one render produced."""

    frames: int
    width: int
    height: int
    seconds: float
    features_match: bool
    """Replayed features equal the recorded ones (determinism check)."""


def render_master(
    rollout: Rollout,
    out: Path,
    bps: int = 16,
    fps: int = 10,
    start: int = 0,
    end: int | None = None,
    chunk: int = 256,
) -> RenderStats:
    """Replay ``rollout``; write frames ``start..end`` to ``out`` (lossless RGB)."""
    t0 = time.perf_counter()
    chunks = replay(rollout, bps=bps, start=start, end=end, chunk=chunk)
    first = next(chunks)
    if first.frames is None:
        msg = "replay returned no frames; pass bps to render"
        raise ValueError(msg)
    h, w = first.frames.shape[1:3]
    feats = [first.features]
    with FrameWriter(out, w, h, fps) as writer:
        writer.write(first.frames)
        for c in chunks:
            if c.frames is not None:
                writer.write(c.frames)
            feats.append(c.features)
    stop = rollout.steps if end is None else min(end, rollout.steps)
    same = np.array_equal(np.concatenate(feats), rollout.features[start : stop + 1])
    n = stop - start + 1
    stats = RenderStats(n, w, h, time.perf_counter() - t0, same)
    if not same:
        logger.warning("replay of %s diverged from the recorded features", out)
    return stats


def render_kept(
    rows: Iterable[Mapping[str, str]], layout: str, runs_dir: Path, out_dir: Path
) -> list[RenderStats]:
    """Re-render each kept window at its final tile size (``FINAL_BPS``)."""
    loaded: dict[str, Rollout] = {}
    stats = []
    for r in kept(rows):
        run = loaded.setdefault(
            r["rollout"], Rollout.load(runs_dir / f"{r['rollout']}.npz")
        )
        bps = FINAL_BPS[(layout, r["game"])]
        stats.append(
            render_master(
                run,
                out_dir / f"{r['id']}.mkv",
                bps,
                start=int(r["start"]),
                end=int(r["end"]),
                chunk=64,
            )
        )
    return stats


def main(argv: list[str] | None = None) -> None:
    """Render masters of the given rollouts, or the kept windows of a reviewed CSV."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("rollouts", nargs="*", type=Path)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_MASTERS)
    p.add_argument("--bps", type=int, default=16)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument(
        "--kept", type=Path, help="reviewed CSV: render its kept windows instead"
    )
    p.add_argument(
        "--layout", choices=sorted({k[0] for k in FINAL_BPS}), default="land"
    )
    p.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    a = p.parse_args(argv)
    t0 = time.perf_counter()
    if a.kept:
        done = render_kept(read_rows(a.kept), a.layout, a.runs_dir, a.out_dir)
    else:
        done = [
            render_master(Rollout.load(r), a.out_dir / f"{r.stem}.mkv", a.bps, a.fps)
            for r in a.rollouts
        ]
    frames = sum(s.frames for s in done)
    ok = all(s.features_match for s in done)
    print(
        f"{len(done)} videos, {frames} frames in {time.perf_counter() - t0:.1f} s;"
        f" replay identical: {ok}"
    )


if __name__ == "__main__":
    main()
