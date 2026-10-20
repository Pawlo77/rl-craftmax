"""Log-spaced checkpoint schedule and params-only checkpoints with an index file."""

import json
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
from flax import serialization


class CheckpointEntry(TypedDict):
    """One row of ``ckpt/index.json``."""

    update: int
    env_steps: int
    wall_s: float
    file: str


def checkpoint_schedule(n_updates: int, n_ckpt: int) -> list[int]:
    """Update 0, the last update and ``n_ckpt`` log-spaced updates in between.

    Learning is fastest early, so linear spacing would spend most checkpoints on
    the plateau.
    """
    spaced = np.unique(np.round(np.geomspace(1, n_updates, n_ckpt)).astype(int))
    return sorted({0, n_updates, *spaced.tolist()})


def load_index(run_dir: Path) -> list[CheckpointEntry]:
    """Checkpoints of a run in the order they were written; empty if none."""
    path = Path(run_dir) / "ckpt" / "index.json"
    if not path.exists():
        return []
    entries: list[CheckpointEntry] = json.loads(path.read_text())
    return entries


def save_checkpoint[T](
    run_dir: Path, update: int, env_steps: int, wall_s: float, params: T
) -> Path:
    """Write ``ckpt/<update>.msgpack`` (params only) and append it to the index."""
    ckpt = Path(run_dir) / "ckpt"
    ckpt.mkdir(parents=True, exist_ok=True)
    path = ckpt / f"{update:05d}.msgpack"
    path.write_bytes(serialization.to_bytes(params))
    entry = CheckpointEntry(
        update=update, env_steps=env_steps, wall_s=round(wall_s, 1), file=path.name
    )
    index = [*load_index(run_dir), entry]
    (ckpt / "index.json").write_text(json.dumps(index, indent=1))
    return path


def load_params[T](run_dir: Path, update: int, template: T) -> T:
    """Restore the params saved at ``update`` into the structure of ``template``."""
    data = (Path(run_dir) / "ckpt" / f"{update:05d}.msgpack").read_bytes()
    return cast(T, serialization.from_bytes(template, data))
