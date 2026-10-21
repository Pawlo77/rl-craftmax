"""Log-spaced params-only checkpoints with an index, and the full resume state."""

import json
import os
from pathlib import Path
from typing import Any, TypedDict, cast

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

STATE_FILE = Path("state") / "state.npz"
"""Full training state of a run, relative to the run directory."""


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


def _is_key(leaf: object) -> bool:
    return isinstance(leaf, jax.Array) and jnp.issubdtype(
        leaf.dtype, jax.dtypes.prng_key
    )


def save_state(run_dir: Path, state: Any) -> Path:
    """Write every leaf of ``state`` to ``state/state.npz``, replacing it atomically.

    Typed PRNG keys are stored as their key data. The file is written to a temporary
    name, synced and renamed, so a save killed half-way leaves the previous state.
    """
    arrays: dict[str, Any] = {
        f"k{i}" if _is_key(leaf) else f"a{i}": np.asarray(
            jax.random.key_data(leaf) if _is_key(leaf) else leaf
        )
        for i, leaf in enumerate(jax.tree.leaves(state))
    }
    path = Path(run_dir) / STATE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        np.savez(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    return path


def _restore(z: Any, i: int, like: Any) -> Any:
    """Leaf ``i`` of the saved state, with the type, dtype and placement of ``like``."""
    if _is_key(like):
        key = jax.random.wrap_key_data(z[f"k{i}"], impl=jax.random.key_impl(like))
        return jax.device_put(key, like.sharding)
    arr = z[f"a{i}"]
    dtype = np.asarray(like).dtype if not isinstance(like, jax.Array) else like.dtype
    if arr.dtype != dtype:  # npz keeps bfloat16 as raw 2-byte void
        arr = arr.view(dtype)
    if isinstance(like, jax.Array):
        return jax.device_put(arr, like.sharding)
    if isinstance(like, np.ndarray):
        return arr
    return arr[()] if isinstance(like, np.generic) else type(like)(arr.item())


def load_state[T](run_dir: Path, like: T) -> T:
    """Restore the state saved by `save_state` into the structure of ``like``.

    ``like`` gives the tree, the leaf types, the key implementations and the device
    placement; numpy leaves take the saved shape, so growing accumulators restore.

    Raises:
        ValueError: If the saved state has another number of leaves.
    """
    leaves, treedef = jax.tree.flatten(like)
    with np.load(Path(run_dir) / STATE_FILE) as z:
        if len(z.files) != len(leaves):
            msg = (
                f"saved state has {len(z.files)} leaves, the template has {len(leaves)}"
            )
            raise ValueError(msg)
        restored = [_restore(z, i, leaf) for i, leaf in enumerate(leaves)]
    return cast(T, jax.tree.unflatten(treedef, restored))
