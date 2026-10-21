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


def _write_atomic(path: Path, data: bytes) -> None:
    """Write ``data`` to a temporary file, then rename it to ``path``.

    A kill at any point leaves the old file or the new one, never half of it.
    """
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def load_index(run_dir: Path) -> list[CheckpointEntry]:
    """Checkpoints of a run in the order they were written; empty if none."""
    path = Path(run_dir) / "ckpt" / "index.json"
    if not path.exists():
        return []
    entries: list[CheckpointEntry] = json.loads(path.read_text())
    return entries


def missing_checkpoints(run_dir: Path) -> list[int]:
    """Updates whose ``ckpt/`` file is gone although ``index.json`` lists it."""
    return [
        e["update"]
        for e in load_index(run_dir)
        if not (Path(run_dir) / "ckpt" / e["file"]).is_file()
    ]


def drop_missing_checkpoints(run_dir: Path) -> list[int]:
    """Drop the `missing_checkpoints` from the index; return their updates."""
    lost = missing_checkpoints(run_dir)
    if lost:
        kept = [e for e in load_index(run_dir) if e["update"] not in lost]
        index = Path(run_dir) / "ckpt" / "index.json"
        _write_atomic(index, json.dumps(kept, indent=1).encode())
    return lost


def save_checkpoint[T](
    run_dir: Path, update: int, env_steps: int, wall_s: float, params: T
) -> Path:
    """Write ``ckpt/<update>.msgpack`` (params only) and append it to the index.

    Both files are replaced atomically (`_write_atomic`), params first.
    """
    ckpt = Path(run_dir) / "ckpt"
    ckpt.mkdir(parents=True, exist_ok=True)
    path = ckpt / f"{update:05d}.msgpack"
    _write_atomic(path, serialization.to_bytes(params))
    entry = CheckpointEntry(
        update=update, env_steps=env_steps, wall_s=round(wall_s, 1), file=path.name
    )
    index = sorted([*load_index(run_dir), entry], key=lambda e: e["update"])
    _write_atomic(ckpt / "index.json", json.dumps(index, indent=1).encode())
    return path


def load_params[T](run_dir: Path, update: int, template: T) -> T:
    """Restore the params saved at ``update`` into the structure of ``template``."""
    data = (Path(run_dir) / "ckpt" / f"{update:05d}.msgpack").read_bytes()
    return cast(T, serialization.from_bytes(template, data))


def _is_key(leaf: object) -> bool:
    return isinstance(leaf, jax.Array) and jnp.issubdtype(
        leaf.dtype, jax.dtypes.prng_key
    )


def _paths(tree: Any) -> list[str]:
    """The key path of every leaf, in `jax.tree.leaves` order."""
    return [
        jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_leaves_with_path(tree)
    ]


def save_state(run_dir: Path, state: Any) -> Path:
    """Write every leaf of ``state`` to ``state/state.npz``, replacing it atomically.

    Typed PRNG keys are stored as their key data, the leaves' tree paths as ``paths``
    (`load_state` checks them). The file is written to a temporary name, synced and
    renamed, so a save killed half-way leaves the previous state.
    """
    arrays: dict[str, Any] = {
        f"k{i}" if _is_key(leaf) else f"a{i}": np.asarray(
            jax.random.key_data(leaf) if _is_key(leaf) else leaf
        )
        for i, leaf in enumerate(jax.tree.leaves(state))
    }
    arrays["paths"] = np.array(_paths(state))
    path = Path(run_dir) / STATE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        np.savez(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    return path


def _misfit(name: str, arr: np.ndarray, dtype: Any, shape: tuple[int, ...]) -> str:
    return (
        f"{name}: saved {arr.dtype} {arr.shape}, the template has {dtype} {shape}; "
        "the state comes from other code (see the run's code/* tags)"
    )


def _restore(z: Any, i: int, like: Any, name: str) -> Any:
    """Leaf ``i`` of the saved state, with the type, dtype and placement of ``like``.

    Raises:
        ValueError: If its shape or dtype differs; a numpy leaf (a growing
            accumulator) may differ in its first dimension only.
    """
    if _is_key(like):
        data, want = z[f"k{i}"], jax.random.key_data(like)
        if data.shape != want.shape:
            raise ValueError(_misfit(name, data, want.dtype, want.shape))
        key = jax.random.wrap_key_data(data, impl=jax.random.key_impl(like))
        return jax.device_put(key, like.sharding)
    arr, ref = z[f"a{i}"], like if isinstance(like, jax.Array) else np.asarray(like)
    grows = isinstance(like, np.ndarray) and like.ndim > 0
    fits = arr.shape[1:] == ref.shape[1:] and arr.ndim == ref.ndim
    void = arr.dtype.kind == "V" and arr.dtype.itemsize == ref.dtype.itemsize
    if not (fits if grows else arr.shape == ref.shape) or not (
        arr.dtype == ref.dtype or void
    ):
        raise ValueError(_misfit(name, arr, ref.dtype, ref.shape))
    if arr.dtype != ref.dtype:  # npz keeps bfloat16 as raw 2-byte void
        arr = arr.view(ref.dtype)
    if isinstance(like, jax.Array):
        return jax.device_put(arr, like.sharding)
    if isinstance(like, np.ndarray):
        return arr
    return arr[()] if isinstance(like, np.generic) else type(like)(arr.item())


def load_state[T](run_dir: Path, like: T) -> T:
    """Restore the state saved by `save_state` into the structure of ``like``.

    ``like`` gives the tree, the leaf types, the key implementations and the device
    placement; numpy leaves take the saved shape, so growing accumulators restore.
    A state saved before the tree paths were stored skips the path check.

    Raises:
        ValueError: If the saved state has another number of leaves, other tree
            paths (fields renamed or reordered), or a leaf of another shape or dtype.
    """
    leaves, treedef = jax.tree.flatten(like)
    names = _paths(like)
    with np.load(Path(run_dir) / STATE_FILE) as z:
        n = len([f for f in z.files if f != "paths"])
        if n != len(leaves):
            msg = f"saved state has {n} leaves, the template has {len(leaves)}"
            raise ValueError(msg)
        if "paths" in z.files and (saved := z["paths"].tolist()) != names:
            got, want = next(
                (a, b) for a, b in zip(saved, names, strict=True) if a != b
            )
            msg = f"saved state has another tree: {got} where the template has {want}"
            raise ValueError(msg)
        restored = [_restore(z, i, leaf, names[i]) for i, leaf in enumerate(leaves)]
    return cast(T, jax.tree.unflatten(treedef, restored))
