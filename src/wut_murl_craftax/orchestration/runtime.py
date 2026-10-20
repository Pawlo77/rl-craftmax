"""Fast JAX runtime for Craftax: compile cache, device roles, CPU shards, textures.

Measured effect of each setting: ``knowledge/craftax/Craftax runtime setup.md``.
"""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import jax
from craftax.craftax import constants as craftax_constants
from craftax.craftax_classic import constants as classic_constants
from jax.sharding import Mesh, NamedSharding, PartitionSpec

DEFAULT_CACHE_DIR = Path(".cache/jax")
"""Persistent XLA compilation cache, relative to the working directory."""

_HOST_DEVICES_FLAG = "--xla_force_host_platform_device_count"


@dataclass(frozen=True, slots=True)
class DeviceRoles:
    """Where environment rollouts and network updates run."""

    actor: jax.Device
    """Device that steps the environments."""
    learner: jax.Device
    """Device that runs the network updates."""


def configure_runtime(
    cache_dir: Path | str = DEFAULT_CACHE_DIR, host_devices: int | None = None
) -> None:
    """Enable the persistent compilation cache and optionally split the CPU.

    Call before the first JAX computation: XLA reads ``XLA_FLAGS`` when its CPU
    backend starts, so ``host_devices`` has no effect afterwards.
    """
    if host_devices and _HOST_DEVICES_FLAG not in os.environ.get("XLA_FLAGS", ""):
        flags = f"{os.environ.get('XLA_FLAGS', '')} {_HOST_DEVICES_FLAG}={host_devices}"
        os.environ["XLA_FLAGS"] = flags.strip()
    cache = str(Path(cache_dir).resolve())
    jax.config.update("jax_compilation_cache_dir", cache)  # type: ignore[no-untyped-call]
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)  # type: ignore[no-untyped-call]


def device_roles() -> DeviceRoles:
    """Pick the actor and learner devices for the backends JAX was started with.

    Apple GPU (``metal``) steps Craftax slower than the CPU but updates networks
    faster, so rollouts stay on the CPU there; a CUDA GPU does both.
    """
    cpu = jax.devices("cpu")[0]
    accelerators = [device for device in jax.devices() if device.platform != "cpu"]
    if not accelerators:
        return DeviceRoles(actor=cpu, learner=cpu)
    learner = accelerators[0]
    actor = cpu if learner.platform == "metal" else learner
    return DeviceRoles(actor=actor, learner=learner)


def env_mesh(devices: list[jax.Device] | None = None) -> Mesh:
    """Return a 1-D ``envs`` mesh over the CPU devices for sharded rollouts.

    The axis type is ``Auto``: Craftax fails with ``ShardingTypeError`` on an
    ``Explicit`` axis inside ``dynamic_update_slice``.
    """
    devices = devices or jax.devices("cpu")
    return jax.make_mesh(
        (len(devices),),
        ("envs",),
        axis_types=(jax.sharding.AxisType.Auto,),
        devices=devices,
    )


def shard_envs[T](tree: T, mesh: Mesh) -> T:
    """Place a batch of environment states with its leading axis split over ``mesh``.

    The batch size must be divisible by the number of devices in ``mesh``.
    """
    spec = PartitionSpec("envs")  # type: ignore[no-untyped-call]
    return cast(T, jax.device_put(tree, NamedSharding(mesh, spec)))


def warm_texture_cache() -> list[Path]:
    """Build the Craftax and Craftax-Classic texture caches; return their paths.

    Craftax builds them on the first pixel render and stores them inside the
    installed package, so a fresh virtual environment starts without them.
    """
    for module in (craftax_constants, classic_constants):
        module.load_all_textures()
    return [
        Path(craftax_constants.TEXTURE_CACHE_FILE),
        Path(classic_constants.TEXTURE_CACHE_FILE),
    ]


def main(argv: list[str] | None = None) -> None:
    """Warm the texture caches and print the device roles (``make warm``)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    configure_runtime()
    for path in warm_texture_cache():
        print(f"texture cache: {path} ({path.stat().st_size / 1e6:.1f} MB)")
    roles = device_roles()
    print(f"actor: {roles.actor}, learner: {roles.learner}")


if __name__ == "__main__":
    main()
