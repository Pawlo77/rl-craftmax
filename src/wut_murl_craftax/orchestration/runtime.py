"""Fast JAX runtime for Craftax: compile cache, device roles, CPU shards, textures.

Measured effect of each setting: ``knowledge/craftax/Craftax runtime setup.md``.
"""

import argparse
import importlib
import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from .profiles import RoleMode

DEFAULT_CACHE_DIR = Path(".cache/jax")
"""Persistent XLA compilation cache, relative to the working directory."""

_HOST_DEVICES_FLAG = "--xla_force_host_platform_device_count"
CRAFTAX_MODULES: tuple[str, ...] = (
    "craftax.craftax.envs.craftax_symbolic_env",
    "craftax.craftax.envs.craftax_pixels_env",
    "craftax.craftax_classic.envs.craftax_symbolic_env",
    "craftax.craftax_classic.envs.craftax_pixels_env",
)
"""The env modules ``make_craftax_env_from_name`` imports on its first call."""


@dataclass(frozen=True, slots=True)
class DeviceRoles:
    """Where environment rollouts and network updates run."""

    actor: jax.Device
    """Device that steps the environments."""
    learner: jax.Device
    """Device that runs the network updates."""


def metal_installed() -> bool:
    """Whether the Apple GPU plugin (``metaljax``, group ``metal``) is installed."""
    try:
        return importlib.util.find_spec("jax_plugins.metal") is not None
    except ModuleNotFoundError:  # no jax_plugins namespace at all
        return False


def select_platforms(mode: RoleMode) -> None:
    """Make the Apple GPU JAX's default backend for the ``cpu-actor`` mode.

    ``device_roles`` only finds the GPU as learner when ``metal`` is the default
    backend, which needs ``JAX_PLATFORMS=metal,cpu`` before the backends start. Other
    machines (Linux CPU, the ``gpu`` mode) and an explicit ``JAX_PLATFORMS`` stay as
    they are. Call before the first JAX computation, like ``configure_runtime``.
    """
    if mode != "cpu-actor" or os.environ.get("JAX_PLATFORMS") or not metal_installed():
        return
    os.environ["JAX_PLATFORMS"] = "metal,cpu"
    # jax read the variable when it was imported, which is earlier than this call
    jax.config.update("jax_platforms", "metal,cpu")  # type: ignore[no-untyped-call]


class BackendError(RuntimeError):
    """The profile's device mode needs an accelerator that JAX did not start."""


def check_backend(mode: RoleMode) -> None:
    """Fail in the ``gpu`` mode when JAX has no GPU (CUDA) device (review M11).

    Without it, a node whose CUDA plugin or driver failed would train on one CPU
    device, silently and far slower. Call after ``configure_runtime``.

    Raises:
        BackendError: ``mode`` is ``gpu`` and no device has the platform ``gpu``.
    """
    if mode != "gpu":
        return
    platforms = sorted({d.platform for d in jax.devices()})
    if "gpu" not in platforms:
        msg = (
            f"profile mode gpu, but JAX has no GPU backend (platforms {platforms}): "
            "check `uv sync --group cuda`, the NVIDIA driver and JAX_PLATFORMS"
        )
        raise BackendError(msg)


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


def import_craftax() -> None:
    """Import every Craftax env module, with its constants and world generation.

    Craftax builds ``jnp`` constants at import and ``make_craftax_env_from_name``
    imports its env lazily: a first import inside a trace (``jax.eval_shape`` or a
    jitted function) stores that trace's tracers as module globals, and the next use
    outside it raises ``UnexpectedTracerError``. Call after ``configure_runtime``: the
    imports start the JAX backends. Pixel envs and renderers also cache their textures
    on the first build (``load_all_textures``, ~0.8 GB, not loaded here): build those
    outside traces too.
    """
    for module in CRAFTAX_MODULES:
        importlib.import_module(module)


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
    paths = []
    for game in ("craftax", "craftax_classic"):
        # imported here: the constants build JAX arrays on import, which starts the
        # backends and would freeze XLA_FLAGS before configure_runtime splits the CPU
        module = importlib.import_module(f"craftax.{game}.constants")
        module.load_all_textures()
        paths.append(Path(module.TEXTURE_CACHE_FILE))
    return paths


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
