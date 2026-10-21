"""Tests for the fast Craftax runtime setup."""

import importlib.util
import inspect
import os
import platform
import subprocess
import sys
from pathlib import Path

import jax
import pytest
from craftax import craftax_env

from wut_murl_craftax.orchestration.runtime import (
    CRAFTAX_MODULES,
    configure_runtime,
    device_roles,
    warm_texture_cache,
)

_SHARDED_ROLLOUT = """
import jax, jax.numpy as jnp
from craftax.craftax_env import make_craftax_env_from_name
from wut_murl_craftax.orchestration.runtime import env_mesh, shard_envs

env = make_craftax_env_from_name("Craftax-Classic-Symbolic-v1", auto_reset=True)
params = env.default_params
step = jax.vmap(env.step, (0, 0, 0, None))

def rollout(state, key):
    def body(state, k):
        keys = jax.random.split(k, 8)
        actions = jax.random.randint(k, (8,), 0, env.action_space(params).n)
        _, state, reward, _, _ = step(keys, state, actions, params)
        return state, reward
    return jax.lax.scan(body, state, jax.random.split(key, 5))[1]

reset_keys = jax.random.split(jax.random.key(0), 8)
_, state = jax.vmap(env.reset, (0, None))(reset_keys, params)
plain = jax.jit(rollout)(state, jax.random.key(1))
sharded = jax.jit(rollout)(shard_envs(state, env_mesh()), jax.random.key(1))
assert len(jax.devices("cpu")) == 4
assert jnp.array_equal(plain, sharded)
print("ok")
"""


def _run(code: str, **env: str) -> str:
    """Run ``code`` in a fresh interpreter so XLA and backend settings apply."""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        check=True,
        timeout=600,
    )
    return result.stdout


def test_configure_runtime_sets_compile_cache(tmp_path: Path) -> None:
    """The persistent compilation cache points at the requested directory."""
    previous = jax.config.jax_compilation_cache_dir
    try:
        configure_runtime(tmp_path / "jax")
        assert jax.config.jax_compilation_cache_dir == str((tmp_path / "jax").resolve())
    finally:
        jax.config.update("jax_compilation_cache_dir", previous)  # type: ignore[no-untyped-call]


def test_device_roles_on_cpu() -> None:
    """Without an accelerator both roles use the CPU."""
    if any(device.platform != "cpu" for device in jax.devices()):
        pytest.skip("an accelerator backend is active")
    roles = device_roles()
    assert roles.actor.platform == roles.learner.platform == "cpu"


@pytest.mark.skipif(
    platform.system() != "Darwin"
    or platform.machine() != "arm64"
    or importlib.util.find_spec("metaljax") is None,
    reason="needs Apple Silicon with metaljax",
)
def test_device_roles_on_apple_gpu() -> None:
    """With metal, rollouts stay on the CPU and updates go to the GPU."""
    code = (
        "from wut_murl_craftax.orchestration.runtime import device_roles\n"
        "r = device_roles(); print(r.actor.platform, r.learner.platform)"
    )
    assert _run(code, JAX_PLATFORMS="metal,cpu").split()[-2:] == ["cpu", "metal"]


def test_sharded_rollout_matches_unsharded() -> None:
    """Splitting the env batch over host devices does not change the rollout."""
    flags = "--xla_force_host_platform_device_count=4"
    assert _run(_SHARDED_ROLLOUT, JAX_PLATFORMS="cpu", XLA_FLAGS=flags).strip() == "ok"


def test_warm_texture_cache_writes_both_caches() -> None:
    """Both games' texture caches exist after warming."""
    paths = warm_texture_cache()
    assert len(paths) == 2
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)


def test_host_devices_split_the_cpu_after_the_package_imports() -> None:
    """Importing the runtime and the training records leaves XLA_FLAGS open."""
    code = (
        "import jax\n"
        "from wut_murl_craftax.orchestration.runtime import configure_runtime\n"
        "from wut_murl_craftax.training import records\n"
        "configure_runtime(host_devices=3)\n"
        "print(len(jax.devices('cpu')), len(records.achievement_names('Craftax-v1')))"
    )
    assert _run(code, XLA_FLAGS="").split()[-2:] == ["3", "67"]


_TRACE_FIRST = """
import sys
import jax
from wut_murl_craftax.orchestration.runtime import configure_runtime, import_craftax
assert not any(m.startswith("craftax") for m in sys.modules)
configure_runtime(host_devices=3)
import_craftax()
from craftax.craftax_env import make_craftax_env_from_name

def reset(name):
    env = make_craftax_env_from_name(name, auto_reset=True)
    return env.reset(jax.random.key(0), env.default_params)

for name in ("Craftax-Symbolic-v1", "Craftax-Classic-Symbolic-v1"):
    jax.eval_shape(lambda: reset(name))  # the env's first use inside a trace
    reset(name)  # eager: no module constant may be a tracer of that trace
print(len(jax.devices("cpu")))
"""


def test_import_craftax_keeps_module_constants_out_of_traces() -> None:
    """After `import_craftax`, building a symbolic env first inside a trace is safe."""
    assert _run(_TRACE_FIRST, XLA_FLAGS="", JAX_PLATFORMS="cpu").split()[-1] == "3"


def test_craftax_modules_cover_every_lazy_env_import() -> None:
    """`CRAFTAX_MODULES` lists each module ``make_craftax_env_from_name`` imports."""
    lines = inspect.getsource(craftax_env).splitlines()
    lazy = {ln.split()[1] for ln in lines if ln.strip().startswith("from craftax.")}
    assert lazy == set(CRAFTAX_MODULES)
