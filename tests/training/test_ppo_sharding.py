"""PPO with the env batch sharded over CPU devices: placement and unchanged numbers.

Each run is a subprocess: the host device count is fixed when JAX starts.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

_RUN = """
import sys
import jax
import jax.numpy as jnp
import numpy as np
from wut_murl_craftax.orchestration import runtime
runtime.configure_runtime()
from wut_murl_craftax.training import ppo

cfg = ppo.PPOConfig(env="Craftax-Classic-Symbolic-v1", steps=512, envs=32, num_steps=8,
                    width=16, minibatches=2)
try:
    state = ppo.init_state(cfg, jax.random.key(0))
except ValueError as e:
    print(f"ValueError: {e}")
    raise SystemExit(0)
state, _ = ppo.train_chunk(state, 2)
n = len(jax.devices("cpu"))
env, params = ppo.make_train_env(cfg)
act = jnp.zeros(cfg.envs, jnp.int32)
act = runtime.shard_envs(act, runtime.env_mesh()) if n > 1 else act
*_, info = env.step(jax.random.key(1), state.env_state, act, params)
envs = jax.tree.leaves((state.env_state, state.obs))
ok = all(len(x.sharding.device_set) == n for x in envs + jax.tree.leaves(info))
# split, not copied; events are not checked: Classic's constant floor is replicated
if n > 1:
    ok = ok and all(x.sharding.spec[0] == "envs" for x in envs)
print(f"devices {n} sharded {ok}")
leaves = jax.tree.leaves(state.train_state.params)
np.savez(sys.argv[1], *(np.asarray(x) for x in leaves))
"""


def _start(devices: int, out: Path) -> subprocess.Popen[str]:
    flags = f"--xla_force_host_platform_device_count={devices}" if devices > 1 else ""
    return subprocess.Popen(
        [sys.executable, "-c", _RUN, str(out)],
        env={**os.environ, "JAX_PLATFORMS": "cpu", "XLA_FLAGS": flags},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def test_sharded_envs_give_the_unsharded_params(tmp_path: Path) -> None:
    """4 devices: states and events stay split, params match 1 device to 1e-5; 3
    devices cannot split 32 envs."""
    runs = {n: _start(n, tmp_path / f"params{n}.npz") for n in (4, 1, 3)}
    out = {}
    for n, proc in runs.items():
        stdout, stderr = proc.communicate(timeout=900)
        assert proc.returncode == 0, stderr[-3000:]
        out[n] = stdout.strip().splitlines()[-1]
    assert out == {
        4: "devices 4 sharded True",
        1: "devices 1 sharded True",
        3: "ValueError: 32 envs do not split over 3 CPU devices",
    }
    four, one = (np.load(tmp_path / f"params{n}.npz") for n in (4, 1))
    assert four.files == one.files
    for k in four.files:
        np.testing.assert_allclose(four[k], one[k], atol=1e-5)
