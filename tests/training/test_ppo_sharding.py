"""PPO with the env batch sharded over CPU devices: placement and unchanged numbers.

Each run is a subprocess: the host device count is fixed when JAX starts.
"""

import subprocess
from collections.abc import Callable
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
n = len(jax.devices("cpu"))
s = ppo._setup(cfg)
params = jax.device_put(state.train_state.params, s.shared_at)
out = s.rollout(params, state.env_state, state.obs, state.episode, state.key_act)
# a per-step leaf held whole on every device multiplies the rollout's memory by n
paths = jax.tree_util.tree_leaves_with_path(out)
copied = [jax.tree_util.keystr(p) for p, x in paths
          if n > 1 and x.size >= cfg.batch and x.sharding.is_fully_replicated]
state, _ = ppo.train_chunk(state, 2)
env, params = ppo.make_train_env(cfg)
act = jnp.zeros(cfg.envs, jnp.int32)
act = runtime.shard_envs(act, runtime.env_mesh()) if n > 1 else act
*_, info = env.step(jax.random.key(1), state.env_state, act, params)
envs = jax.tree.leaves((state.env_state, state.obs))
ok = all(len(x.sharding.device_set) == n for x in envs + jax.tree.leaves(info))
# split, not copied; events are not checked: Classic's constant floor is replicated
if n > 1:
    ok = ok and all(x.sharding.spec[0] == "envs" for x in envs)
print(f"devices {n} sharded {ok} copied {copied}")
leaves = jax.tree.leaves(state.train_state.params)
np.savez(sys.argv[1], *(np.asarray(x) for x in leaves))
"""


def test_sharded_envs_give_the_unsharded_params(
    tmp_path: Path, start_on_cpus: Callable[[str, int, Path], subprocess.Popen[str]]
) -> None:
    """4 devices: states, events and the per-step rollout batch stay split (a copy per
    device took 36 GB at 1024 x 64 Symbolic), params match 1 device to 1e-5; 3
    devices cannot split 32 envs."""
    runs = {n: start_on_cpus(_RUN, n, tmp_path / f"params{n}.npz") for n in (4, 1, 3)}
    out = {}
    for n, proc in runs.items():
        stdout, stderr = proc.communicate(timeout=900)
        assert proc.returncode == 0, stderr[-3000:]
        out[n] = stdout.strip().splitlines()[-1]
    assert out == {
        4: "devices 4 sharded True copied []",
        1: "devices 1 sharded True copied []",
        3: "ValueError: 32 envs do not split over 3 CPU devices",
    }
    four, one = (np.load(tmp_path / f"params{n}.npz") for n in (4, 1))
    assert four.files == one.files
    for k in four.files:
        np.testing.assert_allclose(four[k], one[k], atol=1e-5)
