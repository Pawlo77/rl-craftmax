"""PPO-RNN with the env batch sharded over CPU devices: placement and unchanged numbers.

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
from wut_murl_craftax.training import ppo_rnn

def count_init(key, spec):
    return {"calls": jnp.zeros(spec.n_envs, jnp.int32), "total": jnp.int32(0)}

def count(s, e, e2, o, d, t):
    s = {"calls": s["calls"] + 1, "total": s["total"] + d.size}
    return jnp.zeros(d.shape, jnp.float32), s

cfg = ppo_rnn.PPORNNConfig(
    env="Craftax-Classic-Symbolic-v1", steps=512, envs=32, num_steps=8, width=64,
    rnn_width=64, minibatches=2, init_state_fn=count_init, intrinsic_reward_fn=count,
)
try:
    state = ppo_rnn.init_state(cfg, jax.random.key(0))
except ValueError as e:
    print(f"ValueError: {e}")
    raise SystemExit(0)
n = len(jax.devices("cpu"))

def split(s):
    per_env = jax.tree.leaves((s.env_state, s.obs, s.hidden, s.method_state["calls"]))
    ok = all(len(x.sharding.device_set) == n for x in per_env)
    return ok and (n == 1 or all(x.sharding.spec[0] == "envs" for x in per_env))

ok = split(state)
state, _ = ppo_rnn.train_chunk(state, 2)
ok = ok and split(state) and int(state.method_state["total"]) == 2 * 8 * 32
print(f"devices {n} sharded {ok}")
leaves = jax.tree.leaves(state.train_state.params)
np.savez(sys.argv[1], *(np.asarray(x) for x in leaves))
"""


def test_sharded_envs_give_the_unsharded_params(
    tmp_path: Path, start_on_cpus: Callable[[str, int, Path], subprocess.Popen[str]]
) -> None:
    """4 devices: env states, GRU carry and per-env method state stay split, params
    match 1 device to 1e-5; 3 devices cannot split 32 envs."""
    runs = {n: start_on_cpus(_RUN, n, tmp_path / f"params{n}.npz") for n in (4, 1, 3)}
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
