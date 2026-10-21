"""Golden numbers of the three controls: a refactor that moves them fails here.

Each control trains 2 updates on Craftax-Classic at tiny sizes on one CPU device (a
fresh interpreter, so the test process's backends do not matter). The SHA-256 of its
params and losses is pinned on the recording machine (`GOLDEN`); a few losses and the
params' norm are pinned to a tolerance everywhere (`VALUES`). A deliberate numeric
change updates both.
"""

import json
import os
import platform
import subprocess
import sys

import jax
import pytest

RECORDED_ON = ("Darwin", "arm64", "0.11.2")
"""OS, machine and JAX version the hashes were recorded with; XLA's CPU kernels may
round differently elsewhere."""
GOLDEN = {
    "ppo": "e140c632f539482d2eb594cd66956855dcc1a136759de78b07bd15893a7d65cd",
    "ppo-rnn": "d762a28585967ce7756be646d8d8c12a32a09928dd0f6214223b35b48fa0471b",
    "pqn-rnn": "8bf2ef28f730fc057a9992c0849ffca78b5ba8a1f26e999d1b3d290c0721dd51",
}
"""Registry name -> SHA-256 of params and losses after 2 updates (from a968039)."""
VALUES = {
    "ppo": [
        -0.0025813993997871876,
        0.026626795530319214,
        2.833191156387329,
        1.6937607142608613e-05,
        -0.4402965307235718,
        0.0,
        0.568393349647522,
        -0.0009738719090819359,
        0.07597976922988892,
        2.8331620693206787,
        2.4588514406787e-06,
        0.19728237390518188,
        0.0,
        0.8495850563049316,
        19.623379297166906,
    ],
    "ppo-rnn": [
        -0.0010810387320816517,
        0.01780412718653679,
        2.833204746246338,
        8.218263246817514e-06,
        0.08304870128631592,
        0.0,
        0.4543517827987671,
        -0.0006069457158446312,
        0.07610353082418442,
        2.8331737518310547,
        1.057992449204903e-06,
        -0.1433274745941162,
        0.0,
        0.9062970876693726,
        27.759996577212686,
    ],
    "pqn-rnn": [
        0.10180967301130295,
        0.7134021520614624,
        0.0019197575747966766,
        0.15152806043624878,
        57.22927456192564,
    ],
}
"""Registry name -> the finite losses of both updates, then the params' L2 norm: the
portable check, at a relative tolerance that XLA's rounding on other CPUs stays in."""
RTOL, ATOL = 1e-4, 1e-6
"""Tolerance of `VALUES` (absolute for the near-zero KL and clip terms)."""
_SETUP = (platform.system(), platform.machine())

_TRAIN = """
import hashlib, json
import jax, numpy as np
from wut_murl_craftax.orchestration.runtime import configure_runtime
configure_runtime()
from wut_murl_craftax.training.interface import METHODS

SMALL = {"env": "Craftax-Classic-Symbolic-v1", "envs": 8, "num_steps": 8,
         "minibatches": 2, "steps": 2 * 64}
CONFIGS = {
    "ppo": SMALL | {"width": 32},
    "ppo-rnn": SMALL | {"width": 32, "rnn_width": 32},
    "pqn-rnn": SMALL | {"hidden": 32},
}
out = {}
for name, cfg in CONFIGS.items():
    trainer = METHODS[name]
    state = trainer.init({"method": name, **cfg}, jax.random.key(0))
    state, metrics = trainer.train_chunk(state, 2)
    h = hashlib.sha256()
    losses = np.asarray(metrics["losses"])
    for x in [*jax.tree.leaves(trainer.params(state)), losses]:
        x = np.asarray(x)
        h.update(f"{x.dtype}{x.shape}".encode())
        h.update(x.tobytes())
    leaves = [np.asarray(x, np.float64) for x in jax.tree.leaves(trainer.params(state))]
    norm = float(np.sqrt(sum((x**2).sum() for x in leaves)))
    values = [v for v in losses.ravel().tolist() if np.isfinite(v)] + [norm]
    out[name] = {"sha256": h.hexdigest(), "values": values}
print(json.dumps(out))
"""


@pytest.fixture(scope="module")
def trained() -> dict[str, dict[str, object]]:
    """Hash and values of each control, from one CPU-only interpreter."""
    result = subprocess.run(
        [sys.executable, "-c", _TRAIN],
        env={**os.environ, "JAX_PLATFORMS": "cpu", "XLA_FLAGS": ""},
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    return dict(json.loads(result.stdout.strip().splitlines()[-1]))


@pytest.mark.skipif(
    RECORDED_ON[:2] != _SETUP, reason=f"hashes recorded on {RECORDED_ON[:2]}"
)
@pytest.mark.parametrize("name", list(GOLDEN))
def test_two_classic_updates_reproduce_the_golden_hash(
    trained: dict[str, dict[str, object]], name: str
) -> None:
    assert jax.__version__ == RECORDED_ON[2], "JAX changed: re-record deliberately"
    got = trained[name]
    assert got["sha256"] == GOLDEN[name], f"values {got['values']}"


@pytest.mark.parametrize("name", list(VALUES))
def test_two_classic_updates_reproduce_the_golden_values(
    trained: dict[str, dict[str, object]], name: str
) -> None:
    """The portable guard: Linux and CUDA nodes run it on their CPU too."""
    assert trained[name]["values"] == pytest.approx(VALUES[name], rel=RTOL, abs=ATOL)
