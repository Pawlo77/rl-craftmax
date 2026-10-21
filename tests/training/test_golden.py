"""Golden numbers of the three controls: a refactor that moves them fails here.

Each control trains 2 updates on Craftax-Classic at tiny sizes on one CPU device (a
fresh interpreter, so the test process's backends do not matter); the SHA-256 of its
params and per-update losses is pinned. A deliberate numeric change updates `GOLDEN`.
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
    out[name] = {"sha256": h.hexdigest(), "losses": losses[0].tolist()}
print(json.dumps(out))
"""


@pytest.fixture(scope="module")
def trained() -> dict[str, dict[str, object]]:
    """Hash and first-update losses of each control, from one CPU-only interpreter."""
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


@pytest.mark.parametrize("name", list(GOLDEN))
def test_two_classic_updates_reproduce_the_golden_numbers(
    trained: dict[str, dict[str, object]], name: str
) -> None:
    setup = (platform.system(), platform.machine(), jax.__version__)
    if setup[:2] != RECORDED_ON[:2]:
        pytest.skip(f"golden numbers recorded on {RECORDED_ON[:2]}, not {setup[:2]}")
    assert setup[2] == RECORDED_ON[2], "JAX changed: re-record GOLDEN deliberately"
    got = trained[name]
    assert got["sha256"] == GOLDEN[name], f"first-update losses {got['losses']}"
