"""Tiny 2-update Classic runs, one method per network family, evaluated once.

`method_run` trains a registry entry for 2 updates (checkpoints 0 and 2, as the runner
saves them) and writes both eval modes on `EVAL_SEEDS`; ``ppo`` reuses `tiny_run`.
"""

import atexit
import functools
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import jax
from _learning_run import EVAL_SEEDS, tiny_run

from wut_murl_craftax.training.checkpoints import save_checkpoint
from wut_murl_craftax.training.interface import METHODS
from wut_murl_craftax.visualisation.eval_rollouts import MODES, EvalRun, evaluate

CLASSIC = "Craftax-Classic-Symbolic-v1"
_TINY = {"env": CLASSIC, "envs": 16, "num_steps": 8, "minibatches": 2}
_Q = {"hidden": 16, "epochs": 1, "eps_decay": 2.0}
"""Epsilon falls over 4 updates, so checkpoint 2 acts at about 0.5, not 0.005."""
CONFIGS: dict[str, dict[str, Any]] = {
    "ppo-rnn": _TINY | {"width": 32, "rnn_width": 32},
    "pqn-rnn": _TINY | _Q,
    "smdp-skills": _TINY
    | _Q
    | {"log2_rows": 12, "extra_obs_dim": 26, "timeout_mult": 0.5},
}
"""Run configs as the runner writes them (no ``steps``); the skill network's step
limits are halved, so a wrong ``timeout_mult`` changes its input."""
UPDATES = {"ppo": 4, "ppo-rnn": 2, "pqn-rnn": 2, "smdp-skills": 2}
"""The last checkpoint of each run."""
FAMILY_RUNS = tuple(UPDATES)
"""One registry entry per network family."""


@functools.cache
def method_run(name: str) -> Path:
    """The run of registry entry ``name`` with both eval modes of every checkpoint."""
    if name == "ppo":
        return tiny_run()
    root = Path(tempfile.mkdtemp(prefix=f"{name}_run_"))
    atexit.register(shutil.rmtree, root, ignore_errors=True)
    cfg = {"method": name} | CONFIGS[name]
    (root / "config.json").write_text(json.dumps(cfg))
    trainer = METHODS[name]
    spu = trainer.steps_per_update(cfg)
    state = trainer.init(cfg | {"steps": 2 * spu}, jax.random.key(0))
    save_checkpoint(root, 0, 0, 0.0, trainer.params(state))
    state, _ = trainer.train_chunk(state, 2)
    save_checkpoint(root, 2, 2 * spu, 1.0, trainer.params(state))
    run = EvalRun.load(root)
    for mode in MODES:
        evaluate(run, run.updates, mode, seeds=EVAL_SEEDS)
    return root
