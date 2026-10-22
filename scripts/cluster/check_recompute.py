"""Measure the PPO-RNN recompute gap on this machine (ctrl-metrics review, Important).

PPO-RNN's permanent guard keeps ``ppo/ratio_first_mb`` in [0.999, 1.001]: the
learner's sequence rerun must give the rollout's log-probs back. It was measured on
CPU and Metal only (PPO-RNN design, "Device verification"); TF32 matmuls on a CUDA GPU
may widen the gap. This runs a few updates of the budget's config on the profile's
devices and prints, per update, the largest log-prob and value gaps over every
minibatch of epoch 1 and the trainer's ``ppo/ratio_first_mb``; it exits 1 when a
log-prob gap or ``|ratio - 1|`` reaches ``--bound``. Run it on a node before final-1b:

    uv run python scripts/cluster/check_recompute.py  # cuda.yaml, 1b config, 3 updates
"""

import argparse
import functools
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

REPO = Path(__file__).resolve().parents[2]
GUARD = 1e-3
"""Half-width of the ``ppo/ratio_first_mb`` band in ``configs/methods/ppo-rnn.yaml``."""

Apply = Callable[[Any, Any, jax.Array, jax.Array], tuple[Any, jax.Array, jax.Array]]
"""``(params, h0, obs, resets) -> (last hidden, logits, values)``."""


def recompute_gaps(
    apply: Apply, params: Any, mbs: Mapping[str, Any]
) -> tuple[float, float, float]:
    """Largest log-prob and value gaps over all minibatches, the first one's mean ratio.

    ``mbs`` holds `ppo_rnn.minibatches` of a rollout batch (leading minibatch axis),
    rerun with the params that collected it, as epoch 1 of the update does.
    """
    logp_gap = v_gap = 0.0
    first_ratio = float("nan")
    for i in range(mbs["act"].shape[0]):
        mb = jax.tree.map(lambda x, i=i: x[i], dict(mbs))
        _, logits, v = apply(params, mb["h0"], mb["obs"], mb["reset"])
        lp = jax.nn.log_softmax(logits)
        lp = jnp.take_along_axis(lp, mb["act"][..., None], -1)[..., 0]
        diff = lp - mb["logp"]
        logp_gap = max(logp_gap, float(jnp.abs(diff).max()))
        v_gap = max(v_gap, float(jnp.abs(v - mb["v"]).max()))
        if i == 0:
            first_ratio = float(jnp.exp(diff).mean())
    return logp_gap, v_gap, first_ratio


def _set_up(profile: Path, precision: str | None) -> None:
    """The runner's setup (`runner._set_up_jax`) before any JAX computation."""
    from wut_murl_craftax.orchestration.profiles import load_profile
    from wut_murl_craftax.orchestration.runtime import (
        check_backend,
        configure_runtime,
        import_craftax,
        select_platforms,
    )

    machine = load_profile(profile)
    select_platforms(machine.device_roles)
    configure_runtime(host_devices=machine.host_devices)
    check_backend(machine.device_roles)
    import_craftax()
    if precision:
        jax.config.update("jax_default_matmul_precision", precision)


def measure(budget: str, updates: int, seed: int, shape: dict[str, int]) -> float:
    """Print the gaps of ``updates`` updates; return the worst log-prob or ratio gap."""
    from wut_murl_craftax.orchestration import device_roles
    from wut_murl_craftax.orchestration.queue import (
        budget_steps,
        load_method,
        resolve_config,
    )
    from wut_murl_craftax.training.interface import METHODS
    from wut_murl_craftax.training.ppo_rnn import (
        apply_network,
        collect,
        minibatches,
        train_chunk,
    )

    trainer = METHODS["ppo-rnn"]
    config = resolve_config(load_method("ppo-rnn"), "ppo-rnn", budget) | shape
    state = trainer.init(config | {"steps": budget_steps(budget)}, jax.random.key(seed))
    roles = device_roles()
    apply = jax.jit(functools.partial(apply_network, trainer.policy(state).net))
    print(f"actor {roles.actor}, learner {roles.learner}, config {config}")
    worst = 0.0
    for u in range(1, updates + 1):
        _, batch, _, _ = collect(state)
        batch = jax.device_put(batch, roles.learner)
        _, k = jax.random.split(state.key)  # epoch 1's shuffle key, as `learn`
        mbs = minibatches(batch, k, state.cfg)
        logp, v, ratio = recompute_gaps(apply, state.train_state.params, mbs)
        state, metrics = train_chunk(state, 1)
        guard = float(metrics["ppo/ratio_first_mb"][0])
        print(
            f"update {u}: log-prob gap {logp:.2e}, value gap {v:.2e}, "
            f"ratio_first_mb {guard:.7f} (rerun here {ratio:.7f})"
        )
        worst = max(worst, logp, abs(guard - 1), abs(ratio - 1))
    return worst


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point; the exit code is 1 when the gap reaches the bound."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--profile", type=Path, default=REPO / "configs/profiles/cuda.yaml"
    )
    parser.add_argument("--budget", default="1b", help="the config's budget block")
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--envs", type=int, help="override the budget's envs")
    parser.add_argument("--num-steps", type=int, help="override its num_steps")
    parser.add_argument(
        "--bound",
        type=float,
        default=GUARD / 10,
        help="largest acceptable gap (default: a tenth of the guard band)",
    )
    parser.add_argument(
        "--precision",
        choices=["default", "high", "highest"],
        help="jax_default_matmul_precision to try (default: JAX's, as the runner)",
    )
    args = parser.parse_args(argv)
    _set_up(args.profile, args.precision)
    shape = {"envs": args.envs, "num_steps": args.num_steps}
    worst = measure(
        args.budget,
        args.updates,
        args.seed,
        {k: v for k, v in shape.items() if v is not None},
    )
    verdict = "OK" if worst < args.bound else "OVER"
    print(
        f"worst gap {worst:.2e}, bound {args.bound:.0e}, guard {GUARD:.0e}: {verdict}"
    )
    return 0 if worst < args.bound else 1


if __name__ == "__main__":
    raise SystemExit(main())
