"""Distillation datasets: named features and network labels at stochastic-policy states.

`collect` plays the stochastic policy of one checkpoint on fixed seeds (or reads the
stored eval file when it holds them all), replays each episode, re-runs the network
over the whole episode (`network_outputs`, recurrent carry threaded) and keeps at
most ``max_states_per_episode`` states sampled uniformly. Rows hold `FEATURES` plus
`MEMORY_FEATURES` (from the rollout) and, for two-level policies, `SKILL_FEATURES`
(from the eval extras); the label is the greedy primitive action. Seed splits: plan
[[2026-10-20-craftax-interpretation-and-ruleset]]; files: `dataset_path`.
"""

import os
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import NamedTuple, Self, cast

import numpy as np

from ..training.skills import NUM_SKILLS, SKILLS
from ..visualisation.eval_rollouts import (
    EVAL_SEEDS,
    EvalRun,
    Family,
    load_eval,
    network_family,
    network_outputs,
    replay_episode,
    rollout_seeds,
)
from .features import FEATURES, featurize

type Outputs = Callable[..., tuple[np.ndarray, np.ndarray]]

TRAIN_SEEDS: tuple[int, ...] = tuple(range(30000, 30256))
"""256 training episodes."""
VAL_SEEDS: tuple[int, ...] = EVAL_SEEDS
"""The 64 eval seeds: tree-size choice; their stochastic eval files are reused."""
TEST_SEEDS: tuple[int, ...] = tuple(range(20000, 20064))
"""64 held-out episodes: fidelity and closed-loop play."""
SPLITS: dict[str, tuple[int, ...]] = {
    "train": TRAIN_SEEDS,
    "val": VAL_SEEDS,
    "test": TEST_SEEDS,
}
"""Split name -> seeds; disjoint, so no state of a held-out episode is fitted."""
MAX_STATES = 2000
"""Training states kept per episode."""
MEMORY_FEATURES: tuple[str, ...] = ("prev_action", "steps_since_unlock")
"""Memory proxies of recurrent policies: the previous action (-1 at the start) and
steps since the last unlock (from the episode start before the first)."""
SKILL_FEATURES: tuple[str, ...] = (*(f"skill_{s.name}" for s in SKILLS), "skill_steps")
"""Two-level policies: the running skill one-hot and ``extras/steps_in_skill``."""
RECURRENT: frozenset[Family] = frozenset({"ppo-rnn", "pqn-rnn", "skills"})
"""Families with a carried hidden state (the skill network is an LSTM): the ones
whose trees may use `MEMORY_FEATURES`."""
ROOT = Path("outputs/interpretation")
_BATCH = len(EVAL_SEEDS)
"""Seeds per fresh rollout; a seed plays the same in any batch."""
_FIELDS = ("features", "label", "probs", "episode", "step", "floor")


class Dataset(NamedTuple):
    """States of one policy and split; rows follow the seeds, then the step."""

    features: np.ndarray
    """float32 ``(N, len(columns))``."""
    columns: tuple[str, ...]
    label: np.ndarray
    """int8: the network's greedy primitive action."""
    probs: np.ndarray
    """float32 ``(N, n_actions)``: the stochastic policy's action distribution."""
    episode: np.ndarray
    """int32: the episode's seed."""
    step: np.ndarray
    """int32: step in the episode."""
    floor: np.ndarray
    """int8: the ``floor`` feature (0 on Classic)."""
    family: str
    """`network_family` of the policy."""
    recurrent: bool
    """Whether ``family`` is in `RECURRENT`."""

    def save(self, path: Path) -> None:
        """Write a compressed ``.npz``; a file that exists is complete."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")  # one per process
        with tmp.open("wb") as f:
            arrays = {k: getattr(self, k) for k in _FIELDS}
            arrays |= {
                "columns": np.array(self.columns),
                "family": np.array(self.family),
                "recurrent": np.array(self.recurrent),
            }
            np.savez_compressed(f, **arrays)
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> Self:
        """Read a file written by `save`."""
        with np.load(path) as z:
            arrays = {k: z[k] for k in _FIELDS}
            return cls(
                columns=tuple(z["columns"].tolist()),
                family=str(z["family"]),
                recurrent=bool(z["recurrent"]),
                **arrays,
            )


def dataset_path(
    budget: str, method: str, run: str, split: str, root: Path = ROOT
) -> Path:
    """``<root>/<budget>/<method>/<run>/<split>.npz``."""
    return Path(root) / budget / method / run / f"{split}.npz"


def _batches(
    run: EvalRun, update: int, seeds: Sequence[int]
) -> Iterator[tuple[dict[str, np.ndarray], list[int]]]:
    """Eval arrays and the rows of ``seeds`` in them, in ``seeds`` order.

    The stored stochastic eval file of ``update`` serves when it holds every seed;
    otherwise the seeds roll out afresh in batches of `_BATCH`.
    """
    try:
        ev = load_eval(run.dir, update)
    except FileNotFoundError:
        ev = {"seeds": np.zeros(0, np.int64)}
    stored = {int(s): i for i, s in enumerate(ev["seeds"])}
    if all(s in stored for s in seeds):
        yield ev, [stored[s] for s in seeds]
        return
    params = run.params(update)
    for i in range(0, len(seeds), _BATCH):
        batch = seeds[i : i + _BATCH]
        arrays = rollout_seeds(run, params, batch, greedy=False)
        # without should_stop a rollout always finishes: never None
        yield cast(dict[str, np.ndarray], arrays), list(range(len(batch)))


type Played = tuple[int, np.ndarray, np.ndarray | None, np.ndarray | None]
"""One episode: ``(seed, actions, skill, steps_in_skill)``."""


def _played(arrays: dict[str, np.ndarray], rows: Sequence[int]) -> Iterator[Played]:
    """The episodes of eval-file ``arrays`` at ``rows``."""
    for i in rows:
        n = int(arrays["length"][i])
        skill, steps = (
            arrays[k][i, :n] if k in arrays else None
            for k in ("extras/skill", "extras/steps_in_skill")
        )
        yield int(arrays["seeds"][i]), arrays["actions"][i, :n], skill, steps


def _episode(
    run: EvalRun,
    update: int,
    played: Played,
    cap: int | None,
    outputs: Outputs,
) -> tuple[np.ndarray, ...]:
    """One episode's rows in `_FIELDS` order."""
    seed, actions, skill, steps = played
    ep = replay_episode(run, seed, actions)
    n = len(actions)
    # whole episode first: a recurrent network needs the carry of every earlier step
    args = (ep.obs, actions, ep.done)
    probs = outputs(run, update, *args, False, skill, steps)[0]
    greedy = outputs(run, update, *args, True, skill, steps)[0]
    rng = np.random.default_rng(seed)  # per seed: the same rows in any seed list
    keep = np.arange(n)
    if cap is not None and n > cap:
        keep = np.sort(rng.choice(n, cap, replace=False))
    unlocked = np.flatnonzero(ep.events.unlocked.any(1))
    last = np.r_[-1, unlocked][np.searchsorted(unlocked, keep)]  # before each step
    since = keep - last - 1
    cols = [featurize(ep.obs[keep], run.game), np.r_[-1, actions[:-1]][keep], since]
    if skill is not None and steps is not None:
        cols += [np.eye(NUM_SKILLS)[skill[keep]], steps[keep]]
    feats = np.column_stack(cols).astype(np.float32)
    names = FEATURES[run.game]
    floor = feats[:, names.index("floor")] if "floor" in names else np.zeros(len(keep))
    return (
        feats,
        greedy[keep].argmax(1).astype(np.int8),
        probs[keep],
        np.full(len(keep), seed, np.int32),
        keep.astype(np.int32),
        floor.astype(np.int8),
    )


def collect(
    run: EvalRun,
    update: int,
    seeds: Sequence[int],
    max_states_per_episode: int | None = MAX_STATES,
    outputs: Outputs = network_outputs,
    family: Family | None = None,
) -> Dataset:
    """States the stochastic policy of ``update`` visits on ``seeds``, labelled.

    ``max_states_per_episode`` states are sampled uniformly per episode, seeded by
    its seed (None keeps every state); ``outputs`` is `network_outputs` or a stand-in
    of its signature and ``family`` the toy's stand-in for `network_family`.

    Raises:
        ValueError: If ``seeds`` is empty.
    """
    if not seeds:
        msg = "collect needs at least one seed"
        raise ValueError(msg)
    played = (p for a, rows in _batches(run, update, seeds) for p in _played(a, rows))
    return _dataset(run, update, played, max_states_per_episode, outputs, family)


def relabel(
    run: EvalRun,
    update: int,
    arrays: dict[str, np.ndarray],
    max_states_per_episode: int | None = MAX_STATES,
    outputs: Outputs = network_outputs,
    family: Family | None = None,
) -> Dataset:
    """The states of eval-file ``arrays`` of any policy, labelled by ``update``.

    ``arrays`` come from `rollout_seeds` (a tree's, for a DAgger round); the rest is
    as in `collect`, so the stochastic arrays of ``update`` give `collect`'s rows.
    """
    played = _played(arrays, range(len(arrays["seeds"])))
    return _dataset(run, update, played, max_states_per_episode, outputs, family)


def _dataset(
    run: EvalRun,
    update: int,
    played: Iterable[Played],
    cap: int | None,
    outputs: Outputs,
    family: Family | None,
) -> Dataset:
    """Rows of every ``played`` episode (`_episode`), as one dataset."""
    family = network_family(run.policy) if family is None else family
    rows = [_episode(run, update, p, cap, outputs) for p in played]
    parts = dict(zip(_FIELDS, zip(*rows, strict=True), strict=True))
    arrays = {k: np.concatenate(v) for k, v in parts.items()}
    columns = (*FEATURES[run.game], *MEMORY_FEATURES)
    if arrays["features"].shape[1] > len(columns):  # the eval extras held the skill
        columns += SKILL_FEATURES
    recurrent = family in RECURRENT
    return Dataset(columns=columns, family=family, recurrent=recurrent, **arrays)
