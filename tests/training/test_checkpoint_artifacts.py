"""Wiring of `checkpoint_artifacts` in the method trainers; the writers have tests."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from wut_murl_craftax.training import count_bonus, dyna_q, smdp
from wut_murl_craftax.training.hooks import HookMethod
from wut_murl_craftax.training.interface import METHODS
from wut_murl_craftax.training.ppo_rnn import PPORNNConfig, PPORNNTrainer
from wut_murl_craftax.training.pqn_rnn import PQNRNNConfig, PQNRNNTrainer

Calls = list[tuple[Any, ...]]


def test_count_bonus_writes_its_top_keys_at_each_checkpoint() -> None:
    assert count_bonus.METHOD.artifacts is count_bonus.write_top_keys
    for name in ("count-bonus-ppo-rnn", "count-bonus-pqn-rnn"):
        assert callable(getattr(METHODS[name], "checkpoint_artifacts", None))


@pytest.mark.parametrize(
    ("trainer_type", "cfg"),
    [(PPORNNTrainer, PPORNNConfig), (PQNRNNTrainer, PQNRNNConfig)],
)
def test_a_hook_trainer_hands_the_method_state_to_the_method_writer(
    trainer_type: Any, cfg: Any, tmp_path: Path
) -> None:
    calls: Calls = []
    method = HookMethod(
        count_bonus.CountBonusConfig,
        count_bonus.hooks,
        lambda run_dir, update, state: calls.append((run_dir, update, state)),
    )
    state = SimpleNamespace(method_state="the method state")
    trainer_type(cfg(), method).checkpoint_artifacts(state, tmp_path, 7)
    assert calls == [(tmp_path, 7, "the method state")]
    # no method, or a method without a writer: nothing happens
    trainer_type(cfg()).checkpoint_artifacts(state, tmp_path, 7)
    plain = HookMethod(count_bonus.CountBonusConfig, count_bonus.hooks)
    trainer_type(cfg(), plain).checkpoint_artifacts(state, tmp_path, 7)
    assert len(calls) == 1


def test_smdp_and_dyna_write_the_greedy_table_with_the_q_init(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tables: Calls = []
    graphs: Calls = []
    monkeypatch.setattr(smdp.SMDPTrainer, "params", lambda self, state: "params")
    monkeypatch.setattr(smdp, "write_greedy_table", lambda *a: tables.append(a))
    monkeypatch.setattr(dyna_q, "write_model_graph", lambda *a: graphs.append(a))
    inner: Any = SimpleNamespace(cfg=SimpleNamespace(top=SimpleNamespace(q_init=7.0)))
    smdp.SMDPTrainer().checkpoint_artifacts(inner, tmp_path, 3)
    assert tables == [(tmp_path, 3, "params", 7.0)]
    outer: Any = SimpleNamespace(smdp=inner)
    dyna_q.DynaTrainer().checkpoint_artifacts(outer, tmp_path, 5)
    assert graphs == [(tmp_path, 5, outer)]
    assert tables[1:] == [(tmp_path, 5, "params", 7.0)]
