"""Tests for the combination pair from final-1m (ruling R71) and its statistics."""

import logging
from pathlib import Path
from typing import Any

import mlflow
import pytest
import yaml

from wut_murl_craftax.evaluation import select_pair
from wut_murl_craftax.evaluation.select_pair import (
    NoPairError,
    choose_pair,
    final_scores,
    rank_methods,
    write_combo,
)
from wut_murl_craftax.evaluation.stats import iqm, probability_of_improvement
from wut_murl_craftax.orchestration import queue
from wut_murl_craftax.orchestration.queue import (
    expand_phase,
    load_method,
    resolve_config,
)

REPO = Path(__file__).resolve().parents[2]
FINAL_1M = REPO / "configs" / "phases" / "final-1m.yaml"
TUNED = {
    "count-bonus": {"ppo-rnn": {"beta": 0.2}, "pqn-rnn": {"beta": 0.015}},
    "sil": {"lambda_sil": 0.3},
    "smdp-skills": {"alpha": 0.05},
    "dyna-q": {"plan_n": 20, "plan_kappa": 0.003},
}
"""Tuned blocks of the fixture methods (the shipped ones may still be null)."""
MEANS = {
    "ppo": 2.1,
    "ppo-rnn": 2.2,
    "pqn-rnn": 1.3,
    "count-bonus-ppo-rnn": 2.4,
    "count-bonus-pqn-rnn": 1.8,
    "sil": 2.25,
    "smdp-skills": 3.0,
    "dyna-q": 2.9,
}
"""Final-score level of each entry's tuned runs; the skill methods rank first."""
OFFSETS = [-0.4, -0.1, -0.05, -0.02, 0.0, 0.0, 0.02, 0.05, 0.1, 0.4]
"""Seed s scores ``mean + OFFSETS[s]``: the IQM is the mean."""


# --- statistics -----------------------------------------------------------------


def test_iqm_is_the_mean_of_the_middle_half() -> None:
    assert iqm([1, 2, 3, 4, 5, 6, 7, 8]) == 4.5
    assert iqm([100, 1, 2, 3, 4, 5, 6, 7, 8, -100]) == pytest.approx(4.5)


def test_iqm_needs_a_score() -> None:
    with pytest.raises(ValueError, match="at least one"):
        iqm([])


def test_probability_of_improvement_counts_wins_and_half_ties() -> None:
    assert probability_of_improvement([1, 2, 3], [1, 2, 3]) == 0.5
    assert probability_of_improvement([4, 5], [1, 2, 3]) == 1.0
    assert probability_of_improvement([1, 2], [4, 5]) == 0.0
    assert probability_of_improvement([2], [1, 2, 3]) == 0.5


# --- the final-1m store ---------------------------------------------------------


def _methods(out: Path, tuned: dict[str, Any] = TUNED) -> Path:
    out.mkdir(exist_ok=True)
    for path in (REPO / "configs" / "methods").glob("*.yaml"):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        raw["tuned"] = tuned.get(path.stem, raw["tuned"])
        (out / path.name).write_text(yaml.safe_dump(raw, sort_keys=False))
    return out


class Store:
    """A throwaway store holding the final-1m runs as the runner leaves them."""

    def __init__(self, tmp_path: Path) -> None:
        self.tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
        self.client = mlflow.MlflowClient(tracking_uri=self.tracking_uri)
        self.exp = self.client.create_experiment(
            "final-1m", artifact_location=(tmp_path / "art").as_uri()
        )

    def add(
        self,
        spec: queue.RunSpec,
        score: float,
        status: str = "FINISHED",
        tags: dict[str, str] | None = None,
    ) -> None:
        base = {
            queue.TAG_METHOD: spec.registry_name,
            queue.TAG_CONFIG_HASH: spec.config_hash,
            queue.TAG_PHASES: spec.phase,
            queue.TAG_BUDGET: spec.budget,
        }
        run = self.client.create_run(
            self.exp, run_name=spec.name, tags=base | (tags or {})
        )
        self.client.log_metric(run.info.run_id, queue.SCORE_METRIC, score)
        self.client.set_terminated(run.info.run_id, status)


def _tuned_hash(methods: Path, spec: queue.RunSpec) -> str:
    method = load_method(spec.config_slug, methods)
    cfg = resolve_config(
        method,
        spec.registry_name,
        spec.budget,
        ref="tuned",
        overrides={"env": spec.env},
        methods_dir=methods,
    )
    return queue.config_hash(cfg)


def _fill(
    store: Store,
    methods: Path,
    means: dict[str, float] = MEANS,
    phase: Path = FINAL_1M,
    skip: tuple[str, ...] = (),
) -> None:
    """Every run of ``phase``: tuned ones at ``means``, ablations far above them."""
    for spec in expand_phase(phase, methods_dir=methods):
        if spec.name in skip:
            continue
        tuned = spec.config_hash == _tuned_hash(methods, spec)
        level = means[spec.registry_name] if tuned else 9.0
        store.add(spec, level + OFFSETS[spec.seed])


@pytest.fixture
def methods(tmp_path: Path) -> Path:
    return _methods(tmp_path / "methods")


@pytest.fixture
def store(tmp_path: Path, methods: Path) -> Store:
    store = Store(tmp_path)
    _fill(store, methods)
    return store


def _scores(store: Store, methods: Path, **kw: Any) -> dict[str, list[float]]:
    return final_scores(
        FINAL_1M, tracking_uri=store.tracking_uri, methods_dir=methods, **kw
    )


def test_final_scores_read_the_tuned_config_runs_only(
    store: Store, methods: Path
) -> None:
    scores = _scores(store, methods)
    assert scores.keys() == MEANS.keys()
    assert all(len(s) == 10 for s in scores.values())
    assert iqm(scores["sil"]) == pytest.approx(MEANS["sil"])  # ablation runs at 9


@pytest.mark.parametrize(
    ("status", "tags"),
    [
        ("FAILED", {}),
        ("KILLED", {}),
        ("FINISHED", {queue.TAG_EVAL_PENDING: "true"}),
        ("FINISHED", {queue.TAG_ABORT_BUG: "true"}),
    ],
)
def test_an_unfinished_tuned_run_is_refused_unless_partial(
    tmp_path: Path, methods: Path, status: str, tags: dict[str, str]
) -> None:
    store = Store(tmp_path)
    sil = [
        s
        for s in expand_phase(FINAL_1M, method="sil", methods_dir=methods)
        if s.config_hash == _tuned_hash(methods, s)
    ]
    _fill(store, methods, skip=(sil[3].name,))
    store.add(sil[3], 2.0, status, tags)
    with pytest.raises(ValueError, match=sil[3].name):
        _scores(store, methods)
    assert len(_scores(store, methods, allow_partial=True)["sil"]) == 9


def test_a_missing_tuned_run_is_refused(tmp_path: Path, methods: Path) -> None:
    store = Store(tmp_path)
    spec = expand_phase(FINAL_1M, method="ppo", methods_dir=methods)[0]
    _fill(store, methods, skip=(spec.name,))
    with pytest.raises(ValueError, match=f"ppo: 9 of 10.*{spec.name}"):
        _scores(store, methods)


# --- ranking and the pair -------------------------------------------------------


def test_methods_rank_by_iqm_against_their_base_control(
    store: Store, methods: Path
) -> None:
    ranking = rank_methods(_scores(store, methods), methods_dir=methods)
    assert [(r.registry, r.control) for r in ranking] == [
        ("smdp-skills", "pqn-rnn"),
        ("dyna-q", "pqn-rnn"),  # through smdp-skills.tuned
        ("count-bonus-ppo-rnn", "ppo-rnn"),
        ("sil", "ppo-rnn"),
        ("count-bonus-pqn-rnn", "pqn-rnn"),
    ]
    assert ranking[2].iqm == pytest.approx(2.4)
    assert ranking[2].poi == probability_of_improvement(
        [2.4 + o for o in OFFSETS], [2.2 + o for o in OFFSETS]
    )


def test_an_iqm_tie_goes_to_the_higher_probability_of_improvement() -> None:
    scores = {
        "ppo-rnn": [1.0, 2.0, 3.0, 4.0],
        "sil": [0.0, 2.5, 2.5, 9.0],  # IQM 2.5, PoI 0.5 over the control
        "count-bonus-ppo-rnn": [2.4, 2.5, 2.5, 2.6],  # IQM 2.5, PoI 0.5 as well
    }
    controls = {"sil": "ppo-rnn", "count-bonus-ppo-rnn": "ppo-rnn"}
    ranking = rank_methods(scores, controls=controls)
    assert [r.registry for r in ranking] == ["count-bonus-ppo-rnn", "sil"]  # by name
    scores["sil"] = [1.5, 2.5, 2.5, 9.0]  # PoI 0.5625 now, same IQM
    ranking = rank_methods(scores, controls=controls)
    assert [r.registry for r in ranking] == ["sil", "count-bonus-ppo-rnn"]


def test_the_pair_skips_skill_methods_with_a_reason(
    store: Store, methods: Path, caplog: pytest.LogCaptureFixture
) -> None:
    ranking = rank_methods(_scores(store, methods), methods_dir=methods)
    with caplog.at_level(logging.INFO, logger=select_pair.__name__):
        pair = choose_pair(ranking, methods_dir=methods)
    assert [r.registry for r in pair] == ["count-bonus-ppo-rnn", "sil"]
    assert "smdp-skills: skill method" in caplog.text
    assert "dyna-q: skill method" in caplog.text


def test_no_compatible_pair_raises() -> None:
    controls = {"count-bonus-pqn-rnn": "pqn-rnn", "sil": "ppo-rnn", "dyna-q": "pqn-rnn"}
    scores = {"ppo-rnn": [1.0], "pqn-rnn": [1.0]} | {m: [2.0] for m in controls}
    with pytest.raises(NoPairError, match="no compatible pair"):
        choose_pair(rank_methods(scores, controls=controls))


def test_write_combo_fills_the_pair_and_its_tuned_knobs(
    store: Store, methods: Path
) -> None:
    ranking = rank_methods(_scores(store, methods), methods_dir=methods)
    write_combo(choose_pair(ranking, methods_dir=methods), ranking, methods_dir=methods)
    combo = load_method("combo", methods)
    assert combo.combine == {
        "base": "ppo-rnn",
        "methods": ["count-bonus-ppo-rnn", "sil"],
    }
    assert combo.base["base_from"] == "ppo-rnn.tuned"
    for registry, slug in (("count-bonus-ppo-rnn", "count-bonus"), ("sil", "sil")):
        method = load_method(slug, methods)
        tuned = resolve_config(method, registry, "1m", ref="tuned", methods_dir=methods)
        block = method.base if slug == "sil" else method.base["ppo-rnn"]
        own = set(block) - {"base_from"}
        assert {k: combo.base[k] for k in own} == {k: tuned[k] for k in own}
    assert combo.base["beta"] == 0.2
    assert combo.base["lambda_sil"] == 0.3
    assert combo.tuned is None
    assert [e["name"] for e in combo.coarse] == ["combined"]
    rules = combo.abort_rules("combo") or {}
    assert {"count/table_full", "sil/dominates"} <= set(rules["permanent"])
    names = [f["name"] for f in rules["flags"]]
    assert len(names) == len(set(names))
    assert "kl_high" in names
    text = (methods / "combo.yaml").read_text(encoding="utf-8")
    assert "1 smdp-skills" in text
    assert "TEMPLATE" not in text


def test_the_written_combo_runs_resolve_with_both_methods_knobs(
    store: Store, methods: Path
) -> None:
    ranking = rank_methods(_scores(store, methods), methods_dir=methods)
    write_combo(choose_pair(ranking, methods_dir=methods), ranking, methods_dir=methods)
    specs = expand_phase(REPO / "configs/phases/combo-1m.yaml", methods_dir=methods)
    assert len(specs) == 10
    cfg = specs[0].config
    assert (cfg["beta"], cfg["lambda_sil"], cfg["envs"]) == (0.2, 0.3, 256)


def test_a_pair_setting_one_key_two_ways_is_refused(methods: Path) -> None:
    tuned = TUNED | {"sil": {"lambda_sil": 0.3, "lr": 1.0e-4}}
    tuned["count-bonus"] = {"ppo-rnn": {"beta": 0.2, "lr": 3.0e-4}, "pqn-rnn": {}}
    _methods(methods, tuned)
    scores = {"ppo-rnn": [1.0], "sil": [3.0], "count-bonus-ppo-rnn": [2.0]}
    ranking = rank_methods(scores, methods_dir=methods)
    with pytest.raises(NoPairError, match="lr"):
        choose_pair(ranking, methods_dir=methods)


# --- CLI ------------------------------------------------------------------------


def _main(store: Store, methods: Path, *extra: str, phase: str = "final-1m") -> None:
    select_pair.main([
        "--phase",
        phase,
        "--tracking-uri",
        store.tracking_uri,
        "--methods-dir",
        str(methods),
        *extra,
    ])


def test_main_prints_the_ranking_and_writes_combo_yaml(
    store: Store, methods: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _main(store, methods)
    out = capsys.readouterr().out
    lines = [ln.split()[:2] for ln in out.splitlines() if ln[:1].isdigit()]
    assert lines[:3] == [
        ["1", "smdp-skills"],
        ["2", "dyna-q"],
        ["3", "count-bonus-ppo-rnn"],
    ]
    assert "pair: count-bonus-ppo-rnn + sil on ppo-rnn" in out
    assert load_method("combo", methods).combine["methods"] == [  # type: ignore[index]
        "count-bonus-ppo-rnn",
        "sil",
    ]


def test_main_dry_run_leaves_combo_yaml(store: Store, methods: Path) -> None:
    before = (methods / "combo.yaml").read_text(encoding="utf-8")
    _main(store, methods, "--dry-run")
    assert (methods / "combo.yaml").read_text(encoding="utf-8") == before


def test_main_exits_nonzero_without_a_compatible_pair(
    tmp_path: Path, methods: Path
) -> None:
    raw = yaml.safe_load(FINAL_1M.read_text(encoding="utf-8"))
    del raw["runs"]["sil"]
    phase = tmp_path / "no-sil.yaml"
    phase.write_text(yaml.safe_dump(raw))
    store = Store(tmp_path)
    _fill(store, methods, phase=phase)
    before = (methods / "combo.yaml").read_text(encoding="utf-8")
    with pytest.raises(SystemExit, match="no compatible pair") as exit_info:
        _main(store, methods, phase=str(phase))
    assert exit_info.value.code != 0
    assert (methods / "combo.yaml").read_text(encoding="utf-8") == before


def test_main_exits_nonzero_on_unfinished_runs(tmp_path: Path, methods: Path) -> None:
    store = Store(tmp_path)
    spec = expand_phase(FINAL_1M, method="sil", methods_dir=methods)[0]
    _fill(store, methods, skip=(spec.name,))
    with pytest.raises(SystemExit, match=spec.name):
        _main(store, methods)
