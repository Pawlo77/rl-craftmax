"""Tests for the agent internals overlay."""

from pathlib import Path

import jax
import numpy as np
import pytest
from _learning_run import EVAL_SEEDS, TINY, needs_ffmpeg, tiny_run
from _method_runs import FAMILY_RUNS, UPDATES, method_run
from craftax.craftax_env import make_craftax_env_from_name

from wut_murl_craftax.training.ppo import make_network, policy_value
from wut_murl_craftax.visualisation import ffmpeg
from wut_murl_craftax.visualisation.eval_rollouts import (
    EvalRun,
    episode_outputs,
    load_eval,
)
from wut_murl_craftax.visualisation.overlay import (
    _Episode,
    _header,
    _Panel,
    block_colors,
    decode_obs,
    discounted_return,
    render_overlay,
)


@pytest.mark.parametrize(
    ("env_name", "classic", "rows", "cols", "n_mobs"),
    [
        ("Craftax-Classic-Symbolic-v1", True, 7, 9, 4),
        ("Craftax-Symbolic-v1", False, 9, 11, 5),
    ],
)
def test_decode_obs_matches_the_map_view(
    env_name: str, classic: bool, rows: int, cols: int, n_mobs: int
) -> None:
    env = make_craftax_env_from_name(env_name, auto_reset=False)
    obs, _ = env.reset(jax.random.PRNGKey(0), env.default_params)
    view = decode_obs(np.asarray(obs)[None], classic)
    assert view.blocks.shape == (1, rows, cols)
    assert view.mobs.shape == (1, rows, cols, n_mobs)
    assert len(view.bar_labels) == view.bars.shape[1]
    assert view.bars.min() >= 0
    assert len(block_colors(classic)) > view.blocks.max()
    # the agent stands on a walkable tile at the centre of the view
    assert view.blocks[0, rows // 2, cols // 2] > 1


def test_discounted_return_stops_at_episode_end() -> None:
    r = np.array([1.0, 0.0, 1.0, 5.0])
    np.testing.assert_allclose(discounted_return(r, 3, 0.5), [1.25, 0.5, 1.0, 0.0])


@pytest.mark.parametrize("greedy", [False, True])
def test_episode_replays_observations_and_reruns_the_network(greedy: bool) -> None:
    run, seed = EvalRun.load(tiny_run()), EVAL_SEEDS[0]
    ev = load_eval(run.dir, 4, greedy)
    n = int(ev["length"][0])
    ep = _Episode.load(run, 4, seed, 0.99, greedy)
    # an independent replay: the raw env and the fold_in key scheme
    net, env, env_params = make_network(TINY.env, TINY.width, auto_reset=False)
    base = jax.random.PRNGKey(seed)
    obs, state = env.reset(base, env_params)
    step = jax.jit(env.step)
    seen, rewards = [], []
    for t, a in enumerate(ev["actions"][0, :n].tolist()):
        seen.append(np.asarray(obs))
        k = jax.random.fold_in(jax.random.fold_in(base, 1), t)
        obs, state, r, _, _ = step(k, state, a, env_params)
        rewards.append(float(r))
    logits, value = policy_value(net, run.params(4), np.stack(seen))
    assert ep.n_alive == n
    np.testing.assert_array_equal(ep.act, ev["actions"][0, :n])
    np.testing.assert_allclose(ep.reward, rewards, atol=1e-6)
    np.testing.assert_allclose(ep.probs, jax.nn.softmax(logits), atol=1e-6)
    np.testing.assert_allclose(ep.value, value, atol=1e-6)
    np.testing.assert_array_equal(
        ep.view.blocks, decode_obs(np.stack(seen), True).blocks
    )


@needs_ffmpeg
def test_overlay_is_game_plus_panel_one_frame_per_step(tmp_path: Path) -> None:
    out = render_overlay(
        tiny_run(),
        update=4,
        seed=EVAL_SEEDS[0],
        steps=(0, 6),
        fps=5,
        game_px=144,
        panel=(200, 144),
        out=tmp_path / "o.mp4",
    )
    info = ffmpeg.probe(out)["streams"][0]
    assert (info["width"], info["height"]) == (344, 144)
    assert ffmpeg.count_frames(out) == 6


VALUE_LABELS = {
    "ppo": "V(s)",
    "ppo-rnn": "V(s)",
    "pqn-rnn": "max Q(s, a)",
    "smdp-skills": "max Q(s, a)",
}
"""The value panel's quantity: the critic, or max Q for value-based methods."""


@pytest.mark.parametrize("name", FAMILY_RUNS)
def test_the_panel_names_the_method_and_its_value(name: str) -> None:
    run, u, seed = EvalRun.load(method_run(name)), UPDATES[name], EVAL_SEEDS[0]
    ep = _Episode.load(run, u, seed, 0.99, greedy=True)
    probs, value = episode_outputs(run, u, "greedy", seed)
    np.testing.assert_array_equal(ep.probs, probs)
    np.testing.assert_array_equal(ep.value, value)
    header = _header(run, u, "greedy", seed)
    assert header.startswith(f"{name}, checkpoint ")
    panel = _Panel(run, ep, (200, 144), 0, min(3, ep.n_alive), header)
    assert [t.get_text() for t in panel.fig.texts] == [header]
    panel.draw(1)
    label = VALUE_LABELS[name]
    assert panel.v_title.get_text().startswith(f"{label} = ")
    legend = panel.v_line.axes.get_legend()
    assert f"estimate {label}" in [t.get_text() for t in legend.get_texts()]


@needs_ffmpeg
@pytest.mark.parametrize("name", FAMILY_RUNS)
def test_every_network_family_renders_an_overlay(tmp_path: Path, name: str) -> None:
    out = render_overlay(
        method_run(name),
        update=UPDATES[name],
        steps=(0, 3),
        fps=5,
        game_px=144,
        panel=(200, 144),
        out=tmp_path / "o.mp4",
        mode="greedy",
    )
    assert ffmpeg.count_frames(out) == 3
