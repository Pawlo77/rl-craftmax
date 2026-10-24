"""Agent internals overlay: game frame | observation, action probabilities, value.

One eval step per frame: the decoded symbolic observation (map, mobs, inventory and
vitals), the policy's action distribution with the chosen action highlighted, and the
value estimate (the critic, or max Q for value-based methods) against the discounted
return actually obtained. Every registry method (`eval_rollouts.network_outputs`), both
games. Usage: ``python -m wut_murl_craftax.visualisation.overlay RUN --update 122``
"""

import argparse
import functools
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from craftax.craftax import constants as craftax_constants
from craftax.craftax_classic import constants as classic_constants
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap

from . import ffmpeg
from .charts import INK, MUTED, SERIES, SURFACE, figure, frame, style
from .eval_rollouts import (
    EVAL_SEEDS,
    MODES,
    Q_FAMILIES,
    EvalRun,
    Mode,
    episode_frames,
    episode_outputs,
    network_family,
    stored_episode,
)

logger = logging.getLogger(__name__)

CLASSIC_MOBS = ("zombie", "cow", "skeleton", "arrow")
"""Classic mob channels, in observation order."""
CRAFTAX_MOBS = ("melee", "passive", "ranged", "enemy shot", "own shot")
"""Craftax creature classes (8 type ids each, collapsed)."""
CLASSIC_BARS = (
    "wood",
    "stone",
    "coal",
    "iron",
    "diamond",
    "sapling",
    "wood pick",
    "stone pick",
    "iron pick",
    "wood sword",
    "stone sword",
    "iron sword",
    "health",
    "food",
    "drink",
    "energy",
)
"""Classic inventory then vitals, ``obs[1323:1339]``."""
CRAFTAX_BARS = {
    "wood": 8217,
    "stone": 8218,
    "coal": 8219,
    "iron": 8220,
    "diamond": 8221,
    "sapphire": 8222,
    "ruby": 8223,
    "sapling": 8224,
    "health": 8239,
    "food": 8240,
    "drink": 8241,
    "energy": 8242,
    "mana": 8243,
}
"""Craftax inventory and vitals shown, by flat observation index."""


@dataclass(frozen=True, slots=True)
class ObsView:
    """Decoded symbolic observations, one row per step."""

    blocks: np.ndarray
    """Block id per tile, (T, rows, cols)."""
    mobs: np.ndarray
    """Mob class present per tile, bool (T, rows, cols, classes)."""
    bars: np.ndarray
    """Inventory and vitals in 0..1, (T, K)."""
    bar_labels: tuple[str, ...]
    mob_labels: tuple[str, ...]


def decode_obs(obs: np.ndarray, classic: bool) -> ObsView:
    """Split flat symbolic observations ``(T, D)`` into map, mobs and bars."""
    obs = obs.astype(np.float32)
    if classic:
        rows, cols = classic_constants.OBS_DIM
        nb = len(classic_constants.BlockType)
        mp = obs[:, : rows * cols * (nb + 4)].reshape(-1, rows, cols, nb + 4)
        bars = obs[:, rows * cols * (nb + 4) :][:, : len(CLASSIC_BARS)]
        mobs = mp[..., nb:] > 0
        return ObsView(mp[..., :nb].argmax(-1), mobs, bars, CLASSIC_BARS, CLASSIC_MOBS)
    rows, cols = craftax_constants.OBS_DIM
    mp = obs[:, : rows * cols * 83].reshape(-1, rows, cols, 83)
    nb = len(craftax_constants.BlockType)
    mobs = mp[..., 42:82].reshape(*mp.shape[:3], 5, 8).max(-1) > 0
    bars = obs[:, list(CRAFTAX_BARS.values())]
    return ObsView(
        mp[..., :nb].argmax(-1), mobs, bars, tuple(CRAFTAX_BARS), CRAFTAX_MOBS
    )


@functools.cache
def block_colors(classic: bool) -> tuple[tuple[float, float, float], ...]:
    """Mean RGB of each block texture, so the map reads like the game."""
    consts = classic_constants if classic else craftax_constants
    tex = np.asarray(consts.load_all_textures()[16]["block_textures"], float)
    return tuple(tuple(c) for c in (tex.mean((1, 2)) / 255).clip(0, 1))


def discounted_return(reward: np.ndarray, n_alive: int, gamma: float) -> np.ndarray:
    """Return obtained from each step to the end of the episode (what V estimates)."""
    rtg = np.zeros(len(reward))
    acc = 0.0
    for t in range(n_alive - 1, -1, -1):
        acc = reward[t] + gamma * acc
        rtg[t] = acc
    return rtg


def action_names(classic: bool) -> list[str]:
    """Lower-case action names in id order (17 Classic, 43 Craftax)."""
    consts = classic_constants if classic else craftax_constants
    return [a.name.lower().replace("_", " ") for a in consts.Action]


@dataclass(frozen=True, slots=True)
class _Episode:
    """Per-step arrays of one stored eval episode, rebuilt by replay."""

    probs: np.ndarray
    value: np.ndarray
    reward: np.ndarray
    act: np.ndarray
    view: ObsView
    rtg: np.ndarray
    n_alive: int

    @classmethod
    def load(
        cls, run: EvalRun, update: int, seed: int, gamma: float, greedy: bool = False
    ) -> "_Episode":
        """Replay seed + actions; re-run the network on the observations."""
        ep, actions = stored_episode(run, update, seed, greedy)
        mode = "greedy" if greedy else "stochastic"
        probs, value = episode_outputs(run, update, mode, seed)
        n = len(actions)
        return cls(
            probs,
            value,
            ep.reward,
            actions.astype(int),
            decode_obs(ep.obs, run.classic),
            discounted_return(ep.reward, n, gamma),
            n,
        )


def value_label(run: EvalRun) -> str:
    """``max Q(s, a)`` for value-based methods (`Q_FAMILIES`), else ``V(s)``."""
    return "max Q(s, a)" if network_family(run.policy) in Q_FAMILIES else "V(s)"


def _header(run: EvalRun, update: int, mode: Mode, seed: int) -> str:
    """Panel title: method, checkpoint, eval mode and seed."""
    steps = run.env_steps(update) / 1e6
    return f"{run.method}, checkpoint {steps:.1f}M steps, {mode} eval seed {seed}"


class _Panel:
    """The internals figure; ``draw(t)`` updates its artists for step ``t``."""

    def __init__(
        self,
        run: EvalRun,
        ep: _Episode,
        size: tuple[int, int],
        s0: int,
        s1: int,
        header: str,
    ) -> None:
        self.ep, self.acts, fs = ep, action_names(run.classic), size[1] / 90
        self.value_label = value_label(run)
        self.ent = -(ep.probs * np.log(ep.probs + 1e-8)).sum(1)
        self.fig = fig = figure(*size)
        gs = fig.add_gridspec(
            3,
            2,
            height_ratios=[1.0, 1.45, 0.9],
            width_ratios=[1.2, 1],
            hspace=0.45,
            wspace=0.55,
            left=0.2,
            right=0.96,
            top=0.93,
            bottom=0.07,
        )
        self._observation(fig.add_subplot(gs[0, 0]), run.classic, fs)
        ax_inv = style(fig.add_subplot(gs[0, 1]), fs * 0.7)
        labels = ep.view.bar_labels
        self.inv = ax_inv.barh(np.arange(len(labels)), 0, color=SERIES[2], height=0.7)
        ax_inv.set_yticks(np.arange(len(labels)), labels, fontsize=fs * 0.6)
        ax_inv.invert_yaxis()
        ax_inv.set_xlim(0, 1)
        ax_inv.set_title("inventory, vitals", loc="left", fontsize=fs, color=INK)
        ax_pi = style(fig.add_subplot(gs[1, :]), fs * 0.8)
        self.pi = ax_pi.barh(np.arange(len(self.acts)), 0, color=SERIES[0], height=0.75)
        tick_fs = fs * (0.8 if run.classic else 0.4)
        ax_pi.set_yticks(np.arange(len(self.acts)), self.acts, fontsize=tick_fs)
        ax_pi.invert_yaxis()
        ax_pi.set_xlim(0, 1)
        ax_pi.tick_params(axis="y", length=0)
        self.pi_title = ax_pi.set_title("", loc="left", fontsize=fs, color=INK)
        self._value(style(fig.add_subplot(gs[2, :]), fs), fs, s0, s1)
        fig.text(0.02, 0.975, header, fontsize=size[1] / 80, color=INK)

    def _observation(self, ax: Axes, classic: bool, fs: float) -> None:
        view = self.ep.view
        cmap = ListedColormap(block_colors(classic))
        self.im = ax.imshow(
            view.blocks[0], cmap=cmap, vmin=0, vmax=cmap.N - 1, interpolation="nearest"
        )
        rows, cols = view.blocks.shape[1:]
        ax.plot(
            [cols // 2], [rows // 2], marker="X", ms=fs * 0.9, color="#ffffff", mec="k"
        )
        self.mob_sc = [
            ax.scatter(
                [], [], s=fs * 3, c=SERIES[k % 4], edgecolors="k", linewidths=0.5
            )
            for k in range(len(view.mob_labels))
        ]
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            "observation: map + mobs, X = agent", loc="left", fontsize=fs, color=INK
        )

    def _value(self, ax: Axes, fs: float, s0: int, s1: int) -> None:
        ep, ts = self.ep, np.arange(self.ep.n_alive)
        self.ts = ts
        ax.plot(
            ts,
            ep.rtg[: ep.n_alive],
            color=MUTED,
            lw=1.2,
            label="discounted return obtained",
        )
        (self.v_line,) = ax.plot(
            [], [], color=SERIES[0], lw=2, label=f"estimate {self.value_label}"
        )
        self.cursor = ax.axvline(s0, color=MUTED, lw=1, ls=":")
        lo = min(ep.rtg[s0:s1].min(), ep.value[s0:s1].min())
        span = max(ep.rtg[s0:s1].max(), ep.value[s0:s1].max()) - lo + 1e-6
        ax.set_xlim(s0, max(s1, s0 + 1))
        ax.set_ylim(lo - 0.1 * span, lo + 1.25 * span)
        ax.legend(
            loc="upper right", fontsize=fs * 0.6, frameon=False, ncol=2, labelcolor=INK
        )
        self.v_title = ax.set_title("", loc="left", fontsize=fs, color=INK)
        self.s0 = s0

    def draw(self, t: int) -> np.ndarray:
        """Pixels of the panel at step ``t``."""
        ep = self.ep
        self.im.set_data(ep.view.blocks[t])
        for k, sc in enumerate(self.mob_sc):
            yy, xx = np.nonzero(ep.view.mobs[t, :, :, k])
            sc.set_offsets(np.c_[xx, yy] if len(xx) else np.empty((0, 2)))
        for b, v in zip(self.inv, ep.view.bars[t].clip(0, 1), strict=True):
            b.set_width(float(v))
        for k, b in enumerate(self.pi):
            b.set_width(float(ep.probs[t, k]))
            b.set_color(SERIES[1] if k == ep.act[t] else SERIES[0])
        chosen = self.acts[ep.act[t]]
        self.pi_title.set_text(
            f"policy: chosen {chosen}, entropy {self.ent[t]:.2f} nats"
        )
        self.v_line.set_data(self.ts[self.s0 : t + 1], ep.value[self.s0 : t + 1])
        self.cursor.set_xdata([t, t])
        self.v_title.set_text(
            f"{self.value_label} = {ep.value[t]:.2f}   reward r = {ep.reward[t]:+.1f}"
        )
        return frame(self.fig)


def render_overlay(
    run_dir: Path,
    update: int,
    seed: int = EVAL_SEEDS[0],
    steps: tuple[int, int] = (0, 300),
    fps: int = 10,
    game_px: int = 1080,
    panel: tuple[int, int] = (840, 1080),
    gamma: float = 0.99,
    out: Path = Path("outputs/visualisation/overlay.mp4"),
    mode: Mode = "stochastic",
) -> Path:
    """Game frame (``game_px`` square, letterboxed) beside the internals panel."""
    run = EvalRun.load(run_dir)
    work = out.with_suffix("")
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ep = _Episode.load(run, update, seed, gamma, mode == "greedy")
    s0, s1 = steps[0], min(steps[1], ep.n_alive)
    fig = _Panel(run, ep, panel, s0, s1, _header(run, update, mode, seed))
    with ffmpeg.FrameWriter(work / "internals.mkv", *panel, fps=fps) as writer:
        for t in range(s0, s1):  # streamed: 300 panels at 840 x 1080 would be 0.8 GB
            writer.write(fig.draw(t))
    frames = episode_frames(run, update, seed, greedy=mode == "greedy")[s0:s1]
    h, w = frames.shape[1:3]
    with ffmpeg.FrameWriter(work / "game.mkv", w, h, fps=fps) as writer:
        writer.write(frames)
    g, ph = game_px, panel[1]
    graph = (
        f"[0:v]scale={g}:{g}:force_original_aspect_ratio=decrease:flags=neighbor,"
        f"pad={g}:{ph}:(ow-iw)/2:(oh-ih)/2:color={SURFACE}[g];"
        f"[g][1:v]hstack=inputs=2,format=yuv420p[v]"
    )
    ffmpeg.run([
        "-i",
        work / "game.mkv",
        "-i",
        work / "internals.mkv",
        "-filter_complex",
        graph,
        "-map",
        "[v]",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-movflags",
        "+faststart",
        out,
    ])
    logger.info("overlay %s: %d frames in %.1fs", out, s1 - s0, time.time() - t0)
    return out


def main(argv: list[str] | None = None) -> None:
    """Command line: ``RUN --update U [--seed S] [--mode greedy] [--steps 0:300]``."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run", type=Path)
    p.add_argument("--update", type=int, required=True)
    p.add_argument("--seed", type=int, default=EVAL_SEEDS[0])
    p.add_argument("--steps", default="0:300")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--mode", choices=MODES, default="stochastic")
    p.add_argument("--game-px", type=int, default=1080)
    p.add_argument("--panel", default="840x1080")
    p.add_argument(
        "--out", type=Path, default=Path("outputs/visualisation/overlay.mp4")
    )
    a = p.parse_args(argv)
    s0, s1 = map(int, a.steps.split(":"))
    pw, ph = map(int, a.panel.split("x"))
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    render_overlay(
        a.run,
        a.update,
        a.seed,
        (s0, s1),
        a.fps,
        a.game_px,
        (pw, ph),
        out=a.out,
        mode=a.mode,
    )
    print(f"overlay written to {a.out}")


if __name__ == "__main__":
    main()
