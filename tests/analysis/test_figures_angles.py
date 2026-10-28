"""Tests of the analysis-angle figures F21-F28 on toy frames (`_angles_fixture`).

Every figure writes a PNG, is sized like the other comparison figures, keeps its legend
off the axes and its text inside the canvas, and its artists (``gid``
``<kind>/<budget>/<method>[/<key>]``) carry the numbers of the toy case, computed here
a second way.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from _angles_fixture import (
    DISTILLED,
    FEATURES,
    GAP,
    RAMP,
    REVERSED,
    SLOT_SCORE,
    UPDATES,
    angle_frames,
    checkpoint_steps,
    fidelity_of,
    importance_of,
    write_distillation,
)
from _compare_fixture import LEVEL
from matplotlib.artist import Artist
from matplotlib.collections import PathCollection, QuadMesh
from matplotlib.figure import Figure
from matplotlib.text import Text
from scipy import stats as sps

from wut_murl_craftax.analysis import compare
from wut_murl_craftax.analysis.figures_angles import (
    ANGLE_FIGURES,
    GATE_MIN,
    NAMES,
    gatekeepers,
    greedy_gap,
    plot_ablation_effects,
    plot_distil_features,
    plot_distil_score,
    plot_episode_spread,
    plot_gatekeepers,
    plot_greedy_gap,
    plot_seed_fate,
    plot_tuning_optimism,
    seed_fate,
)
from wut_murl_craftax.analysis.load import final_checkpoint
from wut_murl_craftax.evaluation.hardware import reused_from_tuning
from wut_murl_craftax.evaluation.stats import stratified_bootstrap_ci

N = 100
"""Bootstrap resamples in tests."""
PNG = b"\x89PNG\r\n\x1a\n"
FIGS = [f"F{i}" for i in range(21, 29)]
LABELS = {
    "F21": ("env steps", "greedy minus stochastic"),
    "F22": ("later unlocks", "alive"),
    "F23": ("env steps", "Spearman"),
    "F24": ("ablation minus tuned",),
    "F25": ("tuning mean", "fresh-seed final mean"),
    "F26": ("achievements per episode", "episode length"),
    "F27": ("fidelity at 256 leaves", "final score"),
    "F28": (),
}
"""Substrings the axis labels of a figure must contain."""


@pytest.fixture(scope="module")
def frames() -> dict[str, pd.DataFrame]:
    return angle_frames()


@pytest.fixture(scope="module")
def frames_1m() -> dict[str, pd.DataFrame]:
    return angle_frames(("1m",))


@pytest.fixture(scope="module")
def trees(
    frames: dict[str, pd.DataFrame], tmp_path_factory: pytest.TempPathFactory
) -> Path:
    return write_distillation(tmp_path_factory.mktemp("interpretation"), frames["runs"])


def optimism_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """R101/R102 rows: sil and dyna-q with fresh seeds, ppo n/a (as the store's)."""
    runs = frames["runs"]
    rows = []
    for method, tuning in (("sil", 2.6), ("dyna-q", 1.0), ("ppo", 2.0)):
        final = runs[
            (runs["method"] == method)
            & (runs["config_ref"] == "tuned")
            & (runs["budget"] == "1m")
        ]
        fresh = final[~final["phases"].map(reused_from_tuning)]
        na = method == "ppo"
        rows.append({
            "benchmark": "1m", "block": compare.block_of(method, None),
            "method": method, "role": "control" if na else "method",
            "config_hash": "h", "tuning_phase": "tune-fine", "tuning_seeds": 3,
            "tuning_mean": tuning, "final_seeds": len(final),
            "fresh_seeds": 0 if na else len(fresh),
            "fresh_mean": np.nan if na else fresh["final_score"].mean(),
            "optimism": np.nan if na else tuning - fresh["final_score"].mean(),
            "optimism_note": "n/a: selected on all its final seeds" if na else "",
        })  # fmt: skip
    return pd.DataFrame(rows, columns=compare.TUNING_OPTIMISM_COLUMNS)


def draw(
    fid: str,
    frames: dict[str, pd.DataFrame],
    out_dir: Path,
    theme: str = "report",
    width: str = "column",
    trees: Path | None = None,
) -> Figure:
    """Call the figure of ``fid`` with test-sized bootstraps."""
    b = {"width": width, "n_boot": N}
    root = trees or out_dir / "no-interpretation"
    calls: dict[str, Callable[[], Figure]] = {
        "F21": lambda: plot_greedy_gap(frames, out_dir, theme, **b),
        "F22": lambda: plot_gatekeepers(frames, out_dir, theme, **b),
        "F23": lambda: plot_seed_fate(frames, out_dir, theme, width=width),
        "F24": lambda: plot_ablation_effects(frames, out_dir, theme, **b),
        "F25": lambda: plot_tuning_optimism(
            frames, out_dir, theme, table=optimism_table(frames), **b
        ),
        "F26": lambda: plot_episode_spread(frames, out_dir, theme, width=width),
        "F27": lambda: plot_distil_score(
            frames, out_dir, theme, interpretation=root, width=width
        ),
        "F28": lambda: plot_distil_features(
            frames, out_dir, theme, interpretation=root, width=width
        ),
    }
    return calls[fid]()


def artists(fig: Figure, kind: str) -> list[Artist]:
    return fig.findobj(lambda a: (a.get_gid() or "").startswith(f"{kind}/"))


def by_gid(fig: Figure, gid: str) -> Artist:
    found = fig.findobj(lambda a: a.get_gid() == gid)
    assert len(found) == 1, (gid, len(found))
    return found[0]


def said(fig: Figure) -> str:
    fig.canvas.draw()
    shown = [t.get_text() for t in fig.findobj(Text) if t.get_visible()]
    return " ".join(" ".join(shown).split())


def axis_labels(fig: Figure) -> str:
    return " | ".join(f"{ax.get_xlabel()} | {ax.get_ylabel()}" for ax in fig.axes)


def pos(method: str) -> int:
    return list(LEVEL).index(method)


# --- every figure -------------------------------------------------------------------


def test_every_figure_id_has_a_function_and_a_stem() -> None:
    assert list(ANGLE_FIGURES) == FIGS
    assert set(NAMES) == set(FIGS)
    assert all(f.__name__.startswith("plot_") for f in ANGLE_FIGURES.values())


@pytest.mark.parametrize("theme", ["report", "video"])
@pytest.mark.parametrize("fid", FIGS)
def test_each_figure_writes_a_png_with_its_labels(
    fid: str, theme: str, frames: dict[str, pd.DataFrame], trees: Path, tmp_path: Path
) -> None:
    fig = draw(fid, frames, tmp_path, theme, trees=trees)
    name = NAMES[fid] + ("_video" if theme == "video" else "")
    assert (tmp_path / f"{name}.png").read_bytes().startswith(PNG)
    for want in LABELS[fid]:
        assert want in axis_labels(fig), (fid, want, axis_labels(fig))


@pytest.mark.parametrize("fid", FIGS)
def test_sizes_match_the_other_figures(
    fid: str, frames: dict[str, pd.DataFrame], trees: Path, tmp_path: Path
) -> None:
    fig = draw(fid, frames, tmp_path, trees=trees)
    assert fig.get_size_inches()[0] * 25.4 == pytest.approx(80.0, abs=0.01)
    fig.canvas.draw()
    small = [
        t.get_text()
        for t in fig.findobj(Text)
        if t.get_visible() and t.get_text().strip() and t.get_fontsize() < 7
    ]
    assert not small, small
    wide = draw(fid, frames, tmp_path, width="double", trees=trees)
    assert wide.get_size_inches()[0] * 25.4 == pytest.approx(170.0, abs=0.01)
    assert (tmp_path / f"{NAMES[fid]}_double.png").is_file()
    video = draw(fid, frames, tmp_path, "video", trees=trees)
    w, h = video.get_size_inches() * video.dpi
    assert (round(w), round(h)) == (1920, 1080)


@pytest.mark.parametrize("width", ["column", "double"])
@pytest.mark.parametrize("fid", FIGS)
def test_legends_stay_off_the_axes_and_text_inside_the_canvas(
    fid: str, width: str, frames: dict[str, pd.DataFrame], trees: Path, tmp_path: Path
) -> None:
    fig = draw(fid, frames, tmp_path, width=width, trees=trees)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
    legends = [*fig.legends, *(ax.get_legend() for ax in fig.axes)]
    boxes = [ax.get_window_extent(renderer) for ax in fig.axes if ax.axison]
    for legend in filter(None, legends):
        box = legend.get_window_extent(renderer)
        assert not any(box.overlaps(b) for b in boxes), fid
    inches = fig.get_size_inches()
    drawn = fig.get_tightbbox(renderer)  # inches; only what is drawn
    assert min(drawn.x0, drawn.y0) >= -0.01, (fid, drawn)
    assert drawn.x1 <= inches[0] + 0.01, (fid, drawn)
    assert drawn.y1 <= inches[1] + 0.01, (fid, drawn)


@pytest.mark.parametrize("fid", FIGS)
def test_a_missing_1b_is_pending(
    fid: str, frames_1m: dict[str, pd.DataFrame], trees: Path, tmp_path: Path
) -> None:
    fig = draw(fid, frames_1m, tmp_path, trees=trees)
    assert "Craftax-1B pending" in said(fig)


def test_an_unknown_theme_or_width_is_refused(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="theme"):
        plot_greedy_gap(frames, tmp_path, "poster", n_boot=N)
    with pytest.raises(ValueError, match="width"):
        plot_greedy_gap(frames, tmp_path, width="triple", n_boot=N)


# --- F21 greedy vs stochastic -------------------------------------------------------


def test_greedy_gap_is_greedy_minus_stochastic_per_run_and_checkpoint(
    frames: dict[str, pd.DataFrame],
) -> None:
    table = greedy_gap(frames)
    runs = frames["runs"]
    assert set(table["run_id"]) == set(
        runs.loc[runs["config_ref"] == "tuned", "run_id"]
    )
    row = table[(table["run_id"] == "sil-tuned-1m-4") & (table["update"] == 54)].iloc[0]
    final = runs.set_index("run_id").loc["sil-tuned-1m-4", "final_score"]
    assert row["stochastic"] == pytest.approx(final * RAMP[54] + 1.5 * SLOT_SCORE)
    assert row["gap"] == pytest.approx(-GAP * pos("sil"))
    assert row["env_steps"] == checkpoint_steps("1m", 54)


def test_the_gap_line_is_the_mean_gap_over_seeds(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    fig = plot_greedy_gap(frames, tmp_path, n_boot=N)
    for method in LEVEL:
        line = by_gid(fig, f"gap/1m/{method}")
        steps = [checkpoint_steps("1m", u) for u in UPDATES]
        np.testing.assert_allclose(line.get_xdata(), steps)  # type: ignore[attr-defined]
        np.testing.assert_allclose(
            line.get_ydata(),
            -GAP * pos(method),
            atol=1e-9,  # type: ignore[attr-defined]
        )
    assert by_gid(fig, "band/1m/sil")
    assert {a.get_gid() for a in artists(fig, "gap")} >= {"gap/1b/ppo"}


# --- F22 gatekeepers ----------------------------------------------------------------


def naive_lift(eps: pd.DataFrame, method: str, run: str, ach: str) -> float:
    """The lift of `gatekeepers` by a plain loop over the episodes."""
    names = [c.removeprefix("ach/") for c in eps.columns if c.startswith("ach/")]
    mine = eps[eps["method"] == method]
    at = float(np.median(mine.loc[mine[f"ach/{ach}"], f"unlock_step/{ach}"]))
    groups: dict[bool, list[int]] = {True: [], False: []}
    for _, e in mine[(mine["run_id"] == run) & (mine["length"] > at)].iterrows():
        later = sum(
            bool(e[f"ach/{b}"]) and e[f"unlock_step/{b}"] > at
            for b in names
            if b != ach
        )
        groups[bool(e[f"ach/{ach}"])].append(later)
    if not groups[True] or not groups[False]:
        return float("nan")
    return float(np.mean(groups[True]) - np.mean(groups[False]))


def test_gatekeepers_lift_matches_a_plain_loop(
    frames: dict[str, pd.DataFrame],
) -> None:
    eps = final_checkpoint(frames["eval_episodes"])
    eps = eps[(eps["mode"] == "stochastic") & (eps["budget"] == "1m")]
    table = gatekeepers(eps)
    assert table["lift"].notna().any()
    for r in table.sample(25, random_state=0).itertuples():
        want = naive_lift(eps, r.method, r.run_id, r.achievement)
        if np.isnan(want):
            assert np.isnan(r.lift)
        else:
            assert r.lift == pytest.approx(want)


def test_episodes_dead_before_the_median_unlock_are_left_out() -> None:
    """One run: a at steps 10 and 30 (median 20); a quick death without a is dropped."""
    eps = pd.DataFrame({
        "method": "sil",
        "run_id": "r",
        "length": [100, 100, 100, 15],
        "ach/a": [True, True, False, False],
        "unlock_step/a": [10, 30, -1, -1],
        "ach/b": [True, False, True, True],
        "unlock_step/b": [20, -1, 50, 5],
        "ach/c": [False, True, False, False],
        "unlock_step/c": [-1, 40, -1, -1],
    })
    row = gatekeepers(eps).set_index("achievement").loc["a"]
    assert row["at"] == 20
    assert (row["with_a"], row["without_a"]) == (2, 1)
    assert row["lift"] == pytest.approx(0.5 - 1.0)  # with the dead one it would be 0


def test_a_gatekeeper_dot_is_the_mean_lift_over_seeds(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    fig = plot_gatekeepers(frames, tmp_path, n_boot=N)
    dots = [d for d in artists(fig, "gate") if str(d.get_gid()).split("/")[1] == "1m"]
    assert dots
    eps = final_checkpoint(frames["eval_episodes"])
    eps = eps[(eps["mode"] == "stochastic") & (eps["config_ref"] == "tuned")]
    table = gatekeepers(eps[eps["budget"] == "1m"])
    for dot in dots:
        _, _, method, ach = str(dot.get_gid()).split("/")
        seeds = table[(table["method"] == method) & (table["achievement"] == ach)]
        assert seeds[["with_a", "without_a"]].sum().min() >= GATE_MIN
        want = seeds["lift"].mean()
        assert float(dot.get_xdata()[0]) == pytest.approx(want)  # type: ignore[attr-defined]


# --- F23 fate -----------------------------------------------------------------------


def test_seed_fate_is_the_spearman_of_checkpoint_and_final_score(
    frames: dict[str, pd.DataFrame],
) -> None:
    table = seed_fate(frames).set_index(["budget", "method", "update"])["rho"]
    for method in LEVEL:
        first = table[("1m", method, UPDATES[0])]
        assert first == pytest.approx(-1.0 if method == REVERSED else 1.0)
        assert table[("1m", method, UPDATES[1])] == pytest.approx(1.0)
    runs = frames["runs"]
    mine = runs[(runs["method"] == "dyna-q") & (runs["config_ref"] == "tuned")]
    mine = mine[mine["budget"] == "1b"].sort_values("seed")
    want = sps.spearmanr(mine["final_score"] * RAMP[4], mine["final_score"]).statistic
    assert table[("1b", "dyna-q", 4)] == pytest.approx(want)


def test_the_fate_lines_and_the_pooled_mean(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    fig = plot_seed_fate(frames, tmp_path)
    line = by_gid(fig, f"fate/1m/{REVERSED}")
    np.testing.assert_allclose(line.get_ydata(), [-1.0, 1.0, 1.0])  # type: ignore[attr-defined]
    pooled = by_gid(fig, "fate/1m/pooled")
    first = (len(LEVEL) - 2) / len(LEVEL)  # one -1 among nine
    assert pooled.get_ydata()[0] == pytest.approx(first)  # type: ignore[attr-defined]
    t = sps.t.ppf(0.975, 8)  # 10 seeds: the 5% two-sided critical rho
    null = by_gid(fig, "null/1m")
    top = null.get_y() + null.get_height()  # type: ignore[attr-defined]
    assert top == pytest.approx(t / np.sqrt(8 + t**2))


# --- F24 ablations ------------------------------------------------------------------


def test_ablation_effects_are_the_ablation_table_differences(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    table = compare.ablation_table(frames, n_boot=N)
    fig = plot_ablation_effects(frames, tmp_path, n_boot=N, table=table)
    for r in table.itertuples():
        key = f"{r.benchmark}/{r.method}/{r.config_ref}"
        dot = by_gid(fig, f"effect/{key}")
        assert float(dot.get_xdata()[0]) == pytest.approx(r.diff)  # type: ignore[attr-defined]
        seeds = by_gid(fig, f"seeds/{key}")
        assert len(seeds.get_offsets()) == r.seeds  # type: ignore[attr-defined]
    assert "PoI" in said(fig)


# --- F25 tuning optimism ------------------------------------------------------------


def test_tuning_optimism_points_and_their_fresh_intervals(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    table = optimism_table(frames)
    fig = plot_tuning_optimism(frames, tmp_path, n_boot=N, table=table)
    runs = frames["runs"]
    for r in table[table["optimism"].notna()].itertuples():
        point = by_gid(fig, f"optimism/1m/{r.method}")
        assert float(point.get_xdata()[0]) == pytest.approx(r.tuning_mean)  # type: ignore[attr-defined]
        assert float(point.get_ydata()[0]) == pytest.approx(r.fresh_mean)  # type: ignore[attr-defined]
        mine = runs[
            (runs["method"] == r.method)
            & (runs["config_ref"] == "tuned")
            & (runs["budget"] == "1m")
        ]
        fresh = mine.loc[~mine["phases"].map(reused_from_tuning), "final_score"]
        ci = stratified_bootstrap_ci(
            {"seeds": fresh.tolist()}, lambda d: float(np.mean(d["seeds"])), N, 0
        )
        seg = by_gid(fig, f"ci/1m/{r.method}").get_segments()[0]  # type: ignore[attr-defined]
        assert sorted(seg[:, 1]) == pytest.approx([ci.lo, ci.hi])
    text = said(fig)
    assert "ppo" in text
    assert "selected on all its final seeds" in text


# --- F26 episode spread -------------------------------------------------------------


def test_episode_spread_shares_and_length_profile(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    fig = plot_episode_spread(frames, tmp_path)
    mesh = by_gid(fig, "heat/1m/achievements")
    assert isinstance(mesh, QuadMesh)
    grid = np.ma.filled(mesh.get_array(), np.nan).reshape(len(LEVEL), -1) / 100
    rows = [t.get_text() for t in mesh.axes.get_yticklabels()]
    eps = final_checkpoint(frames["eval_episodes"])
    eps = eps[(eps["mode"] == "stochastic") & (eps["budget"] == "1m")]
    eps = eps[eps["config_ref"] == "tuned"]
    for i, method in enumerate(rows):
        counts = eps.loc[eps["method"] == method, "n_achievements"]
        share = counts.value_counts(normalize=True)
        for k, s in share.items():
            assert grid[i, int(k)] == pytest.approx(s)
        assert np.nansum(grid[i]) == pytest.approx(1.0)
    assert artists(fig, "length")


# --- F27 and F28 distillation -------------------------------------------------------


def test_distil_score_points_and_the_spearman(
    frames: dict[str, pd.DataFrame], trees: Path, tmp_path: Path
) -> None:
    fig = plot_distil_score(frames, tmp_path, interpretation=trees)
    runs = frames["runs"]
    tuned = runs[(runs["config_ref"] == "tuned") & (runs["budget"] == "1m")]
    xs, ys = [], []
    for method in DISTILLED:
        dots = by_gid(fig, f"distil/1m/{method}")
        assert isinstance(dots, PathCollection)
        mine = tuned[tuned["method"] == method].sort_values("seed")
        want = np.c_[
            [fidelity_of(method, s) for s in mine["seed"]], mine["final_score"]
        ]
        np.testing.assert_allclose(
            np.asarray(sorted(map(tuple, dots.get_offsets()))), sorted(map(tuple, want))
        )
        xs += list(want[:, 0])
        ys += list(want[:, 1])
    rho = sps.spearmanr(xs, ys).statistic
    text = said(fig)
    assert f"{rho:.2f}" in text
    for want in ("pending", "pqn-rnn", "combo"):
        assert want in text


def test_distil_features_heat_map_is_the_mean_importance(
    frames: dict[str, pd.DataFrame], trees: Path, tmp_path: Path
) -> None:
    fig = plot_distil_features(frames, tmp_path, interpretation=trees)
    mesh = by_gid(fig, "heat/1m/importance")
    assert isinstance(mesh, QuadMesh)
    grid = np.ma.filled(mesh.get_array(), np.nan).reshape(len(FEATURES), len(DISTILLED))
    rows = [t.get_text() for t in mesh.axes.get_yticklabels()]
    cols = [t.get_text() for t in mesh.axes.get_xticklabels()]
    assert sorted(rows) == sorted(FEATURES)
    assert sorted(cols) == sorted(DISTILLED)
    seeds = range(10)
    for i, f in enumerate(rows):
        for j, m in enumerate(cols):
            want = np.mean([importance_of(m, s, f) for s in seeds])
            if np.isnan(want):
                assert np.isnan(grid[i, j]), (f, m)
            else:
                assert grid[i, j] == pytest.approx(want), (f, m)


def test_distillation_figures_without_any_policy_say_so(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    for fn in (plot_distil_score, plot_distil_features):
        fig = fn(frames, tmp_path, interpretation=tmp_path / "none")
        assert "pending" in said(fig)
