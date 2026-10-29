"""Tests of the analysis-angle figures F21-F28 on toy frames (`_angles_fixture`).

Every figure writes a PNG, is sized like the other comparison figures, keeps its legend
off the axes and its text inside the canvas, and its artists (``gid``
``<kind>/<budget>/<method>[/<key>]``) carry the numbers of the toy case, computed here
a second way.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

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
from _figures_fixture import Built, figure_cache
from matplotlib.artist import Artist
from matplotlib.collections import PathCollection, QuadMesh
from matplotlib.figure import Figure
from matplotlib.text import Text
from scipy import stats as sps

from wut_murl_craftax.analysis import compare, figures_angles
from wut_murl_craftax.analysis.figures_angles import (
    ANGLE_FIGURES,
    GATE_MIN,
    NAMES,
    STEMS,
    gatekeepers,
    greedy_gap,
    plot_ablation_effects,
    plot_achievements_vs_length,
    plot_distil_features,
    plot_distil_score,
    plot_episode_spread,
    plot_gatekeepers,
    plot_greedy_gap,
    plot_seed_fate,
    plot_tuning_optimism,
    seed_fate,
)
from wut_murl_craftax.analysis.load import final_checkpoint, select
from wut_murl_craftax.evaluation.hardware import reused_from_tuning
from wut_murl_craftax.evaluation.stats import stratified_bootstrap_ci
from wut_murl_craftax.visualisation.charts import (
    MAX_ROWS_MM,
    Caption,
    Shape,
    capture_captions,
)

N = 100
"""Bootstrap resamples in tests."""
FIGS = [f"F{i}" for i in range(21, 29)]
LABELS = {
    "F21": ("env steps", "greedy minus stochastic"),
    "F22": ("later unlocks",),
    "F23": ("env steps", "Spearman"),
    "F24": ("ablation minus tuned",),
    "F25": ("tuning mean", "fresh-seed final mean"),
    "F26": ("achievements per episode",),
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


@pytest.fixture(scope="module")
def built(
    frames: dict[str, pd.DataFrame],
    trees: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> Callable[..., Built]:
    """``built(fid, theme, width)``: each figure drawn once for the tests that read it.

    Every angle figure is one 170 mm report image, so ``double`` is the default width.
    """
    return figure_cache(
        lambda fid, theme, width, out: draw(fid, frames, out, theme, width, trees),
        tmp_path_factory,
    )


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


def one_benchmark(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """The frames cut to Craftax-1M, as the results CLI hands them to a figure."""
    return select(frames, budget="1m", pool=True)


def draw(
    fid: str,
    frames: dict[str, pd.DataFrame],
    out_dir: Path,
    theme: str = "report",
    width: str = "double",
    trees: Path | None = None,
) -> Figure:
    """Call the figure of ``fid`` on Craftax-1M with test-sized bootstraps.

    Every angle figure is one 170 mm report image: ``double`` saves it (and registers
    its caption); a ``column`` draw saves nothing.
    """
    frames = one_benchmark(frames)
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


def captions_of(fid: str, *args: Any, **kwargs: Any) -> dict[str, Caption]:
    """The captions the figure of ``fid`` registers, by stem (`draw` arguments)."""
    with capture_captions() as got:
        draw(fid, *args, **kwargs)
    return {c.stem: c for c in got}


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


def test_every_figure_id_has_a_function_and_its_stems() -> None:
    assert list(ANGLE_FIGURES) == FIGS
    assert set(NAMES) == set(FIGS) == set(STEMS)
    assert all(f.__name__.startswith("plot_") for f in ANGLE_FIGURES.values())
    assert STEMS["F26"] == ["episode_spread", "achievements_vs_length"]
    assert all(STEMS[f] == [NAMES[f]] for f in FIGS if f != "F26")


@pytest.mark.parametrize("theme", ["report", "video"])
@pytest.mark.parametrize("fid", FIGS)
def test_each_figure_writes_its_pngs_with_its_labels(
    fid: str, theme: str, built: Callable[..., Built]
) -> None:
    fig, out, _ = built(fid, theme, "double")
    suffix = "_video" if theme == "video" else ""
    for stem in STEMS[fid]:  # the PNG header is `test_figure_style`'s
        assert (out / f"{stem}{suffix}.png").is_file()
    for want in LABELS[fid]:
        assert want in axis_labels(fig), (fid, want, axis_labels(fig))


@pytest.mark.parametrize("fid", FIGS)
def test_sizes_are_the_report_widths(fid: str, built: Callable[..., Built]) -> None:
    """The 1080p frame of the video is audited by `test_figure_style`."""
    fig, out, _ = built(fid, "report", "column")
    assert fig.get_size_inches()[0] * 25.4 == pytest.approx(80.0, abs=0.01)
    assert not list(out.glob("*.png")), "every angle figure is 170 mm: no column file"
    fig.canvas.draw()
    small = [
        t.get_text()
        for t in fig.findobj(Text)
        if t.get_visible() and t.get_text().strip() and t.get_fontsize() < 7
    ]
    assert not small, small
    wide, out, _ = built(fid, "report", "double")
    assert wide.get_size_inches()[0] * 25.4 == pytest.approx(170.0, abs=0.01)
    for stem in STEMS[fid]:  # the figure's one report image, no suffix
        png = (out / f"{stem}.png").read_bytes()
        assert int.from_bytes(png[16:20], "big") == pytest.approx(
            170 / 25.4 * 300, abs=1
        )
        assert not (out / f"{stem}_double.png").exists()


@pytest.mark.parametrize("fid", FIGS)
def test_a_figure_declares_a_size_class_and_has_no_title_or_text_of_its_own(
    fid: str, built: Callable[..., Built]
) -> None:
    fig = built(fid).figs
    assert fig.shape in set(Shape)  # type: ignore[attr-defined]
    assert not fig.texts
    assert getattr(fig, "_suptitle", None) is None
    assert all(not ax.get_title(loc="left") for ax in fig.axes)
    assert all(ax.get_legend() is None for ax in fig.axes)
    if fid != "F24":  # F24 keeps its two-entry marker legend (hollow dot, cross)
        assert not fig.legends


@pytest.mark.parametrize("fid", FIGS)
def test_every_stem_registers_a_caption_with_the_question_and_no_budget_words(
    fid: str, built: Callable[..., Built]
) -> None:
    got = {c.stem: c for c in built(fid, "report", "double").captions}
    assert set(got) == set(STEMS[fid])
    for caption in got.values():
        assert caption.title.strip()
        assert "?" in caption.description  # the question the figure answers first
        assert "1B" not in caption.description, caption.description
        assert "pending" not in caption.title.lower()


def test_figures_of_two_benchmarks_are_refused(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="one benchmark"):
        plot_greedy_gap(frames, tmp_path, n_boot=N)


def test_an_unknown_theme_or_width_is_refused(
    frames_1m: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="theme"):
        plot_greedy_gap(frames_1m, tmp_path, "poster", n_boot=N)
    with pytest.raises(ValueError, match="width"):
        plot_greedy_gap(frames_1m, tmp_path, width="triple", n_boot=N)


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
    frames_1m: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    fig = plot_greedy_gap(frames_1m, tmp_path, n_boot=N)
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
    assert {a.get_gid() for a in artists(fig, "gap")} == {f"gap/1m/{m}" for m in LEVEL}


# --- F22 gatekeepers ----------------------------------------------------------------


def naive_lift(eps: pd.DataFrame, method: str, run: str, ach: str) -> float:
    """The landmark lift by a plain loop: A held at the landmark, not A ever."""
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
        held = bool(e[f"ach/{ach}"]) and e[f"unlock_step/{ach}"] <= at
        groups[held].append(later)
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
    # only episode 0 holds a by step 20; 1 (a at 30) and 2 have not yet, 3 is dead
    assert (row["with_a"], row["without_a"]) == (1, 2)
    assert row["lift"] == pytest.approx(0.0 - 1.0)  # with the dead one: 0 - 2/3


def test_a_late_unlock_counts_as_not_yet_at_the_landmark() -> None:
    """Grouping by A ever would give -0.5 here; the landmark grouping gives +0.5."""
    eps = pd.DataFrame({
        "method": "sil",
        "run_id": "r",
        "length": [100, 100, 100],
        "ach/a": [True, True, False],
        "unlock_step/a": [10, 30, -1],
        "ach/b": [True, False, True],
        "unlock_step/b": [50, -1, 40],
    })
    row = gatekeepers(eps).set_index("achievement").loc["a"]
    assert row["lift"] == pytest.approx(1.0 - 0.5)


def test_an_achievement_without_effect_has_no_lift() -> None:
    """Null world: constant-hazard death, six achievements at independent times."""
    rng = np.random.default_rng(3)
    n, names = 4000, [f"x{i}" for i in range(6)]
    length = rng.geometric(1 / 300, n)
    frame = {"method": "sil", "run_id": np.repeat([f"r{k}" for k in range(8)], n // 8)}
    frame["length"] = length
    for name in names:
        step = rng.exponential(250, n).astype(int) + 1
        got = step <= length
        frame[f"ach/{name}"] = got
        frame[f"unlock_step/{name}"] = np.where(got, step, -1)
    table = gatekeepers(pd.DataFrame(frame))
    means = table.groupby("achievement")["lift"].mean()
    # sampling noise only: grouping by "A ever" gives about +1.0 in this world
    assert means.abs().max() < 0.2, means
    assert abs(means.mean()) < 0.05, means


def test_a_gatekeeper_dot_is_the_mean_lift_over_seeds(
    frames: dict[str, pd.DataFrame], frames_1m: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    fig = plot_gatekeepers(frames_1m, tmp_path, n_boot=N)
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


def test_gatekeeper_rows_no_method_can_fill_are_named_not_drawn(
    frames_1m: dict[str, pd.DataFrame], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(figures_angles, "GATE_MIN", 10_000)
    fig = plot_gatekeepers(frames_1m, tmp_path, n_boot=N)
    assert not artists(fig, "gate")
    assert "No method has 10000 alive episodes" in said(fig)


def test_the_gatekeeper_rows_fit_a_third_of_a_page_and_the_cut_ones_are_named(
    frames_1m: dict[str, pd.DataFrame], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    methods = len(LEVEL)
    limit = figures_angles._gate_limit(methods)
    assert limit == 6  # 9 dodged methods: a row is 10.85 mm, 66 mm of rows
    assert limit * (methods * 1.05 + 1.4) + 14 <= MAX_ROWS_MM
    full = captions_of("F22", frames_1m, tmp_path)["gatekeepers"].description
    monkeypatch.setattr(figures_angles, "_gate_limit", lambda _m: 2)
    got = captions_of("F22", frames_1m, tmp_path)["gatekeepers"].description
    assert "Not drawn, smaller mean lift" in got
    assert "Not drawn" not in full
    fig = plot_gatekeepers(frames_1m, tmp_path, n_boot=N)
    assert len({str(a.get_gid()).split("/")[3] for a in artists(fig, "gate")}) == 2


def test_a_single_gatekeeper_row_has_no_lone_tick_label(
    frames_1m: dict[str, pd.DataFrame], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(figures_angles, "_gate_limit", lambda _m: 1)
    fig = plot_gatekeepers(frames_1m, tmp_path, n_boot=N)
    assert not [t for t in fig.axes[0].get_yticklabels() if t.get_text()]
    got = captions_of("F22", frames_1m, tmp_path)["gatekeepers"].description
    assert "The single row is" in got


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
    frames_1m: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    fig = plot_seed_fate(frames_1m, tmp_path)
    line = by_gid(fig, f"fate/1m/{REVERSED}")
    np.testing.assert_allclose(line.get_ydata(), [-1.0, 1.0, 1.0])  # type: ignore[attr-defined]
    pooled = by_gid(fig, "fate/1m/pooled")
    first = (len(LEVEL) - 2) / len(LEVEL)  # one -1 among nine
    assert pooled.get_ydata()[0] == pytest.approx(first)  # type: ignore[attr-defined]
    t = sps.t.ppf(0.975, 8)  # 10 seeds: the 5% two-sided critical rho
    null = by_gid(fig, "null/1m")
    top = null.get_y() + null.get_height()  # type: ignore[attr-defined]
    assert top == pytest.approx(t / np.sqrt(8 + t**2))
    note = captions_of("F23", frames_1m, tmp_path)["seed_fate"].description
    assert f"from -{top:.2f} to {top:.2f}" in note  # the band text moved to the caption


# --- F24 ablations ------------------------------------------------------------------


def test_ablation_effects_are_the_ablation_table_differences(
    frames_1m: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    frames = frames_1m
    table = compare.ablation_table(frames, n_boot=N)
    fig = plot_ablation_effects(frames, tmp_path, n_boot=N, table=table)
    for r in table.itertuples():
        key = f"{r.benchmark}/{r.method}/{r.config_ref}"
        dot = by_gid(fig, f"effect/{key}")
        assert float(dot.get_xdata()[0]) == pytest.approx(r.diff)  # type: ignore[attr-defined]
        seeds = by_gid(fig, f"seeds/{key}")
        runs = frames["runs"][frames["runs"]["budget"] == r.benchmark]
        tuned = runs[(runs["method"] == r.method) & (runs["config_ref"] == "tuned")]
        mine = runs[(runs["method"] == r.method) & (runs["config_ref"] == r.config_ref)]
        want = mine["final_score"] - tuned["final_score"].mean()
        np.testing.assert_allclose(
            sorted(seeds.get_offsets()[:, 0]),
            sorted(want),  # type: ignore[attr-defined]
        )
    assert "PoI" not in said(fig)  # the stats line is in the caption now
    got = captions_of("F24", frames, tmp_path)["ablation_effects"].description
    for r in table[table["poi_tuned"].notna()].itertuples():
        assert f"{r.method} ({r.config_ref}) PoI {r.poi_tuned:.2f}" in got


def test_ablation_crosses_use_only_the_fresh_tuned_seeds(
    frames_1m: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    frames = frames_1m
    fig = plot_ablation_effects(frames, tmp_path, n_boot=N)
    runs = frames["runs"][frames["runs"]["budget"] == "1m"]
    tuned = runs[(runs["method"] == "sil") & (runs["config_ref"] == "tuned")]
    fresh = tuned[~tuned["phases"].map(reused_from_tuning)]["final_score"]
    assert 0 < len(fresh) < len(tuned)
    ablation = runs[runs["config_ref"] == "single-buffer"]["final_score"]
    cross = by_gid(fig, "fresh/1m/sil/single-buffer")
    want = ablation.mean() - fresh.mean()
    assert float(cross.get_xdata()[0]) == pytest.approx(want)  # type: ignore[attr-defined]
    assert not artists(fig, "fresh/1m/dyna-q")  # no reused tuned seed, no cross


# --- F25 tuning optimism ------------------------------------------------------------


def test_tuning_optimism_points_and_their_fresh_intervals(
    frames_1m: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    frames = frames_1m
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
    got = captions_of("F25", frames, tmp_path)["tuning_optimism"].description
    for r in table[table["optimism"].notna()].itertuples():
        assert f"{r.method} {r.optimism:+.2f}" in got  # the optimism is in the caption
    assert "selected on all its final seeds: ppo" in got
    assert "ppo" not in said(fig)  # no method names on the image: legend_methods


def test_tuning_optimism_has_no_legend_and_no_text_beyond_ticks_and_labels(
    frames_1m: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    fig = plot_tuning_optimism(
        frames_1m, tmp_path, n_boot=N, table=optimism_table(frames_1m)
    )
    assert not fig.legends
    assert fig.axes[0].get_legend() is None
    assert not fig.axes[0].texts


# --- F26 episode spread -------------------------------------------------------------


def test_episode_spread_shares_and_the_mean_triangles(
    frames_1m: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    frames = frames_1m
    fig = plot_episode_spread(frames, tmp_path)
    mesh = by_gid(fig, "heat/1m/achievements")
    assert isinstance(mesh, QuadMesh)
    grid = np.ma.filled(mesh.get_array(), np.nan).reshape(len(LEVEL), -1) / 100
    rows = [t.get_text() for t in mesh.axes.get_yticklabels()]
    eps = tuned_episodes(frames)
    for i, method in enumerate(rows):
        counts = eps.loc[eps["method"] == method, "n_achievements"]
        share = counts.value_counts(normalize=True)
        for k, s in share.items():
            assert grid[i, int(k)] == pytest.approx(s)
        assert np.nansum(grid[i]) == pytest.approx(1.0)
    # mean triangles: the method's mean achievements, at the centre of its count cell
    for method in rows:
        mean = eps.loc[eps["method"] == method, "n_achievements"].mean()
        tri = by_gid(fig, f"mean/1m/{method}")
        assert float(tri.get_xdata()[0]) == pytest.approx(mean + 0.5)  # type: ignore[attr-defined]


def tuned_episodes(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    eps = final_checkpoint(frames["eval_episodes"])
    eps = eps[(eps["mode"] == "stochastic") & (eps["budget"] == "1m")]
    return eps[eps["config_ref"] == "tuned"]


def test_episode_spread_writes_two_files_one_panel_each(
    frames_1m: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    got = captions_of("F26", frames_1m, tmp_path)
    assert set(got) == {"episode_spread", "achievements_vs_length"}
    for stem in got:
        assert (tmp_path / f"{stem}.png").is_file()


def test_the_length_lines_are_the_bin_means_of_the_equal_count_bins(
    frames_1m: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    fig = plot_achievements_vs_length(frames_1m, tmp_path)
    eps = tuned_episodes(frames_1m)
    edges = np.unique(np.quantile(eps["length"], np.linspace(0, 1, 9)))
    for method in ("sil", "dyna-q"):
        mine = eps[eps["method"] == method]
        cut = pd.cut(mine["length"], edges, include_lowest=True)
        want = mine.groupby(cut, observed=True).agg(
            x=("length", "median"), y=("n_achievements", "mean"), n=("length", "size")
        )
        want = want[want["n"] >= 10]
        line = by_gid(fig, f"length/1m/{method}")
        np.testing.assert_allclose(line.get_xdata(), want["x"])  # type: ignore[attr-defined]
        np.testing.assert_allclose(line.get_ydata(), want["y"])  # type: ignore[attr-defined]


# --- F27 and F28 distillation -------------------------------------------------------


def test_distil_score_points_and_the_spearman(
    frames_1m: dict[str, pd.DataFrame], trees: Path, tmp_path: Path
) -> None:
    frames = frames_1m
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
    within = np.mean([
        sps.spearmanr(
            [fidelity_of(m, k) for k in range(10)],
            tuned[tuned["method"] == m].sort_values("seed")["final_score"],
        ).statistic
        for m in DISTILLED
    ])
    text = captions_of("F27", frames, tmp_path, trees=trees)["distil_score"].description
    assert f"Pooled Spearman rho {rho:.2f}" in text
    assert f"within methods {within:.2f}" in text
    for want in ("Distillation pending", "pqn-rnn", "combo"):
        assert want in text
    assert "rho" not in said(fig)  # the numbers moved to the caption


def test_distil_features_heat_map_is_the_mean_importance(
    frames_1m: dict[str, pd.DataFrame], trees: Path, tmp_path: Path
) -> None:
    fig = plot_distil_features(frames_1m, tmp_path, interpretation=trees)
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
    frames_1m: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    for fn in (plot_distil_score, plot_distil_features):
        fig = fn(frames_1m, tmp_path, interpretation=tmp_path / "none")
        assert "pending" in said(fig)
