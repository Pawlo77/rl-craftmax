"""Tests of the in-game figures F6-F17 on the game fixture (10 seeds at 1M, 3 at 1B).

Like `test_figures.py`: every figure writes a PNG (and no PDF), is sized for one
Interspeech column or a 1080p frame, takes its colours from `METHOD_COLORS`, and the
drawn artists (found by their ``gid`` ``<kind>/<budget>/<method>[/<category>]``) are
compared with the tables they come from. The fixture's values are deterministic
functions of (method, seed, eval slot), so a misplaced method or row fails.
"""

import functools
import itertools
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from _compare_fixture import LEVEL
from _game_fixture import FLOOR_COLUMNS, game_frames
from matplotlib.artist import Artist
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from matplotlib.text import Text
from scipy import stats as sps
from test_figures import Bundle, with_gid

from wut_murl_craftax.analysis import compare, figures_game
from wut_murl_craftax.analysis import figures_game as fg
from wut_murl_craftax.analysis.figures_game import (
    GAME_FIGURES,
    GAME_NAMES,
    funnel_shares,
    kendall_permutation,
    order_agreement,
    order_graph,
    radar_table,
    split_table,
    transitive_reduction,
)
from wut_murl_craftax.analysis.hard import stall_stats
from wut_murl_craftax.evaluation.stats import stratified_bootstrap_ci
from wut_murl_craftax.visualisation.charts import (
    METHOD_COLORS,
    METHOD_COLORS_LIGHT,
    capture_captions,
)


def _bundled(fn: Callable[..., dict[str, Figure]]) -> Callable[..., Bundle]:
    """``fn`` returning one `Bundle` of its files, searched as if they were one."""

    @functools.wraps(fn)
    def call(*args: object, **kwargs: object) -> Bundle:
        with capture_captions() as caps:
            figs = fn(*args, **kwargs)
        return Bundle(figs, {c.stem: c.description for c in caps})

    return call


plot_actions = _bundled(fg.plot_actions)
plot_advanced = _bundled(fg.plot_advanced)
plot_combat = _bundled(fg.plot_combat)
plot_coverage = _bundled(fg.plot_coverage)
plot_deaths = _bundled(fg.plot_deaths)
plot_funnel = _bundled(fg.plot_funnel)
plot_funnel_training = _bundled(fg.plot_funnel_training)
plot_order_agreement = _bundled(fg.plot_order_agreement)
plot_order_graph = _bundled(fg.plot_order_graph)
plot_race = _bundled(fg.plot_race)
plot_radar = _bundled(fg.plot_radar)
plot_resources = _bundled(fg.plot_resources)
plot_splits = _bundled(fg.plot_splits)
plot_stalls = _bundled(fg.plot_stalls)
plot_survival = _bundled(fg.plot_survival)
race = _bundled(fg.race)

N = 100
"""Bootstrap resamples and permutations in tests."""
THEMES = ("report", "video")
PNG = b"\x89PNG\r\n\x1a\n"
NOT_LOGGED = ("ppo", "ppo-rnn")
"""The reused runs without the online families (R78)."""
FIGS = ["F6", "F7", "F7G", "F8", "F8T", "F9", "F9S"] + [f"F{i}" for i in range(10, 18)]
"""Every figure id of `GAME_FIGURES`."""
METHODS = (
    "ppo", "ppo-rnn", "pqn-rnn", "count-bonus-ppo-rnn", "count-bonus-pqn-rnn", "sil",
    "combo", "dyna-q", "smdp-skills",
)  # fmt: skip
STEMS = {
    "F6": [f"race_{c}_{b}" for c in ("steps", "wall") for b in ("main", "skills")],
    "F7": ["order_agreement"],
    "F7G": ["order_sil"],
    "F8": ["funnel_tech", "funnel_floor"],
    "F8T": ["funnel_training_depth", "funnel_training_floor"],
    "F9": ["deaths"],
    "F9S": ["stalls_ccdf", "stalls_escape"],
    "F10": ["splits"],
    "F11": ["coverage", "new_tiles"],
    "F12": ["survival_low", "survival_zero", "survival_level"],
    "F13": ["combat_kills", "combat_damage", "combat_fights"],
    "F14": ["actions_share", "actions_entropy", "actions_wasted"],
    "F15": ["resources_use", "resources_time"],
    "F16": ["advanced_mean", "advanced_final"],
    "F17": [f"radar_{m}" for m in METHODS],
}
"""The files each figure id writes on the fixture at 1M (one panel per file)."""
DOUBLE = {"F7G"}
"""Figure ids with a 170 mm variant: the order graphs print node names, so they are
drawn at the double width and as video only."""
LABELS = {
    "F6": ("median first unlock",),
    "F7": (),
    "F7G": ("mean normalised rank",),
    "F8": ("share of episodes",),
    "F8T": ("env steps", "mean tech depth"),
    "F9": ("share of deaths",),
    "F9S": ("stall length", "escape rate"),
    "F10": ("steps to the first unlock",),
    "F11": ("share of the floor seen", "new tiles per 100 steps"),
    "F12": ("share of steps", "mean stat value"),
    "F13": ("damage taken", "fights per episode"),
    "F14": ("entropy", "share of steps wasted"),
    "F15": ("gathered per episode", "steps (log)"),
    "F16": ("env steps", "% of episodes"),
    "F17": (),
}
"""Substrings the axis labels of a figure set must contain."""


def axis_labels(fig: Bundle | Figure) -> str:
    return " | ".join(f"{ax.get_xlabel()} | {ax.get_ylabel()}" for ax in fig.axes)


@pytest.fixture(scope="module")
def frames() -> dict[str, pd.DataFrame]:
    """Craftax-1M (10 seeds): a figure shows one benchmark at a time."""
    return game_frames(("1m",))


@pytest.fixture(scope="module")
def frames_1m(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return frames


@pytest.fixture(scope="module")
def frames_1b() -> dict[str, pd.DataFrame]:
    """Craftax-1B (3 seeds)."""
    return game_frames(("1b",))


def draw(
    fid: str,
    frames: dict[str, pd.DataFrame],
    out_dir: Path,
    theme: str = "report",
    width: str = "column",
) -> Bundle:
    """Call the figure of ``fid`` with test-sized bootstraps."""
    k = {"width": width}
    b = {"n_boot": N, **k}
    calls: dict[str, Callable[[], Bundle]] = {
        "F6": lambda: plot_race(frames, out_dir, theme, **k),
        "F7": lambda: plot_order_agreement(frames, out_dir, theme, n_perm=N, **k),
        "F7G": lambda: plot_order_graph(frames, out_dir, theme, method="sil", **k),
        "F8": lambda: plot_funnel(frames, out_dir, theme, **b),
        "F8T": lambda: plot_funnel_training(frames, out_dir, theme, **b),
        "F9": lambda: plot_deaths(frames, out_dir, theme, **b),
        "F9S": lambda: plot_stalls(frames, out_dir, theme, **b),
        "F10": lambda: plot_splits(frames, out_dir, theme, **b),
        "F11": lambda: plot_coverage(frames, out_dir, theme, **b),
        "F12": lambda: plot_survival(frames, out_dir, theme, **b),
        "F13": lambda: plot_combat(frames, out_dir, theme, **b),
        "F14": lambda: plot_actions(frames, out_dir, theme, **b),
        "F15": lambda: plot_resources(frames, out_dir, theme, **b),
        "F16": lambda: plot_advanced(frames, out_dir, theme, **b),
        "F17": lambda: plot_radar(frames, out_dir, theme, **k),
    }
    return calls[fid]()


def artists(fig: Bundle | Figure, kind: str) -> list[Artist]:
    """Artists whose gid starts with ``kind + '/'``, in drawing order."""
    return fig.findobj(lambda a: (a.get_gid() or "").startswith(f"{kind}/"))


def gids(fig: Bundle | Figure, kind: str) -> set[str]:
    return {a.get_gid() for a in artists(fig, kind)}


def texts(fig: Bundle | Figure) -> list[str]:
    for f in fig.figs.values() if isinstance(fig, Bundle) else [fig]:
        f.canvas.draw()
    return [t.get_text() for t in fig.findobj(Text) if t.get_visible() and t.get_text()]


def said(fig: Bundle) -> str:
    """Every text on the images and in the captions of ``fig`` on one line."""
    return " ".join(" ".join([*texts(fig), *fig.captions.values()]).split())


def xdata(artist: Artist) -> float:
    return float(np.atleast_1d(artist.get_xdata())[0])  # type: ignore[attr-defined]


def ydata(artist: Artist) -> float:
    return float(np.atleast_1d(artist.get_ydata())[0])  # type: ignore[attr-defined]


# --- every figure -------------------------------------------------------------------


@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize("fid", FIGS)
def test_each_figure_writes_its_files_as_png_with_captions(
    fid: str, theme: str, frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    width = "double" if fid in DOUBLE else "column"
    figs = draw(fid, frames, tmp_path, theme, width)
    assert sorted(figs.figs) == sorted(STEMS[fid]) or fid == "F6"
    if fid == "F6":  # the wall-clock and the env-steps race, a file per block
        assert sorted(figs.figs) == sorted(STEMS[fid])
    suffix = "_video" if theme == "video" else ""
    for name in STEMS[fid]:
        assert not (tmp_path / f"{name}{suffix}.pdf").exists()
        assert (tmp_path / f"{name}{suffix}.png").read_bytes().startswith(PNG)
        assert figs.captions[name].strip(), name
    for want in LABELS[fid]:
        assert want in axis_labels(figs), (fid, want, axis_labels(figs))


def test_every_figure_id_has_a_function_and_a_stem() -> None:
    assert set(GAME_FIGURES) == set(FIGS)
    assert set(GAME_NAMES) == set(FIGS) | {"F6W"}
    for fid in FIGS:
        assert GAME_FIGURES[fid].__name__.startswith("plot_")


@pytest.mark.parametrize("fid", [f for f in FIGS if f not in DOUBLE])
def test_report_figures_are_one_column_wide_with_7pt_fonts(
    fid: str, frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    for fig in draw(fid, frames, tmp_path).figs.values():
        assert fig.get_size_inches()[0] * 25.4 == pytest.approx(80.0, abs=0.01)
        fig.canvas.draw()
        small = {
            t.get_text(): t.get_fontsize()
            for t in fig.findobj(Text)
            if t.get_visible() and t.get_text().strip() and t.get_fontsize() < 7
        }
        assert not small, small


@pytest.mark.parametrize("fid", FIGS)
def test_video_figures_fill_a_1080p_frame(
    fid: str, frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    for fig in draw(fid, frames, tmp_path, "video").figs.values():
        width, height = fig.get_size_inches() * fig.dpi
        assert (round(width), round(height)) == (1920, 1080)


@pytest.mark.parametrize("fid", FIGS)
def test_a_figure_has_one_report_image_at_its_own_width(
    fid: str, frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    figs = draw(fid, frames, tmp_path, width="double")
    if fid not in DOUBLE:
        assert not figs.figs
        assert not list(tmp_path.iterdir())
        return
    for fig in figs.figs.values():
        assert fig.get_size_inches()[0] * 25.4 == pytest.approx(170.0, abs=0.01)
    for name in STEMS[fid]:  # the node names need 170 mm: the one image has no suffix
        assert (tmp_path / f"{name}.png").read_bytes().startswith(PNG)
        assert not (tmp_path / f"{name}_double.png").exists()
    assert not draw(
        fid, frames, tmp_path / "col", width="column"
    ).figs  # no low-res copy
    assert not list((tmp_path / "col").glob("*.png"))
    texts(figs)
    for fig in figs.figs.values():
        small = [
            t.get_text()
            for t in fig.findobj(Text)
            if t.get_visible() and t.get_text().strip() and t.get_fontsize() < 7
        ]
        assert not small, small


def test_the_width_option_is_refused_when_unknown(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="width"):
        plot_funnel(frames, tmp_path, width="triple", n_boot=N)


def test_an_unknown_theme_is_refused(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="theme"):
        plot_funnel(frames, tmp_path, "poster", n_boot=N)


def test_a_figure_shows_one_benchmark_at_a_time(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="one benchmark"):
        plot_funnel(game_frames(), tmp_path, n_boot=N)


@pytest.mark.parametrize("fid", FIGS)
def test_1b_figures_have_no_pending_or_indicative_text(
    fid: str, frames_1b: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    """A benchmark's set is drawn from its own runs and mentions no other budget."""
    figs = draw(fid, frames_1b, tmp_path, width="double" if fid in DOUBLE else "column")
    assert figs.figs
    labels = [
        t
        for ax in figs.axes
        for t in (ax.get_xlabel(), ax.get_ylabel(), ax.get_title())
    ]
    for t in [*labels, *figs.captions.values()]:  # a step tick "1B" is data
        assert not any(w in t for w in ("pending", "indicative", "1B", "Craftax")), t


@pytest.mark.parametrize("fid", FIGS)
def test_no_run_data_gives_a_labelled_placeholder(
    fid: str, frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    empty = {k: v.iloc[:0] for k, v in frames.items()}
    figs = draw(fid, empty, tmp_path, width="double" if fid in DOUBLE else "column")
    assert any("no data" in t for t in texts(figs))
    assert list(tmp_path.glob("*.png"))


# --- F6 -------------------------------------------------------------------------


def _race_rows(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    table = compare.race_table(frames)
    return table[table["eligible"].astype(bool)]


def test_f6_dots_are_the_race_table_medians(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    fig = race(frames, tmp_path)
    rows = _race_rows(frames)
    drawn = {a.get_gid(): xdata(a) for a in artists(fig, "race")}
    want = {
        f"race/{r.benchmark}/{r.method}/{r.achievement}": r.median_steps
        for r in rows.itertuples()
    }
    assert drawn == pytest.approx(want)
    assert len(drawn) == len(
        rows[["benchmark", "method", "achievement"]].drop_duplicates()
    )


def test_f6_the_winner_is_ringed_where_the_table_says(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    fig = race(frames, tmp_path)
    table = compare.race_table(frames)
    won = table[table["winner_steps"].astype(bool)]
    want = {
        f"winner/{r.benchmark}/{r.method}/{r.achievement}" for r in won.itertuples()
    }
    assert gids(fig, "winner") == want
    # the fixture's winners: the shortest name unlocks first in each block
    assert {g.split("/")[2] for g in want} == {"sil", "dyna-q"}


def test_f6_wall_clock_is_a_second_file_with_its_winners(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    fig = race(frames, tmp_path, clock="wall")
    assert (tmp_path / "race_wall_main.png").exists()
    assert (tmp_path / "race_wall_skills.png").exists()
    won = compare.race_table(frames)
    assert not won["winner_wall"].any()  # R97: every run is a laptop run, no ring
    assert gids(fig, "winner") == set()
    assert "two concurrent workers" in said(fig)


def test_f6_says_not_logged_for_the_reused_runs_and_how_many_seeds_logged(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    fig = race(frames, tmp_path)
    shown = said(fig)
    assert "ppo, ppo-rnn: first unlocks not logged" in " ".join(shown.split())
    assert "smdp-skills" not in shown.split("not logged")[0].split("Craftax-1M")[-1]
    assert not any(f"/{m}/" in g for g in gids(fig, "race") for m in NOT_LOGGED)
    # a method that logged 7 of its 10 seeds says so (race k of n)
    partial = {**frames}
    runs = frames["runs"]
    drop = runs[(runs["method"] == "sil") & (runs["budget"] == "1m")]["run_id"].iloc[:3]
    partial["curves"] = frames["curves"][
        ~(
            frames["curves"]["run_id"].isin(drop)
            & frames["curves"]["metric"].str.startswith("first_unlock")
        )
    ]
    assert "sil 7 of 10" in said(race(partial, tmp_path))


def test_f6_upto_draws_only_first_unlocks_before_that_step(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    firsts = compare.first_unlocks(frames["curves"], compare.tuned(frames["runs"]))
    steps = firsts["steps"]
    cut = int(np.median(steps))
    fig = race(frames, None, "video", upto=cut)
    full = race(frames, None, "video")
    shown, everything = gids(fig, "race"), gids(full, "race")
    assert shown < everything
    assert xdata(artists(fig, "race")[0]) < cut
    assert all(xdata(a) < cut for a in artists(fig, "race"))
    assert not artists(race(frames, None, "video", upto=int(steps.min())), "race")
    assert (
        gids(race(frames, None, "video", upto=int(steps.max()) + 1), "race")
        == everything
    )
    # the axes of the frames do not move, so a GIF does not rescale
    ax, ref = fig.axes[0], full.axes[0]
    assert ax.get_xlim() == ref.get_xlim()


def test_f6_without_an_out_dir_writes_no_file(
    frames: dict[str, pd.DataFrame], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    race(frames, None)
    assert not list(tmp_path.iterdir())


def test_f6_refuses_an_unknown_clock(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="clock"):
        race(frames, tmp_path, clock="gpu")


# --- F7 -------------------------------------------------------------------------


def test_transitive_reduction_removes_a_to_c_when_a_b_and_b_c_exist() -> None:
    edges = {("a", "b"), ("b", "c"), ("a", "c"), ("a", "d")}
    assert transitive_reduction(edges) == {("a", "b"), ("b", "c"), ("a", "d")}
    longer = {("a", "b"), ("b", "c"), ("c", "d"), ("a", "d"), ("b", "d")}
    assert transitive_reduction(longer) == {("a", "b"), ("b", "c"), ("c", "d")}
    assert transitive_reduction(set()) == set()


def _method_eps(
    frames: dict[str, pd.DataFrame], method: str, budget: str
) -> pd.DataFrame:
    (ctx,) = [c for c in figures_game._contexts(frames) if c.budget == budget]
    return ctx.eps[ctx.eps["method"] == method]


def test_the_order_graph_of_a_chain_keeps_only_the_consecutive_edges(
    frames: dict[str, pd.DataFrame],
) -> None:
    nodes, edges = order_graph(_method_eps(frames, "sil", "1m"))
    names = list(nodes["achievement"])
    assert names == sorted(names, key=list(figures_game.ACHIEVEMENTS).index)
    pairs = set(zip(edges["source"], edges["target"], strict=True))
    # place_table (50% of episodes), defeat_zombie (100%), make_arrow (25%) and
    # enter_dungeon (75%) are nodes; make_arrow shares only 10 episodes with
    # defeat_zombie, so it has no edge; place_table -> enter_dungeon is implied
    assert names == ["place_table", "defeat_zombie", "make_arrow", "enter_dungeon"]
    assert pairs == {
        ("place_table", "defeat_zombie"),
        ("defeat_zombie", "enter_dungeon"),
    }
    assert (edges["p"] == 1.0).all()
    assert (edges["pairs"] >= 20).all()
    assert nodes["rho"].is_monotonic_increasing
    assert nodes["rho"].iloc[0] == pytest.approx(0.0)
    assert nodes["rho"].iloc[-1] == pytest.approx(1.0)


def test_an_edge_needs_twenty_episodes_with_both_achievements(
    frames_1b: dict[str, pd.DataFrame],
) -> None:
    nodes, edges = order_graph(_method_eps(frames_1b, "sil", "1b"))  # 12 episodes
    assert len(nodes) > 1
    assert edges.empty


def test_a_node_needs_five_percent_of_the_episodes(
    frames: dict[str, pd.DataFrame],
) -> None:
    eps = _method_eps(frames, "sil", "1m").copy()
    rare = "unlock_step/enter_sewers"
    eps[rare] = -1
    eps.iloc[:1, eps.columns.get_loc(rare)] = 5  # 1 of 40 episodes: 2.5%
    nodes, _ = order_graph(eps)
    assert "enter_sewers" not in set(nodes["achievement"])
    eps.iloc[:3, eps.columns.get_loc(rare)] = 5  # 7.5%
    assert "enter_sewers" in set(order_graph(eps)[0]["achievement"])


def test_kendall_permutation_matches_scipy_and_the_extremes() -> None:
    rng = np.random.default_rng(3)
    x, y = rng.random(30), rng.random(30)
    tau, p = kendall_permutation(x, y, 500, 1)
    assert tau == pytest.approx(sps.kendalltau(x, y).statistic)
    assert p == pytest.approx(sps.kendalltau(x, y).pvalue, abs=0.15)
    up, p_up = kendall_permutation(x, x * 2 + 1, 500, 1)
    down, _ = kendall_permutation(x, -x, 500, 1)
    assert (up, down) == (pytest.approx(1.0), pytest.approx(-1.0))
    assert p_up == pytest.approx(1 / 501)
    assert all(np.isnan(v) for v in kendall_permutation(x[:2], y[:2]))
    assert all(np.isnan(v) for v in kendall_permutation(x, np.ones(30)))


def test_kendall_ties_use_tau_b() -> None:
    x = np.array([1.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([1.0, 2.0, 2.0, 3.0, 5.0])
    tau, _ = kendall_permutation(x, y, 50, 0)
    assert tau == pytest.approx(sps.kendalltau(x, y, variant="b").statistic)


def test_order_agreement_is_one_row_per_ordered_method_pair(
    frames: dict[str, pd.DataFrame],
) -> None:
    table = order_agreement(frames, n_perm=N)
    one_m = table[table["benchmark"] == "1m"]
    assert len(one_m) == 9 * 8
    assert not (one_m["method"] == one_m["other"]).any()
    row = one_m[(one_m["method"] == "sil") & (one_m["other"] == "ppo")].iloc[0]
    mirror = one_m[(one_m["method"] == "ppo") & (one_m["other"] == "sil")].iloc[0]
    assert row["tau"] == pytest.approx(mirror["tau"])
    assert row["jaccard"] == mirror["jaccard"]
    assert 0.0 <= row["jaccard"] <= 1.0


def test_f7_graph_figure_draws_the_reduced_edges_and_every_node(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    fig = plot_order_graph(frames, tmp_path, method="sil", width="double")
    nodes, edges = order_graph(_method_eps(frames, "sil", "1m"))
    assert gids(fig, "edge") == {
        f"edge/1m/sil/{s}>{t}"
        for s, t in zip(edges["source"], edges["target"], strict=True)
    }
    assert gids(fig, "node") == {f"node/1m/sil/{a}" for a in nodes["achievement"]}
    assert (tmp_path / "order_sil.png").exists()
    assert not (tmp_path / "order_sil_double.png").exists()
    assert "Arrow: a is unlocked before b" in said(fig)


def test_f7_agreement_matrix_is_the_table(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    fig = plot_order_agreement(frames, tmp_path, n_perm=N)
    assert len(gids(fig, "tau")) == 1
    shown = [t for t in texts(fig) if t.replace("-", "").replace(".", "").isdigit()]
    assert len(shown) == len(order_agreement(frames, n_perm=N).dropna(subset=["tau"]))


# --- F8 -------------------------------------------------------------------------


def test_funnel_shares_never_increase_with_depth(
    frames: dict[str, pd.DataFrame],
) -> None:
    (ctx,) = [c for c in figures_game._contexts(frames) if c.budget == "1m"]
    table = funnel_shares(ctx.eps)
    for _, g in table.groupby(["method", "kind"]):
        assert g.sort_values("step")["share"].is_monotonic_decreasing
    floor0 = table[(table["kind"] == "floor") & (table["step"] == 0)]
    assert (floor0["share"] == 1.0).all()
    assert table["share"].nunique() > 5


def test_f8_the_tech_profile_and_floor_dots_are_the_funnel_shares(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    fig = plot_funnel(frames, tmp_path, n_boot=N)
    (ctx,) = [c for c in figures_game._contexts(frames) if c.budget == "1m"]
    table = funnel_shares(ctx.eps)
    tech = table[table["kind"] == "tech"]
    for m in ctx.names:  # a line per method: share at each drawn step, never rising
        line = with_gid(fig, "funnel-tech", "1m", m)
        y = np.asarray(line.get_ydata(), float)  # type: ignore[attr-defined]
        rows = tech[tech["method"] == m].sort_values("step")
        assert y == pytest.approx(rows["share"].to_numpy()[: len(y)])
        assert (rows["share"].to_numpy()[len(y) :] == 0).all()  # all-zero tail: a note
        assert all(a >= b - 1e-9 for a, b in itertools.pairwise(y))
    floor = table[table["kind"] == "floor"]
    for (
        m
    ) in ctx.names:  # more steps than a dot panel holds: a profile like the tech one
        y = np.asarray(with_gid(fig, "funnel-floor", "1m", m).get_ydata(), float)  # type: ignore[attr-defined]
        rows = floor[floor["method"] == m].sort_values("step")
        assert y == pytest.approx(rows["share"].to_numpy()[: len(y)])


def test_f8_training_is_the_sum_of_the_logged_shares(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    fig = plot_funnel_training(frames, tmp_path, n_boot=N)
    line = next(a for a in artists(fig, "depth") if a.get_gid() == "depth/1m/sil")
    # the fixture's depth_k = 100 * ramp * 0.9**k, so the mean depth ends at sum 0.9**k
    want = sum(0.9**k for k in range(1, 11))
    assert float(line.get_ydata()[-1]) == pytest.approx(want, rel=1e-6)  # type: ignore[attr-defined]
    shown = said(fig)
    assert "ppo, ppo-rnn not logged" in shown
    assert not any("/ppo/" in g or "/ppo-rnn/" in g for g in gids(fig, "depth"))


# --- F9 -------------------------------------------------------------------------


def test_f9_death_bars_stack_to_one_with_the_methods_own_cause(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    fig = plot_deaths(frames, tmp_path, n_boot=N)
    bars = artists(fig, "death")
    for method in ("sil", "smdp-skills"):
        mine = [
            b
            for b in bars
            if b.get_gid().split("/")[2] == method
            and b.get_gid().startswith("death/1m")
        ]
        assert sum(b.get_width() for b in mine) == pytest.approx(1.0)  # type: ignore[attr-defined]
    zombie = {b.get_gid(): b.get_width() for b in bars if "smdp-skills" in b.get_gid()}  # type: ignore[attr-defined]
    assert zombie["death/1m/smdp-skills/zombie"] == pytest.approx(1.0)
    starve = {b.get_gid(): b.get_width() for b in bars if "/sil/" in b.get_gid()}  # type: ignore[attr-defined]
    assert starve["death/1m/sil/starvation"] == pytest.approx(1.0)
    assert len(gids(fig, "swatch")) == 9


def test_f9_stall_ccdf_starts_at_one_and_never_rises(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    fig = plot_stalls(frames, tmp_path, n_boot=N)
    lines = [a for a in artists(fig, "ccdf") if a.get_gid().startswith("ccdf/1m")]
    assert {a.get_gid().split("/")[2] for a in lines} == set(
        figures_game._contexts(frames)[0].names
    )
    for line in lines:
        y = np.asarray(line.get_ydata(), float)  # type: ignore[attr-defined]
        assert y[0] == pytest.approx(1.0)
        assert (np.diff(y) <= 1e-12).all()


def test_f9_escape_rate_is_the_stall_stats_value(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    fig = plot_stalls(frames, tmp_path, n_boot=N)
    (ctx,) = [c for c in figures_game._contexts(frames, N) if c.budget == "1m"]
    stats = stall_stats(ctx.eps, "stochastic", n=N).set_index("method")
    for bar in artists(fig, "escape"):
        _, budget, method = bar.get_gid().split("/")
        if budget == "1m":
            assert bar.get_width() == pytest.approx(stats.loc[method, "escape_rate"])  # type: ignore[attr-defined]


# --- F10 ------------------------------------------------------------------------


def test_split_table_counts_the_episodes_that_reached_each_milestone() -> None:
    eps = pd.DataFrame({
        "method": ["a"] * 4 + ["b"] * 4,
        "split/collect_wood": [10, 20, 30, -1, -1, -1, -1, -1],
    })
    table = split_table(eps).set_index(["method", "milestone"])
    a = table.loc[("a", "collect_wood")]
    assert (a["n"], a["share"], a["median"], a["q1"], a["q3"]) == (3, 0.75, 20, 15, 25)
    b = table.loc[("b", "collect_wood")]
    assert b["n"] == 0
    assert np.isnan(b["median"])
    assert b["share"] == 0.0


def test_f10_boxes_are_the_quartiles_and_the_share_is_text(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    fig = plot_splits(frames, tmp_path, n_boot=N)
    (ctx,) = [c for c in figures_game._contexts(frames) if c.budget == "1m"]
    table = split_table(ctx.eps)
    shown = figures_game._split_rows(table)
    assert len(shown) == figures_game.SPLIT_ROWS  # the fixture reaches more
    reached = table[(table["n"] > 0) & table["milestone"].isin(shown)]
    assert gids(fig, "split") == {
        f"split/1m/{r.method}/{r.milestone}" for r in reached.itertuples()
    }
    assert "Not drawn (only the first" in said(fig)
    box = next(
        a for a in artists(fig, "split") if a.get_gid() == "split/1m/sil/collect_wood"
    )
    row = reached[
        (reached["method"] == "sil") & (reached["milestone"] == "collect_wood")
    ].iloc[0]
    verts = box.get_path().vertices  # type: ignore[attr-defined]
    assert verts[:, 0].min() == pytest.approx(row["q1"])
    assert verts[:, 0].max() == pytest.approx(row["q3"])
    note = next(
        a for a in artists(fig, "share") if a.get_gid() == "share/1m/all/collect_wood"
    )
    mine = table[table["milestone"] == "collect_wood"]["share"]
    assert note.get_text() == f"{mine.min():.0%}-{mine.max():.0%}"  # type: ignore[attr-defined]
    assert mine.min() < mine.max()  # the fixture's shares differ by method


# --- F11 - F16: dots with intervals -------------------------------------------------


def _seed_median(
    frames: dict[str, pd.DataFrame], method: str, column: str, budget: str = "1m"
) -> np.ndarray:
    eps = _method_eps(frames, method, budget)
    return eps.groupby("run_id")[column].median().to_numpy(float)


def test_f11_dots_are_the_median_of_the_seed_medians(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    fig = plot_coverage(frames, tmp_path, n_boot=N)
    dots = {a.get_gid(): xdata(a) for a in artists(fig, "coverage")}
    for method in ("ppo", "sil", "combo"):
        for column in FLOOR_COLUMNS:
            floor = column.removeprefix("coverage/")
            values = _seed_median(frames, method, column)
            values = values[np.isfinite(values)]
            gid = f"coverage/1m/{method}/{floor}"
            if values.size:
                assert dots[gid] == pytest.approx(np.median(values)), gid
            else:
                assert gid not in dots
    assert gids(fig, "tiles")
    assert gids(fig, "coverage")


def test_f11_intervals_are_bootstrapped_over_seeds_not_episodes(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    fig = plot_coverage(frames, tmp_path, n_boot=N)
    seg = next(a for a in artists(fig, "ci") if a.get_gid() == "ci/1m/sil/overworld")
    lo, hi = seg.get_segments()[0][:, 0]  # type: ignore[attr-defined]
    seeds = _seed_median(frames, "sil", "coverage/overworld")  # one value per seed
    assert len(seeds) == 10
    want = stratified_bootstrap_ci(
        {"seeds": seeds.tolist()}, lambda d: float(np.median(d["seeds"])), N, 0
    )
    assert (lo, hi) == (pytest.approx(want.lo), pytest.approx(want.hi))
    episodes = _method_eps(frames, "sil", "1m")["coverage/overworld"].to_numpy(float)
    assert len(episodes) == 40  # an episode-level resample would give another interval
    assert (lo, hi) != (np.percentile(episodes, 2.5), np.percentile(episodes, 97.5))


def test_f11_a_floor_never_visited_is_left_out_not_zero(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    fig = plot_coverage(frames, tmp_path, n_boot=N)
    visited = [g for g in gids(fig, "coverage") if g.endswith("/gnomish_mines")]
    eps = _method_eps(frames, "ppo", "1m")
    assert (
        eps["coverage/gnomish_mines"].isna().all()
    )  # ppo never reaches floor 2 in the fixture
    assert "coverage/1m/ppo/gnomish_mines" not in visited
    assert visited  # others do


def test_f12_missing_replay_levels_say_not_recorded(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    fig = plot_survival(frames, tmp_path, n_boot=N)
    shown = said(fig)
    assert "count-bonus-ppo-rnn not recorded" in shown  # NaN levels in the fixture
    assert not any("level/1m/count-bonus-ppo-rnn" in g for g in gids(fig, "level"))
    assert any(g.startswith("share/1m/ppo/low_share/food") for g in gids(fig, "share"))


def test_f12_dots_are_the_median_of_the_seed_medians_per_stat(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    fig = plot_survival(frames, tmp_path, n_boot=N)
    share = {a.get_gid(): xdata(a) for a in artists(fig, "share")}
    zero = {a.get_gid(): xdata(a) for a in artists(fig, "zero")}
    level = {a.get_gid(): xdata(a) for a in artists(fig, "level")}
    for method in ("ppo", "sil", "dyna-q"):
        for stat in ("food", "drink", "energy"):
            for kind, found in (("low", share), ("zero", zero)):
                column = f"{kind}_share/{stat}"
                want = np.median(_seed_median(frames, method, column))
                prefix = "share" if kind == "low" else "zero"
                assert found[f"{prefix}/1m/{method}/{column}"] == pytest.approx(want)
            column = f"survival/level/{stat}"
            want = np.median(_seed_median(frames, method, column))
            assert level[f"level/1m/{method}/{column}"] == pytest.approx(want)
    # the fixture's low and zero shares differ per stat, so a swapped row would fail
    assert share["share/1m/sil/low_share/food"] != share["share/1m/sil/low_share/drink"]


def test_f13_kills_are_mean_per_episode_with_passive_last(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    fig = plot_combat(frames, tmp_path, n_boot=N)
    heat = next(a for a in artists(fig, "heat") if a.get_gid() == "heat/1m/kills")
    cells = np.asarray(heat.get_array(), float)  # type: ignore[attr-defined]
    solo = next(a for a in artists(fig, "heat") if a.get_gid() == "heat/1m/kills-solo")
    alone = np.asarray(solo.get_array(), float)  # type: ignore[attr-defined]
    assert cells.shape == alone.shape == (3, 9)
    eps = _method_eps(frames, "sil", "1m")
    names = figures_game._contexts(frames)[0].names
    j = names.index("sil")
    assert cells[0, j] == pytest.approx(
        eps.groupby("run_id")["kills/zombie"].mean().mean()
    )
    assert alone[2, j] == pytest.approx(  # the passive row has a scale of its own
        eps.groupby("run_id")["kills/passive"].mean().mean()
    )
    assert np.isnan(cells[2]).all()
    assert np.isnan(alone[:2]).all()
    ytick = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert ytick == ["zombie", "skeleton", "passive"]
    assert "Never killed" not in said(fig) or "Not drawn" in said(fig)
    taken = {a.get_gid(): xdata(a) for a in artists(fig, "damage")}
    assert taken["damage/1m/sil"] == pytest.approx(
        eps.groupby("run_id")["damage_taken"].mean().mean()
    )
    fights = {a.get_gid(): xdata(a) for a in artists(fig, "fights")}
    # the fixture stores `slot` fight records per episode (slots 0 to 3): mean 1.5
    assert fights["fights/1m/sil/fights"] == pytest.approx(1.5)
    assert fights["fights/1m/sil/fights"] == pytest.approx(
        eps["fights"].map(len).groupby(eps["run_id"]).mean().mean()
    )


def test_f14_action_shares_sum_to_one_per_method(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    fig = plot_actions(frames, tmp_path, n_boot=N)
    heat = next(a for a in artists(fig, "heat") if a.get_gid() == "heat/1m/actions")
    cells = np.asarray(heat.get_array(), float)  # type: ignore[attr-defined]
    assert cells.shape == (5, 9)
    assert cells.sum(axis=0) == pytest.approx(np.ones(9))
    names = figures_game._contexts(frames)[0].names
    columns = [
        c for c in frames["eval_episodes"].columns if c.startswith("action_count/")
    ]
    for method in ("ppo", "sil"):
        eps = _method_eps(frames, method, "1m")
        counts = eps[columns].to_numpy(float)
        share = pd.DataFrame(counts / counts.sum(axis=1, keepdims=True))
        want = share.groupby(eps["run_id"].to_numpy()).mean().mean().to_numpy()
        assert cells[:, names.index(method)] == pytest.approx(want)
    assert not np.allclose(cells[:, 0], cells[:, 5])  # methods differ


def test_f14_training_entropy_is_not_logged_for_the_reused_runs(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    fig = plot_actions(frames, tmp_path, n_boot=N)
    train = gids(fig, "entropy")
    assert "entropy/1m/sil/train" in train
    assert not any(f"/{m}/train" in g for g in train for m in NOT_LOGGED)
    assert "entropy/1m/ppo/eval" in train  # eval episodes exist for every method
    dot = next(
        a for a in artists(fig, "entropy") if a.get_gid() == "entropy/1m/sil/train"
    )
    assert xdata(dot) == pytest.approx(1.0)  # fixture: 3 - 2 * ramp ends at 1.0
    shown = said(fig)
    assert "ppo, ppo-rnn training entropy not logged" in shown


def test_f15_gathered_and_used_are_the_episode_means(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    fig = plot_resources(frames, tmp_path, n_boot=N)
    dot = next(a for a in artists(fig, "use") if a.get_gid() == "use/1m/sil/wood")
    eps = _method_eps(frames, "sil", "1m")
    assert xdata(dot) == pytest.approx(eps["gathered/wood"].mean())
    assert ydata(dot) == pytest.approx(eps["used/wood"].mean())
    one_m = {g for g in gids(fig, "use") if g.startswith("use/1m/")}
    assert len(one_m) == 9 * 3  # nine methods, three resources
    assert "Episodes reaching diamond" in said(fig)
    t = next(a for a in artists(fig, "time") if a.get_gid() == "time/1m/sil/w2d")
    reach = eps[eps["wood_to_diamond"] >= 0]
    want = reach.groupby("run_id")["wood_to_diamond"].median().median()
    assert xdata(t) == pytest.approx(want)


def test_f16_reused_runs_are_not_logged_and_others_are_drawn(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    fig = plot_advanced(frames, tmp_path, n_boot=N)
    assert not any(f"/{m}/" in g for g in gids(fig, "advanced") for m in NOT_LOGGED)
    line = next(a for a in artists(fig, "advanced") if a.get_gid() == "advanced/1m/sil")
    # fixture: every advanced rate ends at 10 * (position + 1), sil being the sixth
    assert float(line.get_ydata()[-1]) == pytest.approx(
        10 * (list(LEVEL).index("sil") + 1)
    )  # type: ignore[attr-defined]
    assert "ppo, ppo-rnn not logged" in said(fig)
    want = 10 * (list(LEVEL).index("sil") + 1)
    # nine achievements are more rows than a dot panel holds: a profile, a line each
    lines = {a.get_gid(): a for a in artists(fig, "adv")}
    assert len(lines) == 7  # the seven methods that logged them
    y = np.asarray(lines["adv/1m/sil"].get_ydata(), float)  # type: ignore[attr-defined]
    assert y == pytest.approx([want] * 9)
    dyna = np.asarray(lines["adv/1m/dyna-q"].get_ydata(), float)  # type: ignore[attr-defined]
    assert dyna == pytest.approx([10 * (list(LEVEL).index("dyna-q") + 1)] * 9)


# --- F17 ------------------------------------------------------------------------


def test_radar_axes_are_min_max_scaled_over_all_methods_per_benchmark(
    frames: dict[str, pd.DataFrame],
) -> None:
    table = radar_table(frames)
    for (budget, axis), g in table.groupby(["benchmark", "axis"]):
        assert len(g) == 9, (budget, axis)
        ok = g["scaled"].dropna()
        raw = g["raw"].dropna()
        if raw.nunique() > 1:
            assert ok.min() == pytest.approx(0.0)
            assert ok.max() == pytest.approx(1.0)
            want = (raw - raw.min()) / (raw.max() - raw.min())
            assert g["scaled"].to_numpy() == pytest.approx(want.to_numpy())
        else:
            assert (ok == 0.5).all()
    assert set(table["axis"]) == set(figures_game.RADAR_AXES)


def test_radar_raw_values_follow_the_definitions(
    frames: dict[str, pd.DataFrame],
) -> None:
    table = radar_table(frames).set_index(["benchmark", "method", "axis"])["raw"]
    eps = _method_eps(frames, "sil", "1m")
    per_run = eps.groupby("run_id")
    assert table["1m", "sil", "crafter"] == pytest.approx(
        per_run["tech_depth"].mean().mean()
    )
    assert table["1m", "sil", "diver"] == pytest.approx(
        per_run["max_floor"].mean().mean()
    )
    assert table["1m", "sil", "precise"] == pytest.approx(
        1 - per_run["wasted_share"].mean().mean()
    )
    fighter = (eps["kills/zombie"] + eps["kills/skeleton"]) / eps["length"] * 1000
    assert table["1m", "sil", "fighter"] == pytest.approx(
        fighter.groupby(eps["run_id"]).mean().mean()
    )
    eff = eps["n_achievements"] / eps["length"] * 1000
    assert table["1m", "sil", "efficient"] == pytest.approx(
        eff.groupby(eps["run_id"]).mean().mean()
    )
    zero = eps[[f"zero_share/{s}" for s in figures_game.STATS]].max(axis=1)
    assert table["1m", "sil", "survivor"] == pytest.approx(
        1 - zero.groupby(eps["run_id"]).mean().mean()
    )


def test_f17_draws_one_polygon_per_method_at_the_scaled_values(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    fig = plot_radar(frames, tmp_path)
    table = radar_table(frames)
    table = table[table["benchmark"] == "1m"]
    lines = {a.get_gid(): a for a in artists(fig, "radar")}
    assert len(lines) == 9
    for method in ("ppo", "combo"):
        want = (
            table[table["method"] == method]
            .set_index("axis")["scaled"]
            .reindex(figures_game.RADAR_AXES)
            .fillna(0)
            .to_numpy()
        )
        drawn = np.asarray(lines[f"radar/1m/{method}"].get_ydata(), float)  # type: ignore[attr-defined]
        assert drawn[:-1] == pytest.approx(want)
        assert drawn[-1] == drawn[0]
    axes = [ax for ax in fig.axes if ax.get_visible()]
    assert len(axes) == 9
    assert all(ax.name == "polar" for ax in axes)
    assert all(ax.get_ylim() == (0.0, 1.0) for ax in axes)
    assert (tmp_path / "radar_sil.png").exists()
    assert {ax.get_title(loc="left") for ax in axes} == set(lines_methods(lines))


def lines_methods(lines: dict[str, Artist]) -> list[str]:
    return [g.split("/")[2] for g in lines]


# --- colours --------------------------------------------------------------------

METHOD_KINDS = (
    "race", "iqr", "winner", "ci", "edge", "node", "funnel-tech", "funnel-floor",
    "depth", "floor", "advanced", "band", "swatch", "escape", "ccdf", "split",
    "coverage", "tiles", "share", "level", "fights", "entropy", "wasted", "time",
    "adv", "damage", "use", "radar", "radar-fill",
)  # fmt: skip
"""gid kinds drawn in a method's colour (heat maps and stacked causes are ramps)."""


def _hexes(artist: Artist) -> set[str]:
    out: set[str] = set()
    for getter in (
        "get_color",
        "get_facecolor",
        "get_edgecolor",
        "get_markerfacecolor",
    ):
        value = getattr(artist, getter, lambda: None)()
        if value is None:
            continue
        if isinstance(value, str):
            out.add(to_hex(value))
            continue
        arr = np.atleast_2d(np.asarray(value, dtype=float))
        out |= {to_hex(tuple(row[:3])) for row in arr if row.size >= 3}
    arrow = getattr(artist, "arrow_patch", None)
    if arrow is not None:
        out |= {to_hex(arrow.get_edgecolor())}
    return out


@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize("fid", FIGS)
def test_every_method_artist_is_drawn_in_its_method_colour(
    fid: str, theme: str, frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    fig = draw(fid, frames, tmp_path, theme, "double" if fid in DOUBLE else "column")
    colors = METHOD_COLORS if theme == "video" else METHOD_COLORS_LIGHT
    checked = 0
    for kind in METHOD_KINDS:
        for artist in artists(fig, kind):
            checked += 1
            method = artist.get_gid().split("/")[2]
            if method == "all":
                continue
            assert to_hex(colors[method]) in _hexes(artist), artist.get_gid()
    assert checked or fid == "F7"


def test_the_three_controls_are_three_distinct_greys_in_the_race(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    fig = race(frames, tmp_path)
    seen = {
        a.get_gid().split("/")[2]: to_hex(a.get_color())  # type: ignore[attr-defined]
        for a in artists(fig, "race")
    }
    assert seen["pqn-rnn"] == to_hex(METHOD_COLORS_LIGHT["pqn-rnn"])
    controls = {to_hex(METHOD_COLORS_LIGHT[c]) for c in ("ppo", "ppo-rnn", "pqn-rnn")}
    assert len(controls) == 3


# --- review fixes ---------------------------------------------------------------


def test_f6_whiskers_bracket_the_median(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    fig = race(frames, tmp_path)
    whiskers = {
        a.get_gid().removeprefix("iqr/"): a.get_segments()[0][:, 0]  # type: ignore[attr-defined]
        for a in artists(fig, "iqr")
    }
    dots = {a.get_gid().removeprefix("race/"): xdata(a) for a in artists(fig, "race")}
    assert dots.keys() == whiskers.keys()
    for key, x in dots.items():
        lo, hi = whiskers[key]
        assert lo <= x <= hi, key
    assert any(lo < hi for lo, hi in whiskers.values())


def test_f6_says_how_many_cells_fewer_than_half_of_the_seeds_reached(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    runs = frames["runs"]
    rare = runs[(runs["method"] == "sil") & (runs["budget"] == "1m")]["run_id"].iloc[:6]
    curves = frames["curves"]
    drop = curves["run_id"].isin(rare) & curves["metric"].str.endswith("/place_table")
    partial = {**frames, "curves": curves[~drop]}  # sil unlocks it in 4 of 10 seeds
    fig = race(partial, tmp_path)
    assert "k < n/2) and are not drawn" in said(fig)
    assert "1 of" in said(fig)
    assert "race/1m/sil/place_table" not in gids(fig, "race")
    assert "race/1m/sil/collect_wood" in gids(fig, "race")
    assert "k < n/2" not in said(race(frames, tmp_path))


def test_order_agreement_never_compares_methods_of_different_blocks(
    frames: dict[str, pd.DataFrame],
) -> None:
    table = order_agreement(frames, n_perm=N)
    one_m = table[table["benchmark"] == "1m"].set_index(["method", "other"])
    for pair in (
        ("sil", "smdp-skills"),
        ("count-bonus-ppo-rnn", "dyna-q"),
        ("dyna-q", "sil"),
    ):
        cell = one_m.loc[pair]
        assert np.isnan(cell["tau"]), pair
        assert np.isnan(cell["p"]), pair
        assert np.isnan(cell["jaccard"])
        assert cell["n_shared"] == 0
    assert np.isfinite(one_m.loc[("sil", "ppo"), "tau"])  # a control: both blocks
    assert np.isfinite(one_m.loc[("ppo", "smdp-skills"), "tau"])
    assert np.isfinite(one_m.loc[("dyna-q", "smdp-skills"), "tau"])  # same block
    assert np.isfinite(one_m.loc[("sil", "count-bonus-ppo-rnn"), "tau"])


def test_the_agreement_matrix_leaves_cross_block_cells_blank(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    fig = plot_order_agreement(frames, tmp_path, n_perm=N)
    heat = next(a for a in artists(fig, "tau") if a.get_gid() == "tau/1m/all")
    cells = np.asarray(heat.get_array(), float)  # type: ignore[attr-defined]
    names = figures_game._contexts(frames)[0].names
    i, j = names.index("sil"), names.index("smdp-skills")
    assert np.ma.is_masked(heat.get_array()[i, j]) or np.isnan(cells[i, j])
    assert "different blocks are not compared" in said(fig)


def _no_adv(frames: dict[str, pd.DataFrame], method: str) -> dict[str, pd.DataFrame]:
    eps = frames["eval_episodes"].copy()
    adv = [c for c in eps.columns if c.startswith("adv/")]
    eps.loc[eps["method"] == method, adv] = np.nan
    return {**frames, "eval_episodes": eps}


def test_the_radar_leaves_an_axis_that_was_not_recorded_open_never_at_zero(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    broken = _no_adv(frames, "sil")
    table = radar_table(broken).set_index(["benchmark", "method", "axis"])
    assert np.isnan(table.loc[("1m", "sil", "mage"), "raw"])
    assert np.isnan(table.loc[("1m", "sil", "mage"), "scaled"])
    assert table.loc[("1m", "ppo", "mage"), "raw"] == pytest.approx(0.25)  # recorded
    fig = plot_radar(broken, tmp_path)
    line = next(a for a in artists(fig, "radar") if a.get_gid() == "radar/1m/sil")
    y = np.asarray(line.get_ydata(), float)  # type: ignore[attr-defined]
    assert np.isnan(y[figures_game.RADAR_AXES.index("mage")])
    assert np.isfinite(np.delete(y, [6, 8])).all()
    assert "not recorded on mage" in fig.captions["radar_sil"]
    assert "not recorded" not in fig.captions["radar_ppo"]


def test_radar_explorer_and_mage_follow_their_definitions(
    frames: dict[str, pd.DataFrame],
) -> None:
    table = radar_table(frames).set_index(["benchmark", "method", "axis"])["raw"]
    eps = _method_eps(frames, "combo", "1m")
    cover = eps[[c for c in eps.columns if c.startswith("coverage/")]].mean(axis=1)
    want = cover.groupby(eps["run_id"]).mean().mean()
    assert table["1m", "combo", "explorer"] == pytest.approx(want)
    adv = eps[[c for c in eps.columns if c.startswith("adv/")]]
    mage = (adv > 0).mean(axis=1).groupby(eps["run_id"]).mean().mean()
    assert table["1m", "combo", "mage"] == pytest.approx(mage)
    assert mage == pytest.approx(0.25)  # fixture: only drink_potion, in slots 2 and 3


def test_the_radar_says_survivor_is_an_upper_bound(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    assert "upper bound of survival" in said(plot_radar(frames, tmp_path))


def test_f11_says_the_new_tile_rate_is_an_episode_mean(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    assert "not a series over the episode" in said(
        plot_coverage(frames, tmp_path, n_boot=N)
    )


def test_the_race_mask_only_touches_the_first_unlock_rows(
    frames: dict[str, pd.DataFrame],
) -> None:
    masked = figures_game._masked(frames, 10_000)
    before, after = frames["curves"], masked["curves"]
    assert len(before) == len(after)
    other = ~after["metric"].str.startswith("first_unlock")
    kept = after[other].sort_values(["run_id", "env_steps", "metric"])
    base = before[~before["metric"].str.startswith("first_unlock")]
    base = base.sort_values(["run_id", "env_steps", "metric"])
    assert kept["value"].to_numpy() == pytest.approx(base["value"].to_numpy())
    assert np.isinf(after.loc[~other, "value"]).any()


def test_f7g_draws_only_the_tiers_with_a_node_and_has_no_legend(
    frames: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    fig = plot_order_graph(frames, tmp_path, "report", width="double", method="sil")
    nodes = artists(fig, "node")
    used = {round(float(a.get_ydata()[0])) for a in nodes}
    for ax in {a.axes for a in nodes}:  # one row per drawn tier, a lone one unlabelled
        ticks = [t for t in ax.get_yticklabels() if t.get_text()]
        assert len(ticks) == (len(used) if len(used) > 1 else 0)
    assert not [f for f in fig.figs.values() if f.legends]
    notes = fig.captions["order_sil"]
    assert ("No node in the tiers" in notes) == (len(used) < 4)
    assert ("All nodes are in the tier" in notes) == (len(used) == 1)


def test_one_unit_plots_of_hard_have_no_one_entry_legend(tmp_path: Path) -> None:
    from test_hard import floor_frame  # the hard fixtures

    from wut_murl_craftax.analysis.hard import floor_reach, plot_floor_reach

    seen: list[Figure] = []
    plain = Figure.savefig

    def keep(self: Figure, *args: object, **kwargs: object) -> None:
        seen.append(self)
        plain(self, *args, **kwargs)  # type: ignore[arg-type]

    one = floor_reach(floor_frame(), n=N)
    one = one[one["config_ref"] == "tuned"]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Figure, "savefig", keep)
        plot_floor_reach(one, tmp_path)
    assert not seen[0].legends
