"""Tests for the 1M vs 1B figures: one PNG per stem, a caption each, gid per method."""

import pandas as pd
import pytest
from _compare_fixture import make_frames

from wut_murl_craftax.analysis import compare, figures_scale, scale
from wut_murl_craftax.visualisation import charts

N_BOOT = 50


@pytest.fixture(scope="module")
def tables() -> dict[str, pd.DataFrame]:
    frames = make_frames()
    return {
        "scale_scaling": scale.scaling_table(frames, n_boot=N_BOOT),
        "scale_advantage": scale.advantage_table(frames, n_boot=N_BOOT),
        "speed": compare.speed_table(frames),
    }


def test_draw_writes_one_png_and_one_caption_per_stem(
    tables: dict[str, pd.DataFrame], tmp_path
) -> None:
    files = figures_scale.draw(tables, tmp_path)
    stems = {p.stem for p in files if not p.stem.endswith("_video")}
    assert stems == set(figures_scale.STEMS)
    caps = pd.read_csv(tmp_path / "captions.csv")
    assert set(caps["stem"]) == set(figures_scale.STEMS)
    assert caps["description"].str.len().gt(40).all()


def test_the_score_figure_has_one_dot_per_method(
    tables: dict[str, pd.DataFrame], tmp_path
) -> None:
    with charts.capture_captions():
        fig = figures_scale.plot_scores(tables, tmp_path)
    gids = {
        a.get_gid()
        for a in fig.findobj(lambda a: a.get_gid())
        if str(a.get_gid()).startswith("scale/")
    }
    assert len(gids) == len(tables["scale_scaling"])


def test_the_log_axes_use_few_plain_tick_labels(
    tables: dict[str, pd.DataFrame], tmp_path
) -> None:
    with charts.capture_captions():
        fig = figures_scale.plot_scores(tables, tmp_path)
    fig.canvas.draw()
    ax = fig.axes[0]
    for axis in (ax.xaxis, ax.yaxis):
        shown = [t.get_text() for t in axis.get_ticklabels() if t.get_visible()]
        shown = [t for t in shown if t]
        assert 2 <= len(shown) <= 6
        assert not any("\u00d7" in t or "^" in t or "e" in t.lower() for t in shown), (
            shown
        )
