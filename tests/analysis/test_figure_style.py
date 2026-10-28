"""Every figure producer shares one look: PNG only, nothing cropped, no legend on data.

Each figure kind is rendered from the toy fixtures of the figure tests in the report
column, the report double width and the video; each file it saves is audited by
`_figure_audit` (the figure as written, after the layout engine ran).
"""

import functools
import tempfile
from collections.abc import Callable
from pathlib import Path

import matplotlib as mpl
import pandas as pd
import pytest
from _angles_fixture import angle_frames, write_distillation
from _figure_audit import Saved, audit_saves
from _figures_fixture import figure_frames, tuning_frames
from _game_fixture import game_frames
from test_figures import draw as draw_core
from test_figures_angles import FIGS as ANGLE_FIGS
from test_figures_angles import draw as draw_angle
from test_figures_game import FIGS
from test_figures_game import draw as draw_game
from test_hard import achievement_frame, death_frame, floor_frame, stall_frame

from wut_murl_craftax.analysis import hard
from wut_murl_craftax.analysis.compare import literature_table
from wut_murl_craftax.analysis.figures import NAMES
from wut_murl_craftax.analysis.hard import (
    achievement_status,
    death_causes,
    floor_reach,
    stall_ccdf,
    stall_table,
)
from wut_murl_craftax.interpretation import run as distil

N = 100
"""Bootstrap resamples in tests."""
VARIANTS = (("report", "column"), ("report", "double"), ("video", "column"))
ALLOWED_LEGEND_OVER_DATA: dict[str, str] = {}
"""File stem -> why its legend may sit over an axes (empty: legends stay outside)."""


@functools.cache
def _core() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame]:
    frames = figure_frames()
    published = Path(tempfile.mkdtemp(prefix="published_")) / "published.yaml"
    published.write_text("{}")
    lit = literature_table(
        frames, published=published, published_hash=lambda c, _b: f"{c}/tuned",
        n_boot=N,
    )  # fmt: skip
    return frames, tuning_frames(), lit


def _core_case(fid: str, theme: str, width: str, out: Path) -> None:
    frames, tuning, lit = _core()
    draw_core(fid, frames, tuning, lit, out, theme, width)


def _game_case(fid: str, theme: str, width: str, out: Path) -> None:
    draw_game(fid, _game(), out, theme, width)


@functools.cache
def _game() -> dict[str, pd.DataFrame]:
    return game_frames()


@functools.cache
def _angles() -> tuple[dict[str, pd.DataFrame], Path]:
    """The angle fixture and its distillation folders (F27, F28)."""
    frames = angle_frames()
    trees = write_distillation(Path(tempfile.mkdtemp(prefix="trees_")), frames["runs"])
    return frames, trees


def _angle_case(fid: str, theme: str, width: str, out: Path) -> None:
    frames, trees = _angles()
    draw_angle(fid, frames, out, theme, width, trees)


def _hard_case(out: Path) -> None:
    episodes = pd.concat([achievement_frame(), stall_frame(), death_frame()])
    tuned = floor_reach(floor_frame(), n=N)
    both = pd.concat([
        tuned,
        tuned.assign(config_ref="simhash", rate=tuned["rate"] / 2),
    ])
    hard.plot_floor_reach(both, out)
    hard.plot_achievement_status(achievement_status(achievement_frame()), out)
    hard.plot_death_causes(death_causes(death_frame()), out)
    hard.plot_stall_lengths(stall_ccdf(stall_table(episodes)), out)


def _fidelity_case(out: Path) -> None:
    units = [
        ("ppo", "tuned"),
        ("sil", "tuned"),
        ("sil", "simhash"),
        ("dyna-q", "tuned"),
    ]
    rows = [
        {
            "kind": "leaves", "variant": "base", "method": m, "config_ref": ref,
            "run_id": f"{m}-{ref}-{s}", "size": n,
            "top1": min(1.0, 0.4 + 0.1 * k + 0.02 * n + 0.01 * s),
        }
        for k, (m, ref) in enumerate(units)
        for s in range(2)
        for n in (1, 2, 4, 8, 16, 32)
    ]  # fmt: skip
    curves = pd.DataFrame(rows)
    distil._plot_fidelity(curves, out, "fidelity", per_seed=False)
    distil._plot_fidelity(curves, out, "fidelity_seeds", per_seed=True)


def _cases() -> dict[str, Callable[[Path], None]]:
    """Case name -> a function that draws it into a folder."""
    cases: dict[str, Callable[[Path], None]] = {}
    for theme, width in VARIANTS:
        for fid, case in [
            *((f, _core_case) for f in NAMES),
            *((f, _game_case) for f in FIGS),  # F6 also saves F6W, the wall-clock race
            *((f, _angle_case) for f in ANGLE_FIGS),
        ]:
            cases[f"{fid}-{theme}-{width}"] = (
                lambda out, f=fid, c=case, t=theme, w=width: c(f, t, w, out)
            )
    return cases | {"hard": _hard_case, "fidelity": _fidelity_case}


CASES = _cases()


@pytest.fixture(scope="module")
def saved(tmp_path_factory: pytest.TempPathFactory) -> Callable[[str], list[Saved]]:
    """The audited files of one case, rendered once and shared by the tests."""
    done: dict[str, tuple[Path, list[Saved]]] = {}

    def render(case: str) -> list[Saved]:
        if case not in done:
            out = tmp_path_factory.mktemp(case)
            with audit_saves() as files:
                CASES[case](out)
            done[case] = (out, files)
        return done[case][1]

    render.done = done  # type: ignore[attr-defined]  # read back by the PNG test
    return render


@pytest.mark.parametrize("case", CASES)
def test_nothing_is_cropped(case: str, saved: Callable[[str], list[Saved]]) -> None:
    files = saved(case)
    assert files, "no figure was saved"
    for f in files:
        assert not f.cut, f"{f.path.name}: text outside the canvas: {f.cut}"


@pytest.mark.parametrize("case", CASES)
def test_no_legend_sits_over_an_axes(
    case: str, saved: Callable[[str], list[Saved]]
) -> None:
    for f in saved(case):
        if f.path.stem in ALLOWED_LEGEND_OVER_DATA:
            continue
        assert not f.covered, f"{f.path.name}: {f.covered}"


@pytest.mark.parametrize("case", CASES)
def test_only_pngs_are_written(case: str, saved: Callable[[str], list[Saved]]) -> None:
    saved(case)
    out, files = saved.done[case]  # type: ignore[attr-defined]
    assert {p.suffix for p in out.iterdir()} == {".png"}
    assert {f.path.suffix for f in files} == {".png"}
    for f in files:
        assert f.path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_the_theme_does_not_leak_into_the_global_rcparams(tmp_path: Path) -> None:
    before = dict(mpl.rcParams)
    for theme in ("video", "report"):
        _core_case("F2", theme, "column", tmp_path)
    assert dict(mpl.rcParams) == before
