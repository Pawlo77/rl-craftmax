"""Every figure producer shares one look: PNG only, nothing cropped, no legend on data.

Each figure kind is rendered from the toy fixtures of the figure tests in the report
column, the report double width and the video; each file it saves is audited by
`_figure_audit` (the figure as written, after the layout engine ran).
"""

import functools
import re
import tempfile
from collections.abc import Callable
from pathlib import Path

import matplotlib as mpl
import pandas as pd
import pytest
from _angles_fixture import angle_frames, write_distillation
from _figure_audit import Saves, audit_saves
from _figure_unconverted import UNCONVERTED
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
from wut_murl_craftax.visualisation.charts import ASPECT, MAX_ROWS_MM, Shape

N = 100
"""Bootstrap resamples in tests."""
VARIANTS = (("report", "column"), ("report", "double"), ("video", "column"))
ALLOWED_LEGEND_OVER_DATA: dict[str, str] = {}
"""File stem -> why its legend may sit over an axes (empty: legends stay outside)."""
METHOD_LEGEND_STEMS = ("legend_methods",)
"""The one figure that shows the method colours; no other figure has such a legend."""
DATA_NAME_STEMS = ("order_",)
"""Figures whose nodes carry their names inside the axes (data, not sentences)."""


@functools.cache
def _core() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame]:
    frames = figure_frames(("1m",))
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
    return game_frames(("1m",))


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


VARIANTS_OF_ONE = ("tuned", "monte-carlo", "no-flags", "random-top")


def _hard_variants_case(out: Path) -> None:
    """The four hard plots of one method with four variants (tuned + 3 ablations)."""
    episodes = pd.concat([achievement_frame(), stall_frame(), death_frame()])

    def variants(table: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([table.assign(config_ref=r) for r in VARIANTS_OF_ONE])

    hard.plot_floor_reach(variants(floor_reach(floor_frame(), n=N)), out)
    hard.plot_achievement_status(variants(achievement_status(achievement_frame())), out)
    hard.plot_death_causes(variants(death_causes(death_frame())), out)
    hard.plot_stall_lengths(variants(stall_ccdf(stall_table(episodes))), out)


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


DOUBLE_FIDS: set[str] = {"F7G"}
"""Figure ids drawn at 170 mm and as video only (order graphs print node names): they
have no column case. Every other figure is one column wide: no double-width case."""


def _cases() -> dict[str, Callable[[Path], None]]:
    """Case name -> a function that draws it into a folder."""
    cases: dict[str, Callable[[Path], None]] = {}
    for theme, width in VARIANTS:
        for fid, case in [
            *((f, _core_case) for f in NAMES),
            *((f, _game_case) for f in FIGS),  # F6 also saves F6W, the wall-clock race
            *((f, _angle_case) for f in ANGLE_FIGS),
        ]:
            if theme == "report" and (width == "double") != (fid in DOUBLE_FIDS):
                continue
            cases[f"{fid}-{theme}-{width}"] = (
                lambda out, f=fid, c=case, t=theme, w=width: c(f, t, w, out)
            )
    return cases | {
        "hard": _hard_case,
        "hard-variants": _hard_variants_case,
        "fidelity": _fidelity_case,
    }


CASES = _cases()


@pytest.fixture(scope="module")
def saved(tmp_path_factory: pytest.TempPathFactory) -> Callable[[str], Saves]:
    """The audited files of one case, rendered once and shared by the tests."""
    done: dict[str, tuple[Path, Saves]] = {}

    def render(case: str) -> Saves:
        if case not in done:
            out = tmp_path_factory.mktemp(case)
            with audit_saves() as files:
                CASES[case](out)
            done[case] = (out, files)
        return done[case][1]

    render.done = done  # type: ignore[attr-defined]  # read back by the PNG test
    return render


def known(request: pytest.FixtureRequest, case: str, check: str) -> None:
    """Expect ``check`` to fail on a figure kind round 4 has not converted yet.

    The strict xfail turns an entry of `UNCONVERTED` that no longer fails into a
    failure: delete it when a figure is converted, so the audit then guards it.
    """
    if check in UNCONVERTED.get(case, ()):
        request.applymarker(
            pytest.mark.xfail(strict=True, reason="not converted to round 4 yet")
        )


def stem_of(path: Path) -> str:
    """File stem without the ``_double`` / ``_video`` suffix."""
    return re.sub(r"(_double|_video)$", "", path.stem)


@pytest.mark.parametrize("case", CASES)
def test_nothing_is_cropped(
    case: str, saved: Callable[[str], Saves], request: pytest.FixtureRequest
) -> None:
    known(request, case, "cropped")
    files = saved(case)
    assert files, "no figure was saved"
    for f in files:
        assert not f.cut, f"{f.path.name}: {f.cut[:6]}"


@pytest.mark.parametrize("case", CASES)
def test_no_legend_sits_over_an_axes(
    case: str, saved: Callable[[str], Saves], request: pytest.FixtureRequest
) -> None:
    known(request, case, "legend over axes")
    for f in saved(case):
        if f.path.stem in ALLOWED_LEGEND_OVER_DATA:
            continue
        assert not f.covered, f"{f.path.name}: {f.covered}"


@pytest.mark.parametrize("case", CASES)
def test_no_two_texts_overlap(
    case: str, saved: Callable[[str], Saves], request: pytest.FixtureRequest
) -> None:
    known(request, case, "text overlap")
    for f in saved(case):
        assert not f.overlaps, f"{f.path.name}: {f.overlaps[:6]}"


@pytest.mark.parametrize("case", CASES)
def test_the_layout_engine_raises_no_warning(
    case: str, saved: Callable[[str], Saves], request: pytest.FixtureRequest
) -> None:
    known(request, case, "layout warning")
    assert not saved(case).warnings


@pytest.mark.parametrize("case", CASES)
def test_no_title_wraps(
    case: str, saved: Callable[[str], Saves], request: pytest.FixtureRequest
) -> None:
    known(request, case, "wrapped title")
    for f in saved(case):
        assert not f.wrapped_titles, f"{f.path.name}: {f.wrapped_titles}"


@pytest.mark.parametrize("case", CASES)
def test_no_method_legend_except_the_legend_figure(
    case: str, saved: Callable[[str], Saves], request: pytest.FixtureRequest
) -> None:
    known(request, case, "method legend")
    for f in saved(case):
        if stem_of(f.path) in METHOD_LEGEND_STEMS:
            assert f.method_legends, f"{f.path.name} shows no method legend"
        else:
            assert not f.method_legends, f"{f.path.name}: {f.method_legends}"


@pytest.mark.parametrize("case", CASES)
def test_one_data_axes_per_png(
    case: str, saved: Callable[[str], Saves], request: pytest.FixtureRequest
) -> None:
    known(request, case, "one axes")
    for f in saved(case):
        assert f.data_axes <= 1, f"{f.path.name}: {f.data_axes} data axes"
        if stem_of(f.path) in METHOD_LEGEND_STEMS:
            assert f.data_axes == 0


@pytest.mark.parametrize("case", CASES)
def test_no_lone_y_tick_label_names_a_quantity(
    case: str, saved: Callable[[str], Saves], request: pytest.FixtureRequest
) -> None:
    known(request, case, "lone y tick")
    for f in saved(case):
        assert not f.lone_yticks, f"{f.path.name}: {f.lone_yticks}"


@pytest.mark.parametrize("case", CASES)
def test_only_ticks_labels_legends_titles_and_numbers_are_drawn(
    case: str, saved: Callable[[str], Saves], request: pytest.FixtureRequest
) -> None:
    known(request, case, "text kinds")
    for f in saved(case):
        names = stem_of(f.path).startswith(DATA_NAME_STEMS)
        assert not f.figure_texts, f"{f.path.name}: figure text {f.figure_texts}"
        assert not f.forbidden, f"{f.path.name}: {f.forbidden}"
        assert names or not f.sentences, f"{f.path.name}: {f.sentences}"


@pytest.mark.parametrize("case", CASES)
def test_every_figure_registers_a_caption(
    case: str, saved: Callable[[str], Saves], request: pytest.FixtureRequest
) -> None:
    known(request, case, "caption")
    files = saved(case)
    captions = {c.stem: c for c in files.captions}
    for f in files:
        got = captions.get(stem_of(f.path))
        assert got is not None, f"{f.path.name} has no caption"
        assert got.title.strip()
        assert len(got.description.split()) >= 4, got


REPORT_WIDTHS_MM = (80.0, 170.0, 39.0)
"""Column, double and half-column widths: figures are included unscaled."""
VIDEO_PIXELS = (1920, 1080)


@pytest.mark.parametrize("case", CASES)
def test_the_saved_pixels_match_the_declared_size(
    case: str, saved: Callable[[str], Saves], request: pytest.FixtureRequest
) -> None:
    known(request, case, "pixels")
    for f in saved(case):
        for got, want in zip(f.pixels, f.expected_px, strict=True):
            assert abs(got - want) <= 1, f"{f.path.name}: {f.pixels} != {f.expected_px}"
        if f.expected_px != VIDEO_PIXELS:
            width = f.size_mm[0]
            assert any(abs(width - w) < 0.1 for w in REPORT_WIDTHS_MM), (
                f"{f.path.name}: {width:.2f} mm wide"
            )


@pytest.mark.parametrize("case", CASES)
def test_the_height_follows_the_size_class(
    case: str, saved: Callable[[str], Saves], request: pytest.FixtureRequest
) -> None:
    known(request, case, "size class")
    for f in saved(case):
        assert f.shape is not None, f"{f.path.name} declares no size class"
        if f.expected_px == VIDEO_PIXELS:
            continue
        width, height = f.size_mm
        if f.shape in {s.value for s in ASPECT}:
            want = ASPECT[Shape(f.shape)]
            assert abs(height / width - want) < 0.01, (
                f"{f.path.name}: {height / width:.3f} for class {f.shape}"
            )
        elif f.shape == Shape.ROWS:
            assert height <= MAX_ROWS_MM + 0.5, f"{f.path.name}: {height:.0f} mm tall"


@pytest.mark.parametrize("case", CASES)
def test_only_pngs_are_written(case: str, saved: Callable[[str], Saves]) -> None:
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
