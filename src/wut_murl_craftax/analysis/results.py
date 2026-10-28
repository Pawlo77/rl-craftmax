"""Run the method comparison: tables T1-T7, figures F1-F28, report files, MLflow run.

``python -m wut_murl_craftax.analysis.results --phases final-1m [final-1b ...] --tuning
tune-coarse tune-fine --out outputs/results --tracking-uri sqlite:///<db>?timeout=300``
wires `compare`, the `figures*` modules and `latex` (nothing is reimplemented), logs
``tables`` and ``figures`` as run ``comparison-final``. Phases never exported are
skipped (R86). The 1M ``complexity.csv`` is required unless ``--no-complexity`` (skips
its macros, their check, the fidelity figure). A failed reproduction gate is not fatal.
Figure stages keep fingerprint markers in ``<out>/.stages``; a rerun redoes changes.
@see [[2026-10-20-craftax-comparison-and-results]]
@tested-by [[tests/analysis/test_results_cli.py]]
"""

import argparse
import hashlib
import json
import logging
import re
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd

from .load import (
    METRICS_ROOT,
    Frames,
    MissingPhaseError,
    load_phases,
    write_text_atomic,
)

logger = logging.getLogger(__name__)

DEFAULT_PHASES = ("final-1m", "final-1b", "combo-1m", "combo-1b")
"""The final phases of the plan; 1B ones that were never exported are skipped (R86)."""
DEFAULT_TUNING = ("tune-coarse", "tune-fine")
"""The tuning phases T7 and the tuning-optimism table read."""
OUT = Path("outputs/results")
REPORT_DIR = Path("report")
INTERPRETATION = Path("outputs/interpretation")
"""``<budget>/complexity.csv`` and the fidelity figure of the distillation."""
RUN_NAME = "comparison-final"
"""The MLflow run (experiment ``analysis``, contracts section 8)."""
PERMUTATIONS = 10_000
"""Permutations of the Kendall tau test of F7 (the plan's number)."""
STAGES = ".stages"
"""Marker folder of ``out`` (not logged)."""
VARIANTS: tuple[tuple[str, str], ...] = (
    ("report", "column"),
    ("report", "double"),
    ("video", "column"),
)
"""(theme, width) every figure function is called with; the figure writes only the
report variant at its own width (`charts.report_width`) and the video, which has one
size and ignores width."""
BOOT_GAME = ("F8", "F8T", "F9", "F9S", "F10", "F11", "F12", "F13", "F14", "F15", "F16")
"""In-game figures that bootstrap over seeds (the others take no resampling)."""
TABLE_FILES = {"game": "game_summary"}
"""Table key -> CSV stem where they differ."""


class TableCaption(NamedTuple):
    """What one table file is: its id in the plan, a title and a description."""

    id: str
    title: str
    description: str


TABLE_CAPTIONS: dict[str, TableCaption] = {
    "main": TableCaption(
        "T1",
        "Main results",
        "Final score of every tuned method (and control) per block: mean with s.e., "
        "IQM or median with 95% bootstrap CIs, the final-checkpoint eval means, and "
        "against each control the probability of improvement with Holm-adjusted p.",
    ),
    "main_vs_controls": TableCaption(
        "T1",
        "Comparison with the controls",
        "Probability of improvement with its 95% CI and the Holm p of every method "
        "against each control, per block (LaTeX table of the main results).",
    ),
    "poi_matrix": TableCaption(
        "T1",
        "Probability of improvement matrix",
        "P(row beats column) with its 95% CI between every two methods of one block "
        "(the source of the PoI matrix figure); no cross-block pair.",
    ),
    "ablation": TableCaption(
        "T1",
        "Ablations",
        "Each ablation against the tuned runs of its method: difference of means with "
        "a bootstrap CI, IQM, PoI and Mann-Whitney p (Holm over a block); never pooled "
        "with the method.",
    ),
    "reproduction_gate": TableCaption(
        "T2",
        "Reproduction gate",
        "Each control's final score against its published value: pass or fail with the "
        "gap and the band half-width h.",
    ),
    "literature": TableCaption(
        "T2",
        "Literature comparison",
        "Our rows beside the published rows of the comparable score kind, with our CI "
        "and the reproduction-gate status of the controls; published rows keep their "
        "protocol, seeds and spread as copied.",
    ),
    "stability": TableCaption(
        "T3",
        "Stability",
        "Seed sd and CV of the final score and, per run with a score curve, the "
        "deepest drawdown, drops and collapses.",
    ),
    "stability_runs": TableCaption(
        "T3",
        "Stability per run",
        "The per-run part of the stability table: deepest drawdown, any drop and "
        "collapse of each tuned run with a score curve.",
    ),
    "hardware": TableCaption(
        "T4",
        "Hardware",
        "Run medians per hardware profile (steps per second, wall-clock, compile time, "
        "memory, utilisation); profiles are never compared, laptop timings are context "
        "only.",
    ),
    "stability_hardware": TableCaption(
        "T5",
        "Stability and hardware",
        "Seed sd, drawdown, drops and collapses beside steady steps per second, "
        "wall-clock and memory, per profile and block (LaTeX table).",
    ),
    "game_summary": TableCaption(
        "T6",
        "In-game summary",
        "Per method: race wins, deepest floor, most frequent cause of death, coverage, "
        "wasted share and adoption of the advanced mechanics.",
    ),
    "race": TableCaption(
        "T6",
        "First-unlock race",
        "Median first unlock of every achievement per method (env steps and "
        "wall-clock) with the eligibility rule and the winners, per block.",
    ),
    "sensitivity": TableCaption(
        "T7",
        "Hyperparameter sensitivity",
        "Per tuning phase and method: configs tried, the best config, robust share, "
        "the noise floor and the two most sensitive hyperparameters.",
    ),
    "tuning_optimism": TableCaption(
        "T7",
        "Tuning optimism",
        "Mean score of the tuned configuration during tuning against fresh seeds of "
        "the same configuration, per method.",
    ),
}
"""Table stem (CSV and LaTeX share it) -> its `TableCaption`: what each
file of ``tables/<budget>/`` is, written to its ``captions.csv``."""
COMPLEXITY_MACRO = re.compile(
    r"^(?:lEighty|lNinety|depthNinety|fid|expectedFid|closedLoop|spearman|majority|"
    r"gain|peakVal|group)"
    r"[A-Za-z]*?(?:OneM|OneB)"
)
"""Names of the macros built from ``complexity.csv`` (`latex.COMPLEXITY_QUANTITIES`)."""
DISTIL_FIGURES = ("F27", "F28")
"""Figures that read the distillation's policy folders: their stage also hashes them."""
BUDGET_ONLY: dict[str, tuple[str, ...]] = {
    "F3": ("1m",),
    "F18": ("1m",),
    "F20": ("1m",),
}
"""Figure ids that exist for some benchmarks only: the PoI matrix needs the 10 seeds of
1M (3 at 1B), the tuning phases and the hardware figure are 1M's."""
FIDELITY = "fidelity_vs_leaves"
"""Stem of the distillation figure; it sits in ``figures/<budget>/`` (contracts 3).

The distillation writes ``<budget>/fidelity_vs_leaves.png`` (older runs
``fidelity_vs_leaves_<budget>.png``, still found)."""


@dataclass(slots=True)
class ResultsOptions:
    """Everything one comparison run needs; `main` fills it from the command line."""

    phases: Sequence[str] = DEFAULT_PHASES
    tuning: Sequence[str] = DEFAULT_TUNING
    out: Path = OUT
    report_dir: Path = REPORT_DIR
    metrics_root: Path = METRICS_ROOT
    complexity_root: Path = INTERPRETATION
    tracking_uri: str | None = None
    complexity: bool = True
    log: bool = True
    force: bool = False
    n_boot: int | None = None
    """Bootstrap resamples; None is `evaluation.stats.BOOTSTRAP_N`."""
    n_perm: int = PERMUTATIONS
    seed: int = 0
    published: Path | None = None
    """``published_scores.yaml`` of the gate; None is the repository's."""
    published_hash: Callable[[str, str], str] | None = None
    """Config hash of a control's published config; None reads the registry."""


@dataclass(slots=True)
class ResultsSummary:
    """What a run did: stages, failed gates, problems found late, the MLflow run."""

    done: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed_gates: list[str] = field(default_factory=list)
    """One line per control that missed its published value (gap and half-width h)."""
    problems: list[str] = field(default_factory=list)
    """Errors found after the files were written (a macro the report lacks)."""
    run_id: str | None = None


# ---------------------------------------------------------------------- inputs


def _digest(parts: object) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


def _file_hash(path: Path) -> str | None:
    """Content hash of a file, None if it is absent."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def default_published() -> Path:
    """The repository's ``published_scores.yaml`` (what the gate reads by default)."""
    from ..evaluation.reproduction import PUBLISHED_SCORES

    return PUBLISHED_SCORES


def code_files() -> list[Path]:
    """Sources whose change must redo the figures (also the combo-pair inference)."""
    root = Path(__file__).resolve().parents[1]
    return sorted([
        *(root / "analysis").glob("*.py"),
        *(root / "evaluation").glob("*.py"),
        root / "visualisation" / "charts.py",
        root / "training" / "interface.py",  # combo_pair decides a combo's block
    ])


def code_digest() -> str:
    """Hash of `code_files`, so a code change invalidates every stage marker."""
    h = hashlib.sha256()
    for path in code_files():
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def _stat(path: Path) -> list[object] | None:
    """Name, size and mtime of a file, None if it is absent."""
    try:
        st = path.stat()
    except OSError:
        return None
    return [path.name, st.st_size, st.st_mtime_ns]


class _Inputs:
    """The export tables of the run: the run tables at once, the big ones on demand."""

    def __init__(self, o: ResultsOptions) -> None:
        self.o = o
        try:
            self.runs = load_phases(
                o.phases, o.metrics_root, skip_missing=True, tables=("runs",)
            )
            self.tuning = (
                load_phases(o.tuning, o.metrics_root, tables=("runs",))
                if o.tuning
                else None
            )
        except MissingPhaseError as e:
            raise SystemExit(str(e)) from e
        seen = set(self.runs["runs"]["phase"])
        self.phases = [p for p in o.phases if p in seen]

    @cached_property
    def full(self) -> Frames:
        """The final phases with curves and eval episodes (the big tables)."""
        return load_phases(self.phases, self.o.metrics_root, skip_missing=True)

    @cached_property
    def joint(self) -> Frames:
        """Final then tuning runs in one frame, for the tuning-optimism table."""
        return load_phases(
            [*self.phases, *self.o.tuning], self.o.metrics_root, skip_missing=True,
            tables=("runs",),
        )  # fmt: skip

    def fingerprint(self) -> str:
        """Hash of the export files, the options that change a number, the inputs."""
        from .literature import LITERATURE_CSV

        o = self.o
        files = {
            p: [_stat(f) for f in sorted((o.metrics_root / p).glob("*.parquet"))]
            for p in (*self.phases, *o.tuning)
        }
        return _digest({
            "files": files,
            "phases": self.phases,
            "tuning": list(o.tuning),
            "n_boot": o.n_boot,
            "n_perm": o.n_perm,
            "seed": o.seed,
            "published": _file_hash(o.published or default_published()),
            "literature": _stat(LITERATURE_CSV),
            "code": code_digest(),
        })


# ---------------------------------------------------------------------- stages


class _Stages:
    """Run each stage once per input fingerprint (markers in ``<out>/.stages``)."""

    def __init__(
        self, out: Path, fingerprint: str, *, force: bool, s: ResultsSummary
    ) -> None:
        self.out, self.fingerprint, self.force, self.s = out, fingerprint, force, s

    def _marker(self, name: str) -> Path:
        return self.out / STAGES / f"{name}.json"

    def current(self, name: str) -> bool:
        """True if ``name`` finished for these inputs and its files still exist."""
        marker = self._marker(name)
        if self.force or not marker.is_file():
            return False
        try:
            saved = json.loads(marker.read_text())
        except ValueError:  # cut in half by a kill: the stage is not finished
            return False
        return saved["fingerprint"] == self.fingerprint and all(
            (self.out / f).is_file() for f in saved["files"]
        )

    def run(self, name: str, build: Callable[[], list[Path]]) -> bool:
        """Run ``build`` unless current; it returns the files it wrote.

        Returns:
            True if it ran, False if the stage was already finished.
        """
        if self.current(name):
            self.s.skipped.append(name)
            logger.info("stage %s is current, skipped", name)
            return False
        start = time.perf_counter()
        files = build()
        marker = self._marker(name)
        marker.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomic(
            marker,
            json.dumps({
                "fingerprint": self.fingerprint,
                "files": sorted(str(f.relative_to(self.out)) for f in files),
            }),
        )
        self.s.done.append(name)
        logger.info("stage %s done in %.1f s", name, time.perf_counter() - start)
        return True


# ---------------------------------------------------------------------- tables


def build_tables(inputs: _Inputs) -> dict[str, pd.DataFrame]:
    """Every comparison table, keyed as in `write_table_files`."""
    from . import compare

    o, frames = inputs.o, inputs.full
    boot: dict[str, Any] = {"n_boot": o.n_boot, "seed": o.seed}
    gate: dict[str, Any] = {}
    if o.published is not None:
        gate["published"] = o.published
    if o.published_hash is not None:
        gate["published_hash"] = o.published_hash
    t = {
        "main": compare.main_table(frames, **boot),
        "poi_matrix": compare.poi_matrix(frames, **boot),
        "ablation": compare.ablation_table(frames, **boot),
        "reproduction_gate": compare.reproduction_gate(frames, **gate),
        "literature": compare.literature_table(frames, **gate, **boot),
        "stability": compare.stability_table(frames),
        "stability_runs": compare.run_stability(frames),
        "hardware": compare.hardware_table(frames),
        "race": compare.race_table(frames),
        "game": compare.game_summary_table(frames),
    }
    tuned = inputs.tuning is not None
    t["sensitivity"] = (
        compare.sensitivity_table(inputs.tuning)  # type: ignore[arg-type]
        if tuned
        else pd.DataFrame(columns=compare.SENSITIVITY_TABLE_COLUMNS)
    )
    t["tuning_optimism"] = (
        compare.tuning_optimism(inputs.joint)
        if tuned
        else pd.DataFrame(columns=compare.TUNING_OPTIMISM_COLUMNS)
    )
    return t


def write_table_captions(out: Path) -> list[Path]:
    """``captions.csv`` of every ``<out>/tables/<budget>/`` folder: what its files are.

    One row per file stem the folder holds (CSV or LaTeX), from `TABLE_CAPTIONS`
    (``stem,id,title,description``, grouped by table id). Run after the LaTeX tables.
    """
    from ..visualisation import charts

    paths = []
    for folder in sorted(p for p in (out / "tables").iterdir() if p.is_dir()):
        have = {p.stem for p in [*folder.glob("*.csv"), *folder.glob("*.tex")]}
        (folder / "captions.csv").unlink(missing_ok=True)
        by_id: dict[str, list[charts.Caption]] = {}
        for stem, cap in TABLE_CAPTIONS.items():
            if stem in have:
                by_id.setdefault(cap.id, []).append(
                    charts.Caption(stem, cap.title, cap.description)
                )
        for tid, group in by_id.items():
            paths.append(charts.write_captions(folder, tid, group))
    return paths


def write_table_files(tables: Mapping[str, pd.DataFrame], out: Path) -> list[Path]:
    """The CSV of each table into ``<out>/tables/<budget>/``, rewritten on every run.

    Each benchmark folder holds only that benchmark's rows (``1m/main.csv``); a table
    without a ``benchmark`` column is shared and sits in ``<out>/tables`` itself.
    The flat tables of the old layout (`latex.OLD_FLAT_TABLES`) are deleted, no other
    file.
    """
    from . import latex

    folder = out / "tables"
    folder.mkdir(parents=True, exist_ok=True)
    latex.drop_old_flat(folder, latex.OLD_FLAT_TABLES, (".csv", ".tex"))
    files: list[Path] = []
    budgets = sorted({
        str(b) for f in tables.values() if "benchmark" in f for b in f["benchmark"]
    })
    for key, frame in tables.items():
        stem = TABLE_FILES.get(key, key)
        parts = (
            [(b, frame[frame["benchmark"] == b]) for b in budgets]
            if "benchmark" in frame
            else [("", frame)]
        )
        for budget, part in parts:
            if budget and part.empty and not frame.empty:
                continue  # a table with no row at all keeps its header in every folder
            target = folder / budget
            target.mkdir(parents=True, exist_ok=True)
            path = target / f"{stem}.csv"
            part.to_csv(path, index=False)
            files.append(path)
    return files


# --------------------------------------------------------------------- figures


def figure_functions() -> dict[str, Callable[..., Any]]:
    """Figure id -> function, in the plan's order (one function per file family)."""
    from . import figures, figures_angles, figures_game

    core: dict[str, Callable[..., Any]] = dict(figures.FIGURES)
    game: dict[str, Callable[..., Any]] = dict(figures_game.GAME_FIGURES)
    game["F7G"] = figures_game.plot_order_graphs  # one file per method
    order = ["LEG", "F1", "F2", "F3", "F4", "F5", "F18", "F19", "F20"]
    return {k: core[k] for k in order} | game | dict(figures_angles.ANGLE_FIGURES)


def figure_kwargs(
    fid: str, t: Mapping[str, pd.DataFrame], o: ResultsOptions
) -> dict[str, Any]:
    """The precomputed tables and the resampling options a figure function takes."""
    boot: dict[str, Any] = {"n_boot": o.n_boot, "seed": o.seed}
    given: dict[str, dict[str, Any]] = {
        "F1": boot,
        "F2": {"table": t["main"], "ablation": t["ablation"], **boot},
        "F3": {"table": t["poi_matrix"], **boot},
        "F4": {"literature": t["literature"], **boot},
        "F18": {"table": t["sensitivity"]},
        "F19": {"per_run": t["stability_runs"], "table": t["stability"]},
        "F20": {"table": t["hardware"]},
        "F7": {"n_perm": o.n_perm, "seed": o.seed},
        "F21": boot,
        "F22": boot,
        "F24": {"table": t["ablation"], **boot},
        "F25": {"table": t["tuning_optimism"], **boot},
        "F27": {"interpretation": o.complexity_root},
        "F28": {"interpretation": o.complexity_root},
    }
    return given.get(fid, boot if fid in BOOT_GAME else {})


def _snapshot(folder: Path) -> dict[Path, int]:
    return {p: p.stat().st_mtime_ns for p in folder.rglob("*") if p.is_file()}


def for_budget(kwargs: Mapping[str, Any], budget: str) -> dict[str, Any]:
    """``kwargs`` with every table that has a ``benchmark`` column cut to ``budget``."""
    out: dict[str, Any] = {}
    for key, value in kwargs.items():
        has = isinstance(value, pd.DataFrame) and "benchmark" in value.columns
        out[key] = value[value["benchmark"] == budget] if has else value
    return out


def draw(
    fn: Callable[..., Any],
    frames: Frames,
    folder: Path,
    kwargs: Mapping[str, Any],
    fid: str | None = None,
) -> list[Path]:
    """Draw one figure function in every variant, per benchmark; return the files.

    A benchmark's figures go to ``<folder>/<budget>/`` (``1m/``), made from that
    benchmark's runs and table rows alone, so a later 1B run adds ``1b/`` and leaves
    ``1m/`` as it is. With ``fid`` the captions the function registered are written
    to ``<folder>/<budget>/captions.csv`` (`charts.write_captions`).
    """
    from ..visualisation import charts
    from .load import select

    folder.mkdir(parents=True, exist_ok=True)
    before = _snapshot(folder)
    budgets = sorted(set(frames["runs"]["budget"]))
    for budget in budgets:
        if budget not in BUDGET_ONLY.get(fid or "", budgets):
            continue
        out = charts.figure_dir(folder, budget)
        part = frames if len(budgets) == 1 else select(frames, budget=budget, pool=True)
        with charts.capture_captions() as captions:
            for theme, width in VARIANTS:
                fn(part, out, theme, width=width, **for_budget(kwargs, budget))
        if fid is not None and captions:
            charts.write_captions(out, fid, captions)
    return [p for p, m in _snapshot(folder).items() if before.get(p) != m]


def drop_old_pdfs(folder: Path) -> None:
    """Delete what earlier runs left in ``folder`` that this layout replaces.

    The flat multi-panel figures of the old layout (`latex.OLD_FLAT_FIGURES`, every
    variant) and the ``*.pdf`` of any figure (figures are PNG only); other files stay.
    """
    from . import latex

    for path in latex.drop_old_flat(folder, latex.OLD_FLAT_FIGURES, (".png", ".pdf")):
        logger.info("removed %s (old flat figure)", path)
    for path in [
        *latex.drop_pdfs(folder),
        *(p for sub in sorted(folder.glob("*/")) for p in latex.drop_pdfs(sub)),
    ]:
        logger.info("removed %s (figures are PNG only)", path)


def drop_double_files(folder: Path) -> None:
    """Delete the ``<stem>_double.png`` of each benchmark folder once ``<stem>.png`` is.

    A figure has one report image, written as ``<stem>.png`` at its own width; the
    second ``_double`` image of the older layout is stale. Run after the figures are
    drawn; a ``_double`` file without a ``<stem>.png`` stays.
    """
    from . import latex

    for sub in sorted(folder.glob("*/")):
        for path in latex.drop_double_files(sub):
            logger.info("removed %s (one report image per figure)", path)


def draw_figures(inputs: _Inputs, st: _Stages, t: Mapping[str, pd.DataFrame]) -> None:
    """One stage per figure id; F18 reads the tuning phases, the rest the finals."""
    o = inputs.o
    for fid, fn in figure_functions().items():
        frames = inputs.tuning if fid == "F18" else None
        if fid == "F18" and frames is None:
            continue  # no tuning phases were asked for

        def build(
            fid: str = fid, fn: Callable[..., Any] = fn, frames: Frames | None = frames
        ) -> list[Path]:
            return draw(
                fn,
                frames or inputs.full,
                o.out / "figures",
                figure_kwargs(fid, t, o),
                fid,
            )

        if fid == "F20" and "1b" in set(inputs.runs["runs"]["budget"]):
            logger.warning("F20 hardware is drawn for Craftax-1M only (one stem)")
        (_distil_stages(o, st) if fid in DISTIL_FIGURES else st).run(
            f"figures/{fid}", build
        )


def _distil_stages(o: ResultsOptions, st: _Stages) -> _Stages:
    """`st` with a fingerprint over the finished policy folders too.

    A live distillation adds policies without touching the export tables.
    """
    summaries = sorted(o.complexity_root.glob("*/*/*/complexity.json"))
    digest = _digest([st.fingerprint, [_stat(p) for p in summaries]])
    return _Stages(st.out, digest, force=st.force, s=st.s)


def copy_fidelity(
    o: ResultsOptions, folder: Path, budgets: Sequence[str]
) -> list[Path]:
    """Copy the distillation's fidelity figure of each of ``budgets`` to its folder.

    ``<complexity_root>/<budget>/fidelity_vs_leaves[_<budget>].png`` becomes
    ``<folder>/<budget>/fidelity_vs_leaves.png`` with a row in that folder's
    ``captions.csv`` (id ``FID``).
    """
    from ..interpretation import run as interpretation
    from ..visualisation import charts

    text = interpretation.FIDELITY_DESCRIPTION.format(seeds="")  # it owns the text
    copied = []
    for budget in budgets:
        root = o.complexity_root / budget
        found = [root / f"{FIDELITY}.png", root / f"{FIDELITY}_{budget}.png"]
        src = next((p for p in found if p.is_file()), None)
        if src is None:
            if budget == "1m":
                logger.warning(
                    "%s is missing: the report's fidelity figure is not copied",
                    found[0],
                )
            continue
        dest = charts.figure_dir(folder, budget)
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest / f"{FIDELITY}.png")
        charts.write_captions(
            dest,
            "FID",
            [charts.Caption(FIDELITY, "Distillation fidelity", text, charts.COLUMN_MM)],
        )
        copied.append(dest / f"{FIDELITY}.png")
    return copied


# ----------------------------------------------------------------------- report


def write_report_files(
    inputs: _Inputs, t: Mapping[str, pd.DataFrame], s: ResultsSummary
) -> None:
    """LaTeX tables, ``numbers.tex`` and the report figures; errors go to ``s``."""
    from . import latex

    o = inputs.o
    boot: dict[str, Any] = {"n_boot": o.n_boot, "seed": o.seed}
    tex_dir = o.out / "tables"
    latex.write_tables(
        tables={k: t[k] for k in ("main", "literature", "stability", "hardware")}
        | {"game": t["game"]},
        out_dir=tex_dir, **boot,
    )  # fmt: skip
    latex.copy_tables(tex_dir, o.report_dir / "tables")
    numbers = o.report_dir / "numbers.tex"
    if not o.complexity and _has_complexity_macros(numbers):
        numbers = o.report_dir / "numbers.partial.tex"  # keep the full file intact
        logger.warning("%s holds complexity macros: writing %s", o.report_dir, numbers)
    try:
        latex.write_numbers(
            numbers, inputs.runs,
            tables={"main": t["main"], "ablation": t["ablation"]},
            complexity_root=o.complexity_root, complexity=o.complexity,
            report=o.report_dir / "report.tex" if o.complexity else None, **boot,
        )  # fmt: skip
    except ValueError as e:
        logger.error("numbers.tex not complete: %s", e)  # noqa: TRY400
        s.problems.append(str(e))
    report_tex = o.report_dir / "report.tex"
    if not o.complexity and report_tex.is_file():  # check what this run defines
        missing = [
            m
            for m in latex.undefined_macros(report_tex, numbers)
            if not COMPLEXITY_MACRO.match(m)
        ]
        if missing:
            msg = f"{report_tex} uses macros that {numbers} does not define: {missing}"
            logger.error(msg)
            s.problems.append(msg)
    latex.copy_figures(o.out / "figures", o.report_dir / "figures")


def _has_complexity_macros(numbers: Path) -> bool:
    """True if an existing ``numbers.tex`` defines a macro of ``complexity.csv``."""
    from . import latex

    if not numbers.is_file():
        return False
    defined = latex.defined_macros(numbers.read_text())
    return any(COMPLEXITY_MACRO.match(m) for m in defined)


def _gate_lines(gate: pd.DataFrame) -> list[str]:
    fails = gate[gate["status"] == "fail"]
    return [
        f"{r.control} {r.benchmark}: gap {r.gap:+.2f}, h {r.h:.2f}"
        for r in fails.itertuples()
    ]


# ---------------------------------------------------------------------- logging


def _outputs_digest(out: Path) -> str:
    """Content hash of what the MLflow run holds (tables and figures)."""
    h = hashlib.sha256()
    for folder in ("tables", "figures"):
        for p in sorted((out / folder).rglob("*")):
            if p.is_file():
                h.update(str(p.relative_to(out)).encode())
                h.update(p.read_bytes())
    return h.hexdigest()


def _read_marker(path: Path) -> dict[str, Any]:
    """A marker's content; empty if it is cut in half (the stage then reruns)."""
    try:
        saved = json.loads(path.read_text())
    except ValueError:
        return {}
    return saved if isinstance(saved, dict) else {}


def _run_exists(tracking_uri: str, run_id: str) -> bool:
    """True if the logged run is still in the store (and not deleted)."""
    import mlflow

    from ..orchestration.tracking import resolve_store_uri

    uri = resolve_store_uri(tracking_uri)
    try:
        run = mlflow.MlflowClient(uri).get_run(run_id)
    except mlflow.exceptions.MlflowException:
        return False
    return run.info.lifecycle_stage == "active"


def log_results(inputs: _Inputs, st: _Stages, s: ResultsSummary) -> None:
    """Log ``tables`` and ``figures`` as run ``comparison-final`` unless current."""
    o = inputs.o
    marker = st.out / STAGES / "log.json"
    digest = _digest([_outputs_digest(o.out), st.fingerprint, o.tracking_uri])
    if not s.done and not o.force and marker.is_file():
        saved = _read_marker(marker)
        if saved.get("digest") == digest and _run_exists(
            str(o.tracking_uri), saved["run_id"]
        ):
            s.skipped.append("log")
            s.run_id = saved["run_id"]
            return
    from ..orchestration.tracking import log_analysis_run

    tags = {
        "analysis/kind": "comparison",
        "analysis/phases": ",".join(inputs.phases),
        "analysis/tuning": ",".join(o.tuning),
    }
    start = time.perf_counter()
    paths = [o.out / "tables", o.out / "figures"]
    s.run_id = log_analysis_run(o.out, paths, RUN_NAME, tags, str(o.tracking_uri))
    marker.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(marker, json.dumps({"digest": digest, "run_id": s.run_id}))
    s.done.append("log")
    logger.info("logged run %s in %.1f s", s.run_id, time.perf_counter() - start)


# -------------------------------------------------------------------------- run


def check_complexity(o: ResultsOptions) -> None:
    """Refuse a full run without the 1M ``complexity.csv`` (before any work).

    Raises:
        SystemExit: Naming the file and the ``--no-complexity`` alternative.
    """
    has_1m = any(p.endswith("-1m") and (o.metrics_root / p).is_dir() for p in o.phases)
    path = o.complexity_root / "1m" / "complexity.csv"
    if o.complexity and has_1m and not path.is_file():
        msg = (
            f"{path} is missing: the distillation of the 1M finals "
            "(interpretation Task 6) writes it; run that first, or pass "
            "--no-complexity for a partial run without its macros and figure"
        )
        raise SystemExit(msg)


def run_comparison(o: ResultsOptions) -> ResultsSummary:
    """Run the comparison; finished stages are skipped (see the module docstring).

    Raises:
        SystemExit: If the 1M ``complexity.csv`` is missing without ``no_complexity``,
            or a phase has no export.
    """
    if o.n_boot is None:
        from ..evaluation.stats import BOOTSTRAP_N

        o.n_boot = BOOTSTRAP_N
    check_complexity(o)
    s = ResultsSummary()
    inputs = _Inputs(o)
    st = _Stages(o.out, inputs.fingerprint(), force=o.force, s=s)
    o.out.mkdir(parents=True, exist_ok=True)
    t = build_tables(inputs)  # a few seconds; the figures and macros need it
    write_table_files(t, o.out)
    drop_old_pdfs(o.out / "figures")
    s.failed_gates = _gate_lines(t["reproduction_gate"])
    draw_figures(inputs, st, t)
    drop_double_files(o.out / "figures")
    if o.complexity:
        budgets = sorted(set(inputs.runs["runs"]["budget"]))
        copy_fidelity(o, o.out / "figures", budgets)
    else:  # an earlier full run's copies would go on to the report as if current
        for stale in (o.out / "figures").glob(f"*/{FIDELITY}.*"):
            stale.unlink()
    write_report_files(inputs, t, s)
    write_table_captions(o.out)
    if o.log:
        log_results(inputs, st, s)
    return s


# -------------------------------------------------------------------------- CLI


def _parse(
    argv: Sequence[str] | None,
) -> tuple[argparse.ArgumentParser, ResultsOptions]:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("--phases", nargs="+", default=list(DEFAULT_PHASES))
    p.add_argument(
        "--tuning", nargs="*", default=list(DEFAULT_TUNING),
        help="tuning phases for T7 and the tuning optimism (none: those are skipped)",
    )  # fmt: skip
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    p.add_argument("--metrics-root", type=Path, default=METRICS_ROOT)
    p.add_argument("--complexity-root", type=Path, default=INTERPRETATION)
    p.add_argument(
        "--tracking-uri",
        help="sqlite:/// store of the MLflow run; ?timeout=<s> is the busy timeout",
    )
    p.add_argument(
        "--no-complexity", action="store_true",
        help="partial run: skip the complexity macros, their check and the figure",
    )  # fmt: skip
    p.add_argument("--no-log", action="store_true", help="write the files only")
    p.add_argument("--force", action="store_true", help="redo finished stages")
    p.add_argument("--n-boot", type=int, help="bootstrap resamples (default: stats)")
    p.add_argument("--n-perm", type=int, default=PERMUTATIONS)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    if not a.no_log and not (a.tracking_uri or "").partition("?")[0].startswith(
        "sqlite:///"
    ):
        p.error(f"--tracking-uri must be a sqlite:/// store, got {a.tracking_uri}")
    return p, ResultsOptions(
        phases=a.phases, tuning=a.tuning, out=a.out, report_dir=a.report_dir,
        metrics_root=a.metrics_root, complexity_root=a.complexity_root,
        tracking_uri=a.tracking_uri, complexity=not a.no_complexity,
        log=not a.no_log, force=a.force, n_boot=a.n_boot, n_perm=a.n_perm,
        seed=a.seed,
    )  # fmt: skip


def main(argv: Sequence[str] | None = None) -> None:
    """Run the comparison from the command line; exit 1 on a problem found at the end.

    Raises:
        SystemExit: If an input is missing or the report uses a macro that
            ``numbers.tex`` lacks (everything is written and logged first).
    """
    _, o = _parse(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    s = run_comparison(o)
    for line in s.failed_gates:
        logger.warning(
            "reproduction gate FAILED, %s: open an issue note and report the gap",
            line,
        )
    logger.info("%d stages done, %d skipped", len(s.done), len(s.skipped))
    if s.problems:
        raise SystemExit("; ".join(s.problems))


if __name__ == "__main__":
    main()
