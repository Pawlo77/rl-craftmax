"""Run the method comparison: tables T1-T7, figures F1-F20, report files, MLflow run.

``python -m wut_murl_craftax.analysis.results --phases final-1m [final-1b ...] --tuning
tune-coarse tune-fine --out outputs/results --tracking-uri sqlite:///<db>?timeout=300``
wires `compare`, `figures`, `figures_game` and `latex` (nothing is reimplemented), logs
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
from typing import Any

import pandas as pd

from .load import METRICS_ROOT, Frames, MissingPhaseError, load_phases

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
"""(theme, width) of every figure; the video theme has one size and ignores width."""
BOOT_GAME = ("F8", "F8T", "F9", "F9S", "F10", "F11", "F12", "F13", "F14", "F15", "F16")
"""In-game figures that bootstrap over seeds (the others take no resampling)."""
TABLE_FILES = {"game": "game_summary"}
"""Table key -> CSV stem where they differ."""
COMPLEXITY_MACRO = re.compile(
    r"^(?:lEighty|lNinety|depthNinety|fid|expectedFid|closedLoop|spearman)"
    r"[A-Za-z]*?(?:OneM|OneB)"
)
"""Names of the macros built from ``complexity.csv`` (`latex.COMPLEXITY_QUANTITIES`)."""
FIDELITY = "fidelity_vs_leaves_{budget}"
"""Stem of the distillation figure per budget (contracts section 3)."""


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
    """Sources whose change must redo the figures: analysis, evaluation, charts."""
    root = Path(__file__).resolve().parents[1]
    return sorted([
        *(root / "analysis").glob("*.py"),
        *(root / "evaluation").glob("*.py"),
        root / "visualisation" / "charts.py",
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
        saved = json.loads(marker.read_text())
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
        marker.write_text(
            json.dumps({
                "fingerprint": self.fingerprint,
                "files": sorted(str(f.relative_to(self.out)) for f in files),
            })
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


def write_table_files(tables: Mapping[str, pd.DataFrame], out: Path) -> list[Path]:
    """The CSV of each table into ``<out>/tables`` (rewritten on every run)."""
    folder = out / "tables"
    folder.mkdir(parents=True, exist_ok=True)
    files = []
    for key, frame in tables.items():
        path = folder / f"{TABLE_FILES.get(key, key)}.csv"
        frame.to_csv(path, index=False)
        files.append(path)
    return files


# --------------------------------------------------------------------- figures


def figure_functions() -> dict[str, Callable[..., Any]]:
    """Figure id -> function, in the plan's order (one function per file family)."""
    from . import figures, figures_game

    core: dict[str, Callable[..., Any]] = dict(figures.FIGURES)
    game: dict[str, Callable[..., Any]] = dict(figures_game.GAME_FIGURES)
    game["F7G"] = figures_game.plot_order_graphs  # one file per method
    game["F17"] = figures_game.plot_radars  # one file per budget
    order = ["F1", "F2", "F3", "F2F3", "F4", "F5", "F18", "F19", "F20"]
    return {k: core[k] for k in order} | game


def figure_kwargs(
    fid: str, t: Mapping[str, pd.DataFrame], o: ResultsOptions
) -> dict[str, Any]:
    """The precomputed tables and the resampling options a figure function takes."""
    boot: dict[str, Any] = {"n_boot": o.n_boot, "seed": o.seed}
    given: dict[str, dict[str, Any]] = {
        "F1": boot,
        "F2": {"table": t["main"], "ablation": t["ablation"], **boot},
        "F3": {"table": t["poi_matrix"], **boot},
        "F2F3": {"table": t["main"], "poi": t["poi_matrix"], **boot},
        "F4": {"literature": t["literature"], **boot},
        "F18": {"table": t["sensitivity"]},
        "F19": {"per_run": t["stability_runs"], "table": t["stability"]},
        "F20": {"table": t["hardware"]},
        "F7": {"n_perm": o.n_perm, "seed": o.seed},
    }
    return given.get(fid, boot if fid in BOOT_GAME else {})


def _snapshot(folder: Path) -> dict[Path, int]:
    return {p: p.stat().st_mtime_ns for p in folder.glob("*") if p.is_file()}


def draw(
    fn: Callable[..., Any],
    frames: Frames,
    folder: Path,
    kwargs: Mapping[str, Any],
) -> list[Path]:
    """Draw one figure function in every variant; return the files it wrote."""
    folder.mkdir(parents=True, exist_ok=True)
    before = _snapshot(folder)
    for theme, width in VARIANTS:
        fn(frames, folder, theme, width=width, **kwargs)
    return [p for p, m in _snapshot(folder).items() if before.get(p) != m]


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
                fn, frames or inputs.full, o.out / "figures", figure_kwargs(fid, t, o)
            )

        if fid == "F20" and "1b" in set(inputs.runs["runs"]["budget"]):
            logger.warning("F20 hardware is drawn for Craftax-1M only (one stem)")
        st.run(f"figures/{fid}", build)


def copy_fidelity(
    o: ResultsOptions, folder: Path, budgets: Sequence[str]
) -> list[Path]:
    """Copy the distillation's fidelity figures of ``budgets`` into ``folder``."""
    copied = []
    for budget in budgets:
        stem = FIDELITY.format(budget=budget)
        found = [
            src
            for ext in ("pdf", "png")
            if (src := o.complexity_root / budget / f"{stem}.{ext}").is_file()
        ]
        if not found and budget == "1m":
            logger.warning(
                "%s is missing: the report's fidelity figure is not copied",
                o.complexity_root / budget / f"{stem}.pdf",
            )
        for src in found:
            folder.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, folder / src.name)
            copied.append(folder / src.name)
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
    written = latex.write_tables(
        tables={k: t[k] for k in ("main", "literature", "stability", "hardware")}
        | {"game": t["game"]},
        out_dir=tex_dir, **boot,
    )  # fmt: skip
    (o.report_dir / "tables").mkdir(parents=True, exist_ok=True)
    for path in written:
        shutil.copy2(path, o.report_dir / "tables" / path.name)
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


def _run_exists(tracking_uri: str, run_id: str) -> bool:
    """True if the logged run is still in the store (and not deleted)."""
    import mlflow

    from ..orchestration.tracking import resolve_tracking_uri

    base, _, query = tracking_uri.partition("?")
    uri = resolve_tracking_uri(base) + (f"?{query}" if query else "")
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
        saved = json.loads(marker.read_text())
        if saved["digest"] == digest and _run_exists(
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
    marker.write_text(json.dumps({"digest": digest, "run_id": s.run_id}))
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
    s.failed_gates = _gate_lines(t["reproduction_gate"])
    draw_figures(inputs, st, t)
    if o.complexity:
        budgets = sorted(set(inputs.runs["runs"]["budget"]))
        copy_fidelity(o, o.out / "figures", budgets)
    else:  # an earlier full run's copies would go on to the report as if current
        for stale in (o.out / "figures").glob(FIDELITY.format(budget="*") + ".*"):
            stale.unlink()
    write_report_files(inputs, t, s)
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
