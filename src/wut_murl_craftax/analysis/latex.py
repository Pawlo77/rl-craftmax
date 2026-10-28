r"""LaTeX export of the report: booktabs tables, number macros and report figures.

`write_tables` renders T1, T2, T5 and T6 of `compare` as booktabs ``tabular`` (best
value per column bold within a benchmark and block), `write_numbers` one ``\newcommand``
per number named ``<quantity><Budget><Method>[<Ref>][<Control>]`` in letters only
(contracts.md section 7), `copy_figures` the report PDFs. A budget without runs (no
``final-1b`` yet, R75, R86) emits nothing; the 1M ``complexity.csv`` is required (the
error names interpretation Task 6), the 1B one is skipped when missing.
@see [[2026-10-20-craftax-comparison-and-results]]
@tested-by [[tests/analysis/test_latex.py]]
"""

import logging
import math
import re
import shutil
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd

from ..evaluation.stats import BOOTSTRAP_N
from . import compare
from .load import CONTROLS, Frames

logger = logging.getLogger(__name__)

REPORT_DIR = Path("report")
TABLES_DIR = REPORT_DIR / "tables"
FIGURES_DIR = REPORT_DIR / "figures"
NUMBERS_TEX = REPORT_DIR / "numbers.tex"
REPORT_TEX = REPORT_DIR / "report.tex"
INTERPRETATION = Path("outputs/interpretation")
"""``<budget>/complexity.csv`` of interpretation Task 6 (contracts.md section 3)."""
MACRO_NAMES = {
    "ppo": "Ppo",
    "ppo-rnn": "PpoRnn",
    "pqn-rnn": "PqnRnn",
    "count-bonus-ppo-rnn": "CountBonusPpoRnn",
    "count-bonus-pqn-rnn": "CountBonusPqnRnn",
    "sil": "Sil",
    "smdp-skills": "SmdpSkills",
    "dyna-q": "DynaQ",
    "combo": "Combo",
}
"""Registry name -> the hyphen-free CamelCase of a macro name."""
BUDGET_NAMES = {"1m": "OneM", "1b": "OneB"}
REF_NAMES = {
    "tuned": "",
    "simhash": "SimHash",
    "single-buffer": "SingleBuffer",
    "random-top": "RandomTop",
    "monte-carlo": "MonteCarlo",
    "no-flags": "NoFlags",
    "kappa0": "KappaZero",
}
"""``config_ref`` -> its macro word (``tuned`` is omitted)."""
VARIANT_NAMES = {"base": "", "no-memory": "NoMemory", "dagger": "Dagger"}
COMPLEXITY_QUANTITIES = {
    "l80_median": "lEighty",
    "l90_median": "lNinety",
    "l95_median": "lNinetyFive",
    "depth_at_l90_median": "depthNinety",
    "fidelity_16": "fidSixteen",
    "fidelity_64": "fidSixtyFour",
    "fidelity_256": "fidTwoFiftySix",
    "expected_fidelity_256": "expectedFidTwoFiftySix",
    "closed_loop_256": "closedLoop",
}
"""``complexity.csv`` column -> quantity word (numbers spelled out, no digits)."""
SPEARMAN = "spearmanScoreNinety"
"""One cross-method value per (budget, variant); the macro has no method part."""
NEVER = r"\ensuremath{>}4096"
"""A leaf count that was never reached (``inf`` in ``complexity.csv``)."""
NOT_APPLICABLE = "--"
MACRO = re.compile(r"^[A-Za-z]+$")
USED = re.compile(
    r"\\((?:score|scoreSe|iqm|iqmLo|iqmHi|median|evalStoch|evalGreedy|poi|pHolm|"
    r"lEighty|lNinety|lNinetyFive|depthNinety|fid[A-Za-z]*|expectedFid[A-Za-z]*|"
    r"closedLoop|spearman[A-Za-z]*)(?:OneM|OneB)[A-Za-z]*)"
)
"""A control sequence of this module's macro family, as used in a ``.tex`` file."""
_ESCAPES = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}  # fmt: skip


def escape(text: object) -> str:
    """``text`` with LaTeX's special characters escaped (method names, notes)."""
    return "".join(_ESCAPES.get(c, c) for c in str(text))


# ---------------------------------------------------------------------- numbers


def _finite(v: object) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(v)


def _fmt(v: float, digits: int = 2) -> str:
    """``v`` with a math minus (a hyphen in text mode is too short)."""
    return f"{v:.{digits}f}".replace("-", "$-$")


def _p(v: float) -> str:
    """A p-value; ``$<$`` because ``<`` is an inverted "!" in OT1 text fonts."""
    return "$<$0.001" if v < 0.001 else f"{v:.3f}"


def _interval(lo: float, hi: float, digits: int = 2) -> str:
    return f"[{_fmt(lo, digits)}, {_fmt(hi, digits)}]"


def macro(
    quantity: str, budget: str, method: str, ref: str = "tuned", tail: str = ""
) -> str:
    """``<quantity><Budget><Method>[<Ref>]<tail>``, checked to be letters only.

    Raises:
        ValueError: If the budget, method or ref has no macro word, or the name is
            not letters only.
    """
    for kind, table, key in (
        ("budget", BUDGET_NAMES, budget),
        ("method", MACRO_NAMES, method),
        ("config_ref", REF_NAMES, ref),
    ):
        if key not in table:
            msg = f"no macro word for {kind} {key!r}: add it to latex.{kind.upper()}"
            raise ValueError(msg)
    name = (
        f"{quantity}{BUDGET_NAMES[budget]}{MACRO_NAMES[method]}{REF_NAMES[ref]}{tail}"
    )
    if not MACRO.match(name):
        msg = f"macro name {name!r} is not letters only"
        raise ValueError(msg)
    return name


class Number(NamedTuple):
    """One reported number: macro name and the text it expands to."""

    name: str
    text: str


def _unit_numbers(
    row: Mapping[str, Any], budget: str, method: str, ref: str
) -> list[Number]:
    """Score, interval and eval macros of one T1 or ablation row (finite values)."""
    parts = (
        ("score", "mean", _fmt), ("scoreSe", "se", _fmt), ("iqm", "iqm", _fmt),
        ("iqmLo", "iqm_lo", _fmt), ("iqmHi", "iqm_hi", _fmt),
        ("median", "median", _fmt), ("evalStoch", "eval_stochastic", _fmt),
        ("evalGreedy", "eval_greedy", _fmt),
    )  # fmt: skip
    return [
        Number(macro(q, budget, method, ref), fmt(row[col]))
        for q, col, fmt in parts
        if col in row and _finite(row[col])
    ]


def _versus_numbers(row: Mapping[str, Any], budget: str, method: str) -> list[Number]:
    """PoI and Holm p of one T1 row against each control (1M, finite values)."""
    out = []
    for control in CONTROLS:
        for q, col, fmt in (
            ("poi", f"poi_{control}", _fmt),
            ("pHolm", f"p_holm_{control}", _p),
        ):
            if col in row and _finite(row[col]):
                out.append(
                    Number(
                        macro(q, budget, method, tail=MACRO_NAMES[control]),
                        fmt(row[col]),
                    )
                )
    return out


def _leaves(v: float) -> str:
    if math.isinf(v):
        return NEVER
    return f"{v:.0f}" if v == int(v) else f"{v:.1f}"


def _variant(variant: object) -> str:
    """The macro word of a ``complexity.csv`` variant.

    Raises:
        ValueError: Naming a variant the macros have no word for.
    """
    try:
        return VARIANT_NAMES[str(variant)]
    except KeyError:
        msg = f"complexity.csv variant {variant!r} has no macro word in VARIANT_NAMES"
        raise ValueError(msg) from None


def complexity_numbers(frame: pd.DataFrame) -> list[Number]:
    """Macros of a ``complexity.csv``: leaves to fidelity 80/90/95%, fidelities, ratios.

    Names follow `macro` with ``NoMemory`` or ``Dagger`` appended for those variants;
    a leaf count of ``inf`` reads "> 4096"; the Spearman value is one per budget and
    variant.
    """
    out = []
    for r in frame.to_dict("records"):
        tail = _variant(r["variant"])
        for col, quantity in COMPLEXITY_QUANTITIES.items():
            v = r.get(col)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            whole = col.startswith(("l80", "l90", "l95", "depth"))
            text = _leaves(float(v)) if whole else _fmt(float(v), 3)
            out.append(
                Number(
                    macro(quantity, r["budget"], r["method"], r["config_ref"], tail),
                    text,
                )
            )
    for (budget, variant), g in frame.groupby(["budget", "variant"]):
        v = g["spearman_score_l90"].dropna()
        if len(v):
            name = f"{SPEARMAN}{BUDGET_NAMES[str(budget)]}{_variant(variant)}"
            out.append(Number(name, _fmt(float(v.iloc[0]), 3)))
    return out


def _read_complexity(root: Path, budgets: Sequence[str]) -> pd.DataFrame:
    """The ``complexity.csv`` of each budget with runs: 1M required, 1B optional."""
    parts = []
    for budget in budgets:
        path = root / budget / "complexity.csv"
        if path.exists():
            parts.append(pd.read_csv(path))
        elif budget == "1m":
            msg = (
                f"{path} is missing: run interpretation Task 6 (the distillation of "
                "the 1M finals) before the LaTeX numbers"
            )
            raise FileNotFoundError(msg)
        else:
            logger.warning("%s missing: no %s complexity macros (R86)", path, budget)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def collect_numbers(
    frames: Frames,
    *,
    tables: Mapping[str, pd.DataFrame] | None = None,
    complexity_root: Path = INTERPRETATION,
    complexity: bool = True,
    n_boot: int = BOOTSTRAP_N,
    seed: int = 0,
) -> list[Number]:
    """Every reported number as a `Number`, sorted by name.

    ``tables`` may carry precomputed ``main`` (T1) and ``ablation`` tables. With
    ``complexity=False`` the ``complexity.csv`` macros are left out (a partial run
    before the distillation exists) and the file is never read.

    Raises:
        FileNotFoundError: Without the 1M ``complexity.csv`` while 1M runs exist.
        ValueError: If one macro name would take two different values.
    """
    tables = tables or {}
    t1 = tables.get("main")
    t1 = compare.main_table(frames, n_boot=n_boot, seed=seed) if t1 is None else t1
    ab = tables.get("ablation")
    ab = compare.ablation_table(frames, n_boot=n_boot, seed=seed) if ab is None else ab
    found: list[Number] = []
    for r in t1.to_dict("records"):
        found += _unit_numbers(r, r["benchmark"], r["method"], compare.TUNED)
        found += _versus_numbers(r, r["benchmark"], r["method"])
    for r in ab.to_dict("records"):
        found += _unit_numbers(r, r["benchmark"], r["method"], r["config_ref"])
    budgets = sorted(set(frames["runs"]["budget"]), key=lambda b: "mb".find(b[-1]))
    if complexity:
        found += complexity_numbers(_read_complexity(complexity_root, budgets))
    merged: dict[str, str] = {}
    for n in found:
        if merged.setdefault(n.name, n.text) != n.text:  # a control repeats per block
            msg = (
                f"macro {n.name} would take two values: {merged[n.name]!r}, {n.text!r}"
            )
            raise ValueError(msg)
    return [Number(k, merged[k]) for k in sorted(merged)]


def write_numbers(
    path: Path = NUMBERS_TEX,
    frames: Frames | None = None,
    *,
    tables: Mapping[str, pd.DataFrame] | None = None,
    complexity_root: Path = INTERPRETATION,
    complexity: bool = True,
    n_boot: int = BOOTSTRAP_N,
    seed: int = 0,
    report: Path | None = REPORT_TEX,
) -> list[Number]:
    r"""Write ``numbers.tex``: one ``\newcommand`` per number; return the numbers.

    With an existing ``report`` the file is checked afterwards: every macro of this
    family that the report uses must be defined (`undefined_macros`). ``complexity``
    as in `collect_numbers`; pass ``report=None`` with ``complexity=False``, since the
    report may use the macros left out.

    Raises:
        ValueError: Without ``frames``, as `collect_numbers`, or when ``report`` uses
            a macro the written file does not define (the file is still written).
    """
    if frames is None:
        msg = "write_numbers needs the phase frames (analysis.load.load_phases)"
        raise ValueError(msg)
    numbers = collect_numbers(
        frames, tables=tables, complexity_root=complexity_root, complexity=complexity,
        n_boot=n_boot, seed=seed,
    )  # fmt: skip
    lines = ["% Generated by wut_murl_craftax.analysis.latex; do not edit."]
    lines += [f"\\newcommand{{\\{n.name}}}{{{n.text}}}" for n in numbers]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    if report is not None and report.exists():
        missing = undefined_macros(report, path)
        if missing:
            msg = f"{report} uses macros that {path} does not define: {missing}"
            raise ValueError(msg)
    return numbers


def defined_macros(numbers_tex: str) -> set[str]:
    r"""Names defined by ``\newcommand`` in the text of a ``numbers.tex``."""
    return set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", numbers_tex))


def undefined_macros(
    report: Path = REPORT_TEX, numbers: Path = NUMBERS_TEX
) -> list[str]:
    """Macros of this module's family that ``report`` uses and ``numbers`` lacks.

    ``report.tex`` uses none yet, so this is empty today (R75, R86).
    """
    used = set(USED.findall(report.read_text()))
    known = defined_macros(numbers.read_text()) if numbers.exists() else set()
    return sorted(used - known)


# ----------------------------------------------------------------------- tables


class Column(NamedTuple):
    """One table column: header, cell text, and which value is best (bold)."""

    header: str
    cell: Callable[[Mapping[str, Any]], str]
    best: Callable[[Mapping[str, Any]], float] | None = None
    higher: bool = True


def _best_mask(rows: Sequence[Mapping[str, Any]], col: Column) -> list[bool]:
    """Rows whose value of ``col`` is the best of ``rows`` (ties all bold)."""
    if col.best is None:
        return [False] * len(rows)
    values = [col.best(r) for r in rows]
    ok = [v for v in values if _finite(v)]
    if not ok:
        return [False] * len(rows)
    top = max(ok) if col.higher else min(ok)
    return [_finite(v) and v == top for v in values]


def tabular(
    columns: Sequence[Column],
    groups: Sequence[tuple[str, Sequence[Mapping[str, Any]]]],
) -> str:
    r"""A booktabs ``tabular``: one heading row per group, the best cell bold.

    ``groups`` are (heading, rows); the best value of each column is bold within a
    group (a method is never ranked against the other block). The first column is
    left-aligned, the rest right-aligned.
    """
    align = "l" + "r" * (len(columns) - 1)
    out = [f"\\begin{{tabular}}{{{align}}}", "\\toprule"]
    out.append(" & ".join(c.header for c in columns) + r" \\")
    for heading, rows in groups:
        out.append("\\midrule")
        if heading:
            span = f"\\multicolumn{{{len(columns)}}}{{l}}"
            out.append(f"{span}{{\\emph{{{escape(heading)}}}}} \\\\")
        masks = [_best_mask(rows, c) for c in columns]
        for i, row in enumerate(rows):
            cells = [
                f"\\textbf{{{c.cell(row)}}}" if masks[j][i] else c.cell(row)
                for j, c in enumerate(columns)
            ]
            out.append(" & ".join(cells) + r" \\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


def _num_cell(key: str, digits: int = 2) -> Callable[[Mapping[str, Any]], str]:
    def cell(r: Mapping[str, Any]) -> str:
        return _fmt(r[key], digits) if _finite(r.get(key)) else NOT_APPLICABLE

    return cell


def _pm_cell(r: Mapping[str, Any]) -> str:
    if not _finite(r.get("mean")):
        return NOT_APPLICABLE
    return f"{_fmt(r['mean'])} $\\pm$ {_fmt(r['se'])}"


def _ci_cell(est: str, lo: str, hi: str) -> Callable[[Mapping[str, Any]], str]:
    def cell(r: Mapping[str, Any]) -> str:
        if not _finite(r.get(est)):
            return NOT_APPLICABLE
        return f"{_fmt(r[est])} {_interval(r[lo], r[hi])}"

    return cell


def _name_cell(r: Mapping[str, Any]) -> str:
    return escape(r["method"])


def _group_rows(
    table: pd.DataFrame, budget: str
) -> list[tuple[str, list[dict[str, Any]]]]:
    """(block heading, rows) of ``budget`` in table order (controls on top of both)."""
    out = []
    for block in compare.BLOCKS:
        rows = table[(table["benchmark"] == budget) & (table["block"] == block)]
        if len(rows):
            out.append((f"{_title(block)} block", rows.to_dict("records")))
    return out


def _title(block: str) -> str:
    return {"main": "Main", "skills": "Uses hand-written skills"}[block]


def main_tex(t1: pd.DataFrame, budget: str) -> str:
    """T1 of one benchmark: mean +- s.e., IQM (1M) or median (1B) with 95% CIs, evals.

    A 1B table is marked indicative, with its seed count, in its first heading.
    """
    significant = budget in compare.SIGNIFICANCE_BUDGETS
    centre = (
        Column(
            "IQM [95\\% CI]", _ci_cell("iqm", "iqm_lo", "iqm_hi"), lambda r: r["iqm"]
        )
        if significant
        else Column(
            "median [95\\% CI]",
            _ci_cell("median", "median_lo", "median_hi"),
            lambda r: r["median"],
        )
    )
    columns = [
        Column("Method", _name_cell),
        Column("n", lambda r: str(int(r["seeds"]))),
        Column("mean $\\pm$ s.e.", _pm_cell, lambda r: r["mean"]),
        centre,
        Column("eval", _num_cell("eval_stochastic"), lambda r: r["eval_stochastic"]),
        Column("greedy", _num_cell("eval_greedy"), lambda r: r["eval_greedy"]),
    ]
    groups = _group_rows(t1, budget)
    if not significant and groups:
        seeds = sorted({int(r["seeds"]) for _, rows in groups for r in rows})
        count = "/".join(map(str, seeds))
        groups[0] = (
            f"Craftax-1B, indicative ({count} seeds): {groups[0][0]}",
            groups[0][1],
        )
    return tabular(columns, groups)


def _versus_columns(c: str) -> list[Column]:
    """PoI [CI] and Holm p of every method against control ``c``."""

    def p_cell(r: Mapping[str, Any]) -> str:
        return _p(r[f"p_holm_{c}"]) if _finite(r[f"p_holm_{c}"]) else NOT_APPLICABLE

    def poi(r: Mapping[str, Any]) -> float:
        return float(r[f"poi_{c}"])

    def p_value(r: Mapping[str, Any]) -> float:
        return float(r[f"p_holm_{c}"])

    return [
        Column(f"PoI vs {c}", _ci_cell(f"poi_{c}", f"poi_lo_{c}", f"poi_hi_{c}"), poi),
        Column(f"p Holm {c}", p_cell, p_value, higher=False),
    ]


def versus_tex(t1: pd.DataFrame, budget: str = "1m") -> str:
    """T1's comparison with the controls: PoI [95% CI] and Holm p (1M only)."""
    columns = [Column("Method", _name_cell)]
    for c in CONTROLS:
        columns += _versus_columns(c)
    rows = t1[t1["role"] == "method"]
    groups = [
        (
            f"{_title(b)} block",
            rows[(rows["benchmark"] == budget) & (rows["block"] == b)].to_dict(
                "records"
            ),
        )
        for b in compare.BLOCKS
    ]
    return tabular(columns, [(h, r) for h, r in groups if r])


def literature_tex(t2: pd.DataFrame) -> str:
    """T2: our rows beside the published ones, with our CI and the gate per control.

    The gate cell is the reproduction-gate status of a control row, with the gap (our
    mean minus the published value) and the band half-width h when they exist.
    """

    def score(r: Mapping[str, Any]) -> str:
        if not _finite(r.get("score_pct")):
            return NOT_APPLICABLE
        text = _fmt(r["score_pct"])
        if _finite(r.get("ci_lo")) and _finite(r.get("ci_hi")):
            return f"{text} {_interval(r['ci_lo'], r['ci_hi'])}"
        if r.get("spread_kind") not in (None, "none", "") and _finite(r.get("spread")):
            text += f" $\\pm$ {_fmt(r['spread'])} ({escape(r['spread_kind'])})"
        return text

    def seeds(r: Mapping[str, Any]) -> str:
        return str(int(r["seeds"])) if _finite(r.get("seeds")) else NOT_APPLICABLE

    def gate(r: Mapping[str, Any]) -> str:
        status = r.get("gate")
        if not isinstance(status, str) or not status:
            return NOT_APPLICABLE
        text = escape(status)
        if _finite(r.get("gap")) and _finite(r.get("h")):
            text += f" (gap {_fmt(r['gap'])}, h {_fmt(r['h'])})"
        return text

    columns = [
        Column("Name", lambda r: escape(r["name"])),
        Column("Protocol", lambda r: escape(r["protocol"])),
        Column("n", seeds),
        Column("score (\\% of max)", score, lambda r: r["score_pct"]),
        Column("gate", gate),
    ]
    groups = []
    for (kind, budget), g in t2.groupby(["score_kind", "benchmark"], sort=False):
        groups.append((
            f"Craftax-{str(budget).upper()}, {kind} score",
            g.to_dict("records"),
        ))
    return tabular(columns, groups)


def stability_hardware(t3: pd.DataFrame, t4: pd.DataFrame) -> pd.DataFrame:
    """T5: T3's headline columns joined to T4 per (benchmark, method, profile)."""
    hw = t4.drop_duplicates(["benchmark", "profile", "method"])
    keep = [
        "benchmark",
        "method",
        "sd",
        "cv",
        "max_drawdown_median",
        "runs_with_drops",
        "runs_with_collapse",
    ]
    st = t3[keep].drop_duplicates(["benchmark", "method"])
    cols = [
        "benchmark",
        "profile",
        "method",
        "role",
        "steady_sps",
        "wall_clock_s",
        "compile_s",
        "peak_host_mem_mb",
        "note",
    ]
    return hw[cols].merge(st, on=["benchmark", "method"], how="left")


def stability_hardware_tex(t5: pd.DataFrame) -> str:
    """T5: seed sd, drawdown, drops, collapses, steady SPS, wall-clock, memory."""
    columns = [
        Column("Method", _name_cell),
        Column("sd", _num_cell("sd"), lambda r: r["sd"], higher=False),
        Column(
            "max drawdown",
            _num_cell("max_drawdown_median"),
            lambda r: r["max_drawdown_median"],
            higher=False,
        ),
        Column(
            "drops",
            lambda r: (
                str(int(r["runs_with_drops"]))
                if _finite(r["runs_with_drops"])
                else NOT_APPLICABLE
            ),
        ),
        Column(
            "collapses",
            lambda r: (
                str(int(r["runs_with_collapse"]))
                if _finite(r["runs_with_collapse"])
                else NOT_APPLICABLE
            ),
        ),
        Column("SPS", _num_cell("steady_sps", 0)),  # context only, never ranked (R97)
        Column(
            "wall (h)",
            lambda r: (
                _fmt(r["wall_clock_s"] / 3600)
                if _finite(r["wall_clock_s"])
                else NOT_APPLICABLE
            ),
        ),
        Column(
            "mem (GB)",
            lambda r: (
                _fmt(r["peak_host_mem_mb"] / 1024, 1)
                if _finite(r["peak_host_mem_mb"])
                else "n.m."
            ),
        ),
    ]
    groups = []
    for (b, p), g in t5.groupby(["benchmark", "profile"], sort=False):
        notes = sorted({str(n) for n in g["note"] if isinstance(n, str) and n})
        tail = f" ({'; '.join(notes)})" if notes else ""
        groups.append((f"Craftax-{str(b).upper()}, {p}{tail}", g.to_dict("records")))
    return tabular(columns, groups)


def game_summary_tex(t6: pd.DataFrame, budget: str) -> str:
    """T6: race wins, deepest floor, top death, coverage, wasted share, adoption."""

    def count(key: str) -> Callable[[Mapping[str, Any]], str]:
        return lambda r: str(int(r[key])) if _finite(r[key]) else "not logged"

    columns = [
        Column("Method", _name_cell),
        Column("race wins", count("race_wins_steps"), lambda r: r["race_wins_steps"]),
        Column(
            "deepest floor",
            lambda r: (
                f"{int(r['deepest_floor'])} ({_fmt(r['deepest_floor_share'])})"
                if _finite(r["deepest_floor"])
                else NOT_APPLICABLE
            ),
        ),
        Column(
            "top death",
            lambda r: (
                f"{escape(r['top_death_cause'])} ({_fmt(r['top_death_share'])})"
                if r["top_death_cause"]
                else NOT_APPLICABLE
            ),
        ),
        Column("coverage", _num_cell("coverage"), lambda r: r["coverage"]),
        Column(
            "wasted",
            _num_cell("wasted_share"),
            lambda r: r["wasted_share"],
            higher=False,
        ),
        Column("adoption", _num_cell("adoption"), lambda r: r["adoption"]),
    ]
    return tabular(columns, _group_rows(t6, budget))


def _table_texts(given: Mapping[str, pd.DataFrame]) -> dict[str, str]:
    """The ``.tex`` text of every table that has rows, by file stem."""
    texts: dict[str, str] = {}
    for budget in BUDGET_NAMES:
        if (given["main"]["benchmark"] == budget).any():
            texts[f"main_{budget}"] = main_tex(given["main"], budget)
        if (given["game"]["benchmark"] == budget).any():
            texts[f"game_summary_{budget}"] = game_summary_tex(given["game"], budget)
    if (given["main"]["benchmark"] == "1m").any():
        texts["main_vs_1m"] = versus_tex(given["main"])
    if len(given["literature"]):
        texts["literature"] = literature_tex(given["literature"])
    if len(given["hardware"]):
        texts["stability_hardware"] = stability_hardware_tex(
            stability_hardware(given["stability"], given["hardware"])
        )
    for plan, real in (("main", "main_1m"), ("game_summary", "game_summary_1m")):
        if real in texts:  # the plan's names are the 1M tables (1M files come first)
            texts[plan] = texts[real]
    return texts


def write_tables(
    frames: Frames | None = None,
    out_dir: Path = TABLES_DIR,
    *,
    tables: Mapping[str, pd.DataFrame] | None = None,
    n_boot: int = BOOTSTRAP_N,
    seed: int = 0,
) -> list[Path]:
    """Write the report tables (T1, T2, T5, T6) into ``out_dir``.

    Files: ``main_<budget>.tex`` and ``game_summary_<budget>.tex`` (also written as
    ``main.tex`` and ``game_summary.tex``, the plan's names, holding the 1M table),
    ``main_vs_1m.tex``, ``literature.tex`` and ``stability_hardware.tex``.

    ``tables`` takes precomputed ``main``, ``literature``, ``stability``,
    ``hardware`` and ``game`` frames (T1 to T4, T6); the others are computed from
    ``frames``. A budget without rows gets no file (1B before ``final-1b`` exists).

    Raises:
        ValueError: If a table is missing and ``frames`` is None.
    """
    given = dict(tables or {})
    builders: dict[str, Callable[[Frames], pd.DataFrame]] = {
        "main": lambda f: compare.main_table(f, n_boot=n_boot, seed=seed),
        "literature": lambda f: compare.literature_table(f, n_boot=n_boot, seed=seed),
        "stability": compare.stability_table,
        "hardware": compare.hardware_table,
        "game": compare.game_summary_table,
    }
    for key, build in builders.items():
        if key not in given:
            if frames is None:
                msg = f"write_tables needs frames or the {key!r} table"
                raise ValueError(msg)
            given[key] = build(frames)
    out_dir.mkdir(parents=True, exist_ok=True)
    texts = _table_texts(given)
    paths = []
    for stem, text in texts.items():
        path = out_dir / f"{stem}.tex"
        path.write_text(text)
        paths.append(path)
    return paths


# ---------------------------------------------------------------------- figures


def copy_figures(results_dir: Path, out_dir: Path = FIGURES_DIR) -> list[Path]:
    """Copy the report PDFs of ``results_dir`` (no ``_video`` file) to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for src in sorted(results_dir.glob("*.pdf")):
        if src.stem.endswith("_video"):
            continue
        shutil.copy2(src, out_dir / src.name)
        copied.append(out_dir / src.name)
    return copied
