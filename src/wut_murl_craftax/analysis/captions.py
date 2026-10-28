"""Figure and table notes of one benchmark, built from the captions the stages wrote.

``python -m wut_murl_craftax.analysis.captions --budget 1m`` reads the
``captions.csv`` files (``stem,id,title,description``) of ``outputs/results/figures/
<budget>/``, ``outputs/results/tables/<budget>/`` and the training plots of every method
(``outputs/analysis/<budget>/<method>/plots/``) and writes ``knowledge/results/Figures
<BUDGET>.md`` and ``Tables <BUDGET>.md``: per figure group a table of figure link and
description, and a table of table file links and descriptions. The images carry no
sentences; this is where the report's captions are taken from.
@see [[2026-10-20-craftax-comparison-and-results]]
@tested-by [[tests/analysis/test_captions.py]]
"""

import argparse
import csv
import logging
import re
from collections.abc import Callable, Sequence
from pathlib import Path

from .load import write_text_atomic

logger = logging.getLogger(__name__)

FAMILIES = {
    "LEG": "Method legend",
    "F1": "Learning curves", "F2": "Aggregate scores", "F3": "PoI matrix",
    "F4": "Seeds and published scores", "F5": "Achievements reached",
    "F6": "First-unlock race", "F7": "Order agreement", "F7G": "Order graphs",
    "F8": "Funnel", "F8T": "Funnel in training", "F9": "Deaths", "F9S": "Stalls",
    "F10": "Speedrun splits", "F11": "Exploration", "F12": "Survival", "F13": "Combat",
    "F14": "Actions", "F15": "Resources", "F16": "Advanced mechanics", "F17": "Radars",
    "F18": "Tuning", "F19": "Stability", "F20": "Hardware", "F21": "Greedy gap",
    "F22": "Early unlocks", "F23": "Seed fate", "F24": "Ablation effects",
    "F25": "Tuning optimism", "F26": "Episode spread", "F27": "Distillation and score",
    "F28": "Distillation inputs", "FID": "Distillation fidelity",
}  # fmt: skip
"""Figure id (the stage that drew it) -> its rows' heading."""
INTERPRETATION = ("F27", "F28", "FID")
"""Figure ids of the interpretation (decision-tree distillation) group."""
TABLE_GROUPS = {
    "T1": "Main results", "T2": "Literature", "T3": "Stability", "T4": "Hardware",
    "T5": "Stability and hardware", "T6": "In-game", "T7": "Tuning",
}  # fmt: skip
"""Table id -> the heading of its rows."""
UP = "../.."
"""From ``knowledge/results/`` to the repository root."""


def read_captions(path: Path) -> list[dict[str, str]]:
    """The rows of a ``captions.csv`` (``[]`` if it does not exist)."""
    if not path.is_file():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _natural(key: str) -> tuple[int, list[object]]:
    """Sort key that orders ``F2`` before ``F10`` and ``LEG`` before both."""
    parts: list[object] = [
        int(t) if t.isdigit() else t for t in re.split(r"(\d+)", key)
    ]
    return (key != "LEG", parts)


def _cell(row: dict[str, str]) -> str:
    return f"**{row['title']}.** {row['description']}"


def _ref(repo: Path, rel: str, label: str) -> str:
    """A link to the repository file ``rel`` if it exists, else its path in code font.

    The vault check rejects links to files that are not there (``outputs/`` is not
    tracked), so a note built before the files exist names them without a link.
    """
    if (repo / rel).is_file():
        return f"[{label}]({UP}/{rel})"
    return f"`{rel}`"


def _figure_path(stem: str, budget: str, repo: Path) -> str:
    """The report PNG of ``stem`` (the double-width one when it has no column file)."""
    folder = f"report/figures/{budget}"
    double = (repo / folder / f"{stem}_double.png").is_file()
    column = (repo / folder / f"{stem}.png").is_file()
    return (
        f"{folder}/{stem}_double.png"
        if double and not column
        else f"{folder}/{stem}.png"
    )


def _frontmatter(title: str, tags: Sequence[str], today: str) -> str:
    code = ["captions.py", "results.py"]
    lines = [
        "---",
        f'title: "{title}"',
        "type: concept",
        "domain: analysis",
        'parent: "[[Final results]]"',
        f"tags: [{', '.join(tags)}]",
        f"created: {today}",
        f"updated: {today}",
        "related-code:",
        *(f'  - "[[src/wut_murl_craftax/analysis/{c}]]"' for c in code),
        "---",
    ]
    return "\n".join(lines)


def _table(heads: tuple[str, str], rows: Sequence[tuple[str, str]]) -> list[str]:
    out = [f"| {heads[0]} | {heads[1]} |", "|---|---|"]
    out += [f"| {a} | {b} |" for a, b in rows]
    return out


def figures_note(
    budget: str,
    results: Path,
    analysis: Path,
    repo: Path,
    today: str,
) -> str:
    """The Markdown of ``Figures <BUDGET>.md``: figure link and description tables."""
    name = budget.upper()
    rows = read_captions(results / "figures" / budget / "captions.csv")
    comparison = [r for r in rows if r["id"] not in INTERPRETATION]
    interpretation = [r for r in rows if r["id"] in INTERPRETATION]
    lines = [
        _frontmatter(
            f"Figures {name}",
            ["concept", "results", "figures", f"craftax-{budget}"],
            today,
        ),
        f"# Figures {name}",
        "",
        (
            "Every figure is one panel in one PNG without text beyond axes, ticks and "
            "a legend where one is needed; this note holds what the report prints "
            "around them. Colours, line styles and markers of the methods: "
            "`legend_methods`. Generated by `make figure-notes` from the "
            "`captions.csv` of each folder."
        ),
    ]

    def section(
        heading: str, group: list[dict[str, str]], target: Callable[[str], str]
    ) -> None:
        if not group:
            return
        lines.extend(["", f"## {heading}"])
        for fid in sorted({r["id"] for r in group}, key=_natural):
            mine = sorted((r for r in group if r["id"] == fid), key=lambda r: r["stem"])
            lines.extend(["", f"### {FAMILIES.get(fid, fid)}", ""])
            lines.extend(
                _table(
                    ("Figure", "Description"),
                    [(target(r["stem"]), _cell(r)) for r in mine],
                )
            )

    def shown(stem: str) -> str:
        return _ref(repo, _figure_path(stem, budget, repo), stem)

    section("Comparison", comparison, shown)
    section("Interpretation", interpretation, shown)
    methods = sorted(p.name for p in (analysis / budget).glob("*") if p.is_dir())
    trained = [
        (m, read_captions(analysis / budget / m / "plots" / "captions.csv"))
        for m in methods
    ]
    if any(rows for _, rows in trained):
        lines.extend(["", "## Training plots"])
        for method, mine in trained:
            if not mine:
                continue
            lines.extend(["", f"### {method}", ""])
            plots = f"outputs/analysis/{budget}/{method}/plots"
            lines += _table(
                ("Figure", "Description"),
                [
                    (_ref(repo, f"{plots}/{r['stem']}.png", r["stem"]), _cell(r))
                    for r in sorted(mine, key=lambda r: r["stem"])
                ],
            )
    return "\n".join(lines) + "\n"


def tables_note(budget: str, results: Path, repo: Path, today: str) -> str:
    """The Markdown of ``Tables <BUDGET>.md``: table file links and descriptions."""
    name = budget.upper()
    rows = read_captions(results / "tables" / budget / "captions.csv")
    lines = [
        _frontmatter(
            f"Tables {name}",
            ["concept", "results", "tables", f"craftax-{budget}"],
            today,
        ),
        f"# Tables {name}",
        "",
        (
            "Each file holds the rows of this benchmark only; the CSV is in "
            f"`outputs/results/tables/{budget}/`, the LaTeX table in "
            f"`report/tables/{budget}/`. Generated by `make figure-notes` from the "
            "`captions.csv` of the folder."
        ),
    ]
    for tid in sorted({r["id"] for r in rows}, key=_natural):
        lines.extend(["", f"## {TABLE_GROUPS.get(tid, tid)}", ""])
        body = []
        for r in sorted((r for r in rows if r["id"] == tid), key=lambda r: r["stem"]):
            files = []
            if (results / "tables" / budget / f"{r['stem']}.csv").is_file():
                rel = f"outputs/results/tables/{budget}/{r['stem']}.csv"
                files.append(_ref(repo, rel, f"{r['stem']}.csv"))
            if (results / "tables" / budget / f"{r['stem']}.tex").is_file():
                rel = f"report/tables/{budget}/{r['stem']}.tex"
                files.append(_ref(repo, rel, f"{r['stem']}.tex"))
            body.append((" ".join(files) or r["stem"], _cell(r)))
        lines += _table(("Table", "Description"), body)
    return "\n".join(lines) + "\n"


def write_notes(
    budget: str,
    results: Path,
    analysis: Path,
    repo: Path,
    out_dir: Path,
    today: str,
) -> list[Path]:
    """Write ``Figures <BUDGET>.md`` and ``Tables <BUDGET>.md`` into ``out_dir``."""
    name = budget.upper()
    notes = {
        f"Figures {name}.md": figures_note(budget, results, analysis, repo, today),
        f"Tables {name}.md": tables_note(budget, results, repo, today),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for file, text in notes.items():
        write_text_atomic(out_dir / file, text)
        paths.append(out_dir / file)
    return paths


def main(argv: Sequence[str] | None = None) -> None:
    """Command line: the notes of one benchmark from the finished stages' captions."""
    import datetime

    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("--budget", default="1m")
    p.add_argument("--results", type=Path, default=Path("outputs/results"))
    p.add_argument("--analysis", type=Path, default=Path("outputs/analysis"))
    p.add_argument("--repo", type=Path, default=Path(), help="where links resolve")
    p.add_argument("--out-dir", type=Path, default=Path("knowledge/results"))
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    today = datetime.date.today().isoformat()
    for path in write_notes(a.budget, a.results, a.analysis, a.repo, a.out_dir, today):
        logger.info("wrote %s", path)


if __name__ == "__main__":
    main()
