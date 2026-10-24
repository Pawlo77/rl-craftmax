"""Published Craftax scores a result is compared with: the rows of blocks A, A2 and D.

``configs/literature/existing_implementations.csv`` copies those table rows of the note
[[Existing Implementations]] (cells as written; ``score_pct``, ``spread`` and
``spread_kind`` are read off the score cell, nothing is recomputed). Only like is
compared with like: training-episode score at 1B with block A, at 1M with block D,
stochastic fixed-seed evaluation at 1B with block A2. Every other block and score kind
raises, so a figure can never put our number beside an incomparable one.

@see [[2026-10-20-craftax-comparison-and-results]]
@tested-by [[tests/analysis/test_literature.py]]
"""

from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
LITERATURE_CSV = _REPO_ROOT / "configs" / "literature" / "existing_implementations.csv"
"""The copied rows of blocks A, A2 and D."""
NOTES_DIR = _REPO_ROOT / "research" / "Existing Implementations"
"""Where the source note and the per-paper notes of the ``note`` column live."""
LITERATURE_COLUMNS = (
    "block",
    "implementation",
    "note",
    "env",
    "observation",
    "budget",
    "protocol",
    "score_pct",
    "spread",
    "spread_kind",
    "seeds",
    "score_text",
)
"""``note`` is the linked note's file stem; ``score_text`` is the score cell as written
(it keeps the qualifiers `score_pct` leaves out, e.g. a raw-return sd or a CI)."""
SPREAD_KINDS = ("sd", "s.e.", "none")
"""``none``: no sd or s.e. of the % score is stated (raw sd or CI do not count)."""
COMPARABLE_BLOCKS = {
    ("training", "1b"): "A",
    ("training", "1m"): "D",
    ("stochastic", "1b"): "A2",
}
"""(score kind, export ``budget`` value) -> the like-for-like block."""


def load_literature(path: Path = LITERATURE_CSV) -> pd.DataFrame:
    """Read the table; ``score_pct`` and ``spread`` are float, the rest text."""
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(table.columns) != list(LITERATURE_COLUMNS):
        missing = sorted(set(LITERATURE_COLUMNS) - set(table.columns))
        extra = sorted(set(table.columns) - set(LITERATURE_COLUMNS))
        msg = (
            f"{path}: columns must be {list(LITERATURE_COLUMNS)}; "
            f"missing {missing}, unexpected {extra}"
        )
        raise ValueError(msg)
    bad = sorted(set(table["spread_kind"]) - set(SPREAD_KINDS))
    if bad:
        msg = f"{path}: spread_kind {bad} not in {list(SPREAD_KINDS)}"
        raise ValueError(msg)
    table["score_pct"] = table["score_pct"].astype(float)
    table["spread"] = pd.to_numeric(table["spread"].mask(table["spread"] == ""))
    return table


def comparable_rows(score_kind: str, budget: str) -> pd.DataFrame:
    """The literature rows our ``score_kind`` score at ``budget`` is compared with.

    ``score_kind`` is ``training`` or ``stochastic`` and ``budget`` is ``1m`` or ``1b``;
    any combination without a like-for-like block (greedy evaluation, stochastic at 1M)
    raises ValueError.
    """
    block = COMPARABLE_BLOCKS.get((score_kind, budget))
    if block is None:
        known = ", ".join(f"({k!r}, {b!r})" for k, b in COMPARABLE_BLOCKS)
        msg = (
            f"no comparable literature block for ({score_kind!r}, {budget!r}); "
            f"known: {known}"
        )
        raise ValueError(msg)
    table = load_literature()
    return table[table["block"] == block].reset_index(drop=True)
