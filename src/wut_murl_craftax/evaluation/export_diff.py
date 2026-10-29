"""Table-by-table comparison of two exports of a phase: what a re-export changed.

`diff_phase` reads ``<root>/<phase>/<table>.parquet`` of an old and a new root and
lists, per table, rows only in one export, added and removed columns, and for every
common column the rows that changed and the largest absolute change (numbers; NaN
equals NaN). `unexpected` keeps what the numfix redo may not change: anything but the
new ``any_zero_share`` column and the kill and damage columns of ``eval_episodes``.
CLI: ``python -m wut_murl_craftax.evaluation.export_diff --old outputs/metrics --new
outputs/metrics_v2 --phase final-1m``; exit status 1 if something unexpected changed.
@tested-by [[tests/evaluation/test_export_diff.py]]
"""

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

KEYS: dict[str, list[str]] = {
    "runs": ["run_id"],
    "curves": ["run_id", "env_steps", "metric"],
    "eval_episodes": ["run_id", "update", "mode", "eval_seed"],
    "eval_inventory": ["run_id", "update", "mode", "eval_seed"],
    "skill_ends": ["run_id", "update", "mode", "eval_seed", "start"],
}
"""Row key of each export table (a repeated key is numbered in file order)."""
REPORT_COLUMNS = ["phase", "table", "column", "change", "rows", "max_abs"]
EXPECTED = {
    "eval_episodes": ("any_zero_share", "kills/", "damage_dealt", "projectile_damage"),
}
"""Columns of a table that the redo may add or change (a name ending in ``/`` is a
family); ``kills/*`` and the damage columns move only where the mismatch CSVs say."""
ROWS = "(rows)"


def _plain(v: Any) -> str:
    return json.dumps(
        v, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)
    )


def _keyed(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["_n"] = df.groupby(keys).cumcount()
    return df.set_index([*keys, "_n"]).sort_index()


def diff_table(
    phase: str, table: str, old: pd.DataFrame, new: pd.DataFrame
) -> list[dict[str, Any]]:
    """The report rows (`REPORT_COLUMNS`) of one table pair."""
    base = {"phase": phase, "table": table}
    out: list[dict[str, Any]] = []

    def add(column: str, change: str, rows: int, max_abs: float = np.nan) -> None:
        out.append(base | {"column": column, "change": change, "rows": rows,
                           "max_abs": max_abs})  # fmt: skip

    a, b = _keyed(old, KEYS[table]), _keyed(new, KEYS[table])
    for name, cols in (("added", b.columns.difference(a.columns)),
                       ("removed", a.columns.difference(b.columns))):  # fmt: skip
        for c in cols:
            add(c, name, len(b if name == "added" else a))
    if n := len(a.index.difference(b.index)):
        add(ROWS, "only in old", n)
    if n := len(b.index.difference(a.index)):
        add(ROWS, "only in new", n)
    both = a.index.intersection(b.index)
    a, b = a.loc[both], b.loc[both]
    for c in a.columns.intersection(b.columns):
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            xa, ya = x.to_numpy(np.float64), y.to_numpy(np.float64)
            same = (xa == ya) | (np.isnan(xa) & np.isnan(ya))
            if not same.all():
                add(
                    c,
                    "changed",
                    int((~same).sum()),
                    float(np.nanmax(np.abs(xa - ya)[~same])),
                )
        else:
            x, y = x.map(_plain), y.map(_plain)
            if n := int((x != y).sum()):
                add(c, "changed", n)
    return out


def diff_phase(old_root: Path, new_root: Path, phase: str) -> pd.DataFrame:
    """`diff_table` of every export table both roots hold for ``phase``.

    Raises:
        FileNotFoundError: If a table exists in one root but not in the other.
    """
    rows: list[dict[str, Any]] = []
    for table in KEYS:
        paths = [r / phase / f"{table}.parquet" for r in (old_root, new_root)]
        if not any(p.is_file() for p in paths):
            continue
        if not all(p.is_file() for p in paths):
            msg = f"{table}: {paths[0]} and {paths[1]} must both exist"
            raise FileNotFoundError(msg)
        old, new = (pd.read_parquet(p) for p in paths)
        rows += diff_table(phase, table, old, new)
    return pd.DataFrame(rows, columns=REPORT_COLUMNS)


def unexpected(report: pd.DataFrame) -> pd.DataFrame:
    """The rows of ``report`` outside `EXPECTED` (row-set differences always count)."""

    def fine(r: pd.Series) -> bool:
        names = EXPECTED.get(r["table"], ())
        return r["column"] != ROWS and any(
            r["column"] == n or (n.endswith("/") and r["column"].startswith(n))
            for n in names
        )

    return report[~report.apply(fine, axis=1)] if len(report) else report


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--old", type=Path, required=True, help="root of the old export")
    p.add_argument("--new", type=Path, required=True, help="root of the new export")
    p.add_argument("--phase", action="append", required=True, help="repeatable")
    p.add_argument("--out", type=Path, help="also write the report as CSV")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Print the report; exit 1 if anything outside `EXPECTED` changed."""
    a = _parse(argv)
    report = pd.concat(
        [diff_phase(a.old, a.new, p) for p in a.phase], ignore_index=True
    )
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(a.out, index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(report.to_string(index=False) if len(report) else "no difference")
    bad = unexpected(report)
    if len(bad):
        print(f"\n{len(bad)} unexpected change(s):\n{bad.to_string(index=False)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
