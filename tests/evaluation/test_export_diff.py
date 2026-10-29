"""Tests for the comparison of two exports of a phase."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wut_murl_craftax.evaluation.export_diff import (
    REPORT_COLUMNS,
    diff_phase,
    diff_table,
    main,
    unexpected,
)

KEY = {"run_id": "r", "update": 1, "mode": "stochastic"}


def _episodes() -> pd.DataFrame:
    return pd.DataFrame({
        **{k: [v] * 3 for k, v in KEY.items()},
        "eval_seed": [10, 11, 12],
        "length": [100, 200, 300],
        "kills/zombie": [1, 2, 3],
        "damage_dealt": [1.5, 2.5, np.nan],
        "stalls": [[{"a": 1}], [], [{"a": 2}]],
    })


def _rows(report: pd.DataFrame) -> dict[tuple[str, str], tuple[int, float | None]]:
    """(column, change) -> (rows, max_abs); None for NaN, which never equals itself."""
    return {
        (r.column, r.change): (r.rows, None if np.isnan(r.max_abs) else r.max_abs)
        for r in report.itertuples()
    }


def test_an_identical_pair_has_no_difference() -> None:
    old = _episodes()
    assert diff_table("p", "eval_episodes", old, old.copy()) == []


def test_changed_added_removed_and_missing_rows_are_listed() -> None:
    old = _episodes()
    new = old.copy()
    new.loc[1, "kills/zombie"] = 5  # one row, 3 more
    new.loc[0, "damage_dealt"] = 1.0  # 0.5 less; the NaN equals itself
    new["any_zero_share"] = 0.1
    new = new.drop(columns="length")
    new = new.iloc[:2].copy()  # the seed 12 row goes
    got = pd.DataFrame(
        diff_table("p", "eval_episodes", old, new), columns=REPORT_COLUMNS
    )
    assert _rows(got) == {
        ("any_zero_share", "added"): (2, None),
        ("length", "removed"): (3, None),
        ("(rows)", "only in old"): (1, None),
        ("damage_dealt", "changed"): (1, 0.5),
        ("kills/zombie", "changed"): (1, 3.0),
    }


def test_a_json_column_compares_by_value() -> None:
    old = _episodes()
    new = old.copy()
    new.at[0, "stalls"] = [{"a": 9}]
    got = pd.DataFrame(
        diff_table("p", "eval_episodes", old, new), columns=REPORT_COLUMNS
    )
    assert _rows(got) == {("stalls", "changed"): (1, None)}


def test_unexpected_keeps_what_the_redo_may_not_change() -> None:
    old = _episodes()
    new = old.copy()
    new["any_zero_share"] = 0.1
    new.loc[0, "kills/zombie"] = 9
    new.loc[1, "damage_dealt"] = 0.0
    new.loc[2, "length"] = 301
    got = pd.DataFrame(
        diff_table("p", "eval_episodes", old, new), columns=REPORT_COLUMNS
    )
    assert unexpected(got)["column"].tolist() == ["length"]
    assert unexpected(pd.DataFrame(columns=REPORT_COLUMNS)).empty
    rows = pd.DataFrame(
        diff_table("p", "eval_episodes", old, old.iloc[:2]), columns=REPORT_COLUMNS
    )
    assert unexpected(rows)["column"].tolist() == ["(rows)"]


def _write(root: Path, phase: str, eps: pd.DataFrame, runs: pd.DataFrame) -> None:
    (root / phase).mkdir(parents=True)
    eps.to_parquet(root / phase / "eval_episodes.parquet")
    runs.to_parquet(root / phase / "runs.parquet")


def test_the_cli_reports_and_exits_1_on_an_unexpected_change(tmp_path: Path) -> None:
    runs = pd.DataFrame({"run_id": ["r"], "steady_sps": [1.0]})
    eps = _episodes()
    good = eps.copy()
    good["any_zero_share"] = 0.2
    _write(tmp_path / "old", "p", eps, runs)
    _write(tmp_path / "ok", "p", good, runs)
    _write(tmp_path / "bad", "p", good, runs.assign(steady_sps=2.0))
    argv = ["--phase", "p", "--old", str(tmp_path / "old")]
    main([*argv, "--new", str(tmp_path / "ok"), "--out", str(tmp_path / "r.csv")])
    assert pd.read_csv(tmp_path / "r.csv")["column"].tolist() == ["any_zero_share"]
    with pytest.raises(SystemExit) as e:
        main([*argv, "--new", str(tmp_path / "bad")])
    assert e.value.code == 1
    assert diff_phase(tmp_path / "old", tmp_path / "old", "p").empty


def test_a_table_in_one_root_only_is_an_error(tmp_path: Path) -> None:
    runs = pd.DataFrame({"run_id": ["r"], "steady_sps": [1.0]})
    _write(tmp_path / "old", "p", _episodes(), runs)
    _write(tmp_path / "new", "p", _episodes(), runs)
    (tmp_path / "new" / "p" / "runs.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="runs"):
        diff_phase(tmp_path / "old", tmp_path / "new", "p")
