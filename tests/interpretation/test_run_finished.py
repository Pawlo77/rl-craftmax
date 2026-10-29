"""When a policy folder counts as finished: its summary, DAgger and the seed counts."""

import json
from pathlib import Path

import pandas as pd
import pytest

from wut_murl_craftax.interpretation import run as cli

FULL = {"train": range(256), "val": range(64), "test": range(64)}
SMOKE = {k: v[:2] for k, v in FULL.items()}


def write(folder: Path, **extra: object) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / cli.SUMMARY).write_text(json.dumps({"fits": {"base": {}}, **extra}))
    return folder


def counts(seeds: dict[str, range]) -> dict[str, int]:
    return {k: len(v) for k, v in seeds.items()}


def test_no_summary_is_unfinished(tmp_path: Path) -> None:
    assert not cli._finished(tmp_path, False, FULL)


def test_a_summary_of_the_same_seeds_is_finished(tmp_path: Path) -> None:
    folder = write(tmp_path, split_seeds=counts(FULL))
    assert cli._finished(folder, False, FULL)
    assert not cli._stale(folder, FULL)


def test_a_smoke_summary_is_stale_for_a_full_run(tmp_path: Path) -> None:
    folder = write(tmp_path, split_seeds=counts(SMOKE))
    assert not cli._finished(folder, False, FULL)
    assert cli._stale(folder, FULL)
    assert not cli._larger(folder, FULL)
    assert cli._finished(folder, False, SMOKE)


def test_a_full_summary_survives_a_smoke_call(tmp_path: Path) -> None:
    folder = write(tmp_path, split_seeds=counts(FULL))
    assert not cli._stale(folder, SMOKE)  # never emptied
    assert cli._finished(folder, False, SMOKE)  # skipped
    assert cli._larger(folder, SMOKE)


def test_a_smoke_call_leaves_a_full_policy_folder_untouched(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    folder = write(tmp_path / "m" / "r1", split_seeds=counts(FULL))
    (folder / "train.npz").write_bytes(b"full")
    rows = pd.DataFrame({"run_id": ["r1"], "config_ref": ["tuned"], "seed": [0]})
    summary = cli.DistilSummary()
    with caplog.at_level("WARNING"):
        redone = cli._distil_method(
            rows, tmp_path / "m", summary, tracking_uri="unused", dagger=False,
            workers=1, seeds=SMOKE, force=False,
        )  # fmt: skip
    assert not redone
    assert summary.skipped == ["r1"]
    assert (folder / "train.npz").read_bytes() == b"full"
    assert "more seeds than requested" in caplog.text


def test_a_summary_from_before_the_seed_counts_stays_finished(tmp_path: Path) -> None:
    folder = write(tmp_path)  # the 1M production files
    assert cli._finished(folder, False, FULL)
    assert not cli._stale(folder, FULL)


@pytest.mark.parametrize("dagger", [False, True])
def test_dagger_still_needs_its_fit(tmp_path: Path, dagger: bool) -> None:
    folder = write(tmp_path, split_seeds=counts(FULL))
    assert cli._finished(folder, dagger, FULL) is (not dagger)
