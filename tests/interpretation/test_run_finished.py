"""When a policy folder counts as finished: its summary, DAgger and the seed counts."""

import json
from pathlib import Path

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


def test_a_smoke_summary_is_stale_for_a_full_run_and_back(tmp_path: Path) -> None:
    folder = write(tmp_path, split_seeds=counts(SMOKE))
    assert not cli._finished(folder, False, FULL)
    assert cli._stale(folder, FULL)
    assert cli._finished(folder, False, SMOKE)


def test_a_summary_from_before_the_seed_counts_stays_finished(tmp_path: Path) -> None:
    folder = write(tmp_path)  # the 1M production files
    assert cli._finished(folder, False, FULL)
    assert not cli._stale(folder, FULL)


@pytest.mark.parametrize("dagger", [False, True])
def test_dagger_still_needs_its_fit(tmp_path: Path, dagger: bool) -> None:
    folder = write(tmp_path, split_seeds=counts(FULL))
    assert cli._finished(folder, dagger, FULL) is (not dagger)
