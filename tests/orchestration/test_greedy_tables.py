"""Tests for the eval-only rebuild of SMDP greedy tables in a store (ruling R81)."""

import json
from pathlib import Path

import mlflow
import numpy as np
import pyarrow.parquet as pq
import pytest

from wut_murl_craftax.orchestration import greedy_tables
from wut_murl_craftax.orchestration.runner import TAG_WARN_ARTIFACTS
from wut_murl_craftax.orchestration.tracking import artifact_path
from wut_murl_craftax.training import smdp
from wut_murl_craftax.training.checkpoints import mark_pruned, save_checkpoint
from wut_murl_craftax.training.skills import NUM_SKILLS, SKILLS

SCHEMA_WARNING = (
    "update 244: ArrowInvalid: Schema at index 1 was different: \nupdate: int32\n"
    "key: int32\nfloor: int8\ngreedy_skill: null\nvalue: fl"
)
"""The start of the tag the 20ec0807 runs carry (the runner keeps 500 characters)."""


def _run(client: mlflow.MlflowClient, exp: str, name: str, **tags: str) -> str:
    """An smdp-skills run whose checkpoints 0 and 5 store 0 and 2 keys."""
    run = client.create_run(
        exp, run_name=name, tags={"method": "smdp-skills", **tags}
    ).info
    out = artifact_path(run.artifact_uri)
    out.mkdir(parents=True)
    (out / "config.json").write_text(json.dumps({"method": "smdp-skills"}))
    for u, keys in ((0, []), (5, [3, 9])):
        rows = np.full(1024, smdp.SENTINEL_ROW, np.int32)
        rows[: len(keys)] = keys
        q = np.full((1024, NUM_SKILLS), 10.0, np.float32)
        q[: len(keys), 1] = 11.0
        fields = np.zeros((1024, 16), np.int8)
        save_checkpoint(out, u, u, 0.0, {"rows": rows, "q_rows": q, "fields": fields})
    return run.run_id


@pytest.fixture
def store(tmp_path: Path) -> tuple[str, mlflow.MlflowClient, dict[str, str]]:
    uri = f"sqlite:///{tmp_path / 'mlruns.db'}"
    client = mlflow.MlflowClient(uri)
    exp = client.create_experiment(
        "final-1m", artifact_location=(tmp_path / "mlruns").as_uri()
    )
    ids = {
        "schema": _run(
            client,
            exp,
            "a",
            config_hash="20ec0807",
            **{TAG_WARN_ARTIFACTS: SCHEMA_WARNING},
        ),
        "other": _run(
            client,
            exp,
            "b",
            config_hash="20ec0807",
            **{TAG_WARN_ARTIFACTS: "update 3: OSError: disk full"},
        ),
        "untouched": _run(client, exp, "c", config_hash="aaaaaaaa"),
    }
    return uri, client, ids


def test_the_cli_rebuilds_the_runs_of_a_config_and_clears_only_schema_warnings(
    store: tuple[str, mlflow.MlflowClient, dict[str, str]],
) -> None:
    uri, client, ids = store
    greedy_tables.main(["--tracking-uri", uri, "--config-hash", "20ec0807"])
    for key in ("schema", "other"):
        out = artifact_path(client.get_run(ids[key]).info.artifact_uri)
        table = pq.read_table(out / "eval" / smdp.GREEDY_TABLE_FILE)
        assert table["update"].to_pylist() == [5, 5]
        assert table["greedy_skill"].to_pylist() == [SKILLS[1].name] * 2
    tags = {k: client.get_run(i).data.tags for k, i in ids.items()}
    assert TAG_WARN_ARTIFACTS not in tags["schema"]
    assert tags["other"][TAG_WARN_ARTIFACTS] == "update 3: OSError: disk full"
    untouched = artifact_path(client.get_run(ids["untouched"]).info.artifact_uri)
    assert not (untouched / "eval").exists()


def test_the_cli_refuses_a_config_of_another_method(
    store: tuple[str, mlflow.MlflowClient, dict[str, str]],
) -> None:
    uri, client, ids = store
    client.set_tag(ids["other"], "method", "dyna-q")
    with pytest.raises(ValueError, match="not smdp-skills"):
        greedy_tables.main(["--tracking-uri", uri, "--config-hash", "20ec0807"])
    out = artifact_path(client.get_run(ids["schema"]).info.artifact_uri)
    assert not (out / "eval").exists()  # checked before any write


def test_a_rebuild_keeps_the_rows_of_pruned_checkpoints_and_never_reads_them(
    store: tuple[str, mlflow.MlflowClient, dict[str, str]],
) -> None:
    """Storage retention: a pruned checkpoint is known-absent, never a crash."""
    _, client, ids = store
    out = artifact_path(client.get_run(ids["untouched"]).info.artifact_uri)
    path = smdp.rebuild_greedy_table(out)  # update 0 stores no keys, update 5 two
    before = pq.read_table(path)
    mark_pruned(out, [5], "test")
    (out / "ckpt" / "00005.msgpack").unlink()
    assert pq.read_table(smdp.rebuild_greedy_table(out)).equals(before)
    mark_pruned(out, [0], "test")
    (out / "ckpt" / "00000.msgpack").unlink()
    assert pq.read_table(smdp.rebuild_greedy_table(out)).equals(before)


def test_a_rebuild_with_every_checkpoint_pruned_and_no_table_fails_clearly(
    store: tuple[str, mlflow.MlflowClient, dict[str, str]],
) -> None:
    """Review M7: nothing to read and nothing to keep is a clear error."""
    _, client, ids = store
    out = artifact_path(client.get_run(ids["untouched"]).info.artifact_uri)
    mark_pruned(out, [0, 5], "test")
    for u in (0, 5):
        (out / "ckpt" / f"{u:05d}.msgpack").unlink()
    with pytest.raises(FileNotFoundError, match="every checkpoint is pruned"):
        smdp.rebuild_greedy_table(out)
