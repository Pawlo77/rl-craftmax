"""Rebuild the SMDP greedy tables of stored runs from their checkpoints (ruling R81).

Eval-only repair: no training, no new eval rollouts. For every run of one config
hash, `smdp.rebuild_greedy_table` rewrites ``eval/smdp_greedy_table.parquet`` in
place; the run's `TAG_WARN_ARTIFACTS` goes only when it is the greedy table's Arrow
schema failure. CLI: ``python -m wut_murl_craftax.orchestration.greedy_tables
--tracking-uri URI --config-hash HASH8``.
"""

import argparse
import logging

import mlflow
import pyarrow.parquet as pq

from .queue import TAG_CONFIG_HASH, TAG_METHOD, _runs
from .runner import TAG_WARN_ARTIFACTS
from .tracking import artifact_path, resolve_tracking_uri

logger = logging.getLogger(__name__)

METHOD = "smdp-skills"
"""The registry entry whose ``config.json`` the rebuild reads."""


def schema_warning(text: str) -> bool:
    """Is a `TAG_WARN_ARTIFACTS` text the greedy table's null-vs-text schema failure?"""
    return "ArrowInvalid: Schema at index" in text and "greedy_skill" in text


def regenerate(client: mlflow.MlflowClient, config_hash: str) -> dict[str, int]:
    """Rebuild the greedy table of every run of ``config_hash``; run name -> rows.

    Raises:
        ValueError: If a run of the hash is not `METHOD` (checked before any write).
    """
    # imported here: smdp loads Craftax and JAX, which `orchestration` imports lazily
    from ..training.smdp import rebuild_greedy_table

    runs = _runs(client, f"tags.{TAG_CONFIG_HASH} = '{config_hash}'")
    if other := sorted(
        r.info.run_name for r in runs if r.data.tags.get(TAG_METHOD) != METHOD
    ):
        msg = f"runs of {config_hash} are not {METHOD}: {other}"
        raise ValueError(msg)
    rows: dict[str, int] = {}
    for run in sorted(runs, key=lambda r: r.info.run_name):
        path = rebuild_greedy_table(artifact_path(run.info.artifact_uri))
        rows[run.info.run_name] = n = pq.read_metadata(path).num_rows
        warning = run.data.tags.get(TAG_WARN_ARTIFACTS)
        if warning is not None and schema_warning(warning):
            client.delete_tag(run.info.run_id, TAG_WARN_ARTIFACTS)
        logger.info("%s: %d rows", run.info.run_name, n)
    return rows


def main(argv: list[str] | None = None) -> None:
    """Command line: ``--tracking-uri URI --config-hash HASH8``."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tracking-uri", required=True)
    p.add_argument("--config-hash", required=True)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    client = mlflow.MlflowClient(resolve_tracking_uri(a.tracking_uri))
    rows = regenerate(client, a.config_hash)
    logger.info("rebuilt %d runs of %s", len(rows), a.config_hash)


if __name__ == "__main__":
    main()
