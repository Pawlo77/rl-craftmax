"""Tests for the literature table: the CSV against its source note, block mapping."""

import re

import numpy as np
import pandas as pd
import pytest
import yaml

from wut_murl_craftax.analysis.literature import (
    COMPARABLE_BLOCKS,
    LITERATURE_COLUMNS,
    LITERATURE_CSV,
    NOTES_DIR,
    comparable_rows,
    load_literature,
)

NOTE = NOTES_DIR / "Existing Implementations.md"
BLOCKS = ("A", "A2", "D")


def _note_rows() -> dict[str, list[list[str]]]:
    """The table rows of blocks A, A2 and D of the note, as cell lists."""
    rows: dict[str, list[list[str]]] = {}
    block = None
    for line in NOTE.read_text().splitlines():
        head = re.match(r"\*\*(A2|A|D)\. ", line)
        if head or line.startswith("**"):
            block = head.group(1) if head else None
            if block:
                rows[block] = []
        elif (
            block
            and line.startswith("|")
            and not line.startswith(("|---", "| Implementation"))
        ):
            rows[block].append([
                c.strip() for c in line.strip().strip("|").split(" | ")
            ])
    return rows


def test_csv_has_the_contract_columns_in_order():
    assert list(load_literature().columns) == list(LITERATURE_COLUMNS)
    assert LITERATURE_COLUMNS[:11] == (
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
    )


def test_one_row_per_note_row_with_the_cells_as_written():
    """Blocks A, A2, D: same count, order and cells as the note."""
    lit = load_literature()
    rows = _note_rows()
    assert set(rows) == set(BLOCKS)
    for block, cells in rows.items():
        got = lit[lit["block"] == block]
        assert len(got) == len(cells) > 0
        for (impl, env, obs, budget, score, seeds, *_), (_, r) in zip(
            cells, got.iterrows(), strict=True
        ):
            assert (
                r.implementation,
                r.env,
                r.observation,
                r.budget,
                r.seeds,
                r.score_text,
            ) == (
                impl,
                env,
                obs,
                budget,
                seeds,
                score,
            )
    assert set(lit["block"]) == set(BLOCKS)


def test_every_note_names_an_existing_file():
    lit = load_literature()
    assert lit["note"].map(lambda n: (NOTES_DIR / f"{n}.md").is_file()).all()
    # the linked note is the one the implementation cell links to
    assert [
        f"[[{n}]]" in i for n, i in zip(lit["note"], lit["implementation"], strict=True)
    ] == [True] * len(lit)


def test_score_and_spread_fields_are_consistent():
    lit = load_literature()
    assert lit["score_pct"].notna().all()
    assert lit["score_pct"].between(0, 100).all()
    assert set(lit["spread_kind"]) <= {"sd", "s.e.", "none"}
    assert lit["spread"].isna().tolist() == (lit["spread_kind"] == "none").tolist()


@pytest.mark.parametrize(
    ("block", "note", "score", "spread", "kind"),
    [
        ("A2", "PPO-LSTM (Jiralerspong 2026)", 19.54, 0.34, "s.e."),
        ("D", "Simulus (Cohen 2025)", 6.59, 0.49, "sd"),
        ("A", "PPO-RNN (Matthews 2024)", 15.3, np.nan, "none"),
        # the sd belongs to h040, not to the headline h096 value
        ("A", "Auto-Craftax PPO-GRU (Creus Castanyer 2026)", 17.9, np.nan, "none"),
        # a raw-return sd is not a spread of the % score, and a CI is no sd or s.e.
        ("A2", "Feasibility Classification (Zabounidis 2026)", 19.4, np.nan, "none"),
        ("A2", "Sable Inference Search (Chalumeau 2025)", 18.8, np.nan, "none"),
    ],
)
def test_first_row_of_a_note_in_a_block(block, note, score, spread, kind):
    lit = load_literature()
    row = lit[(lit["block"] == block) & (lit["note"] == note)].iloc[0]
    assert row.score_pct == score
    assert (np.isnan(row.spread) and np.isnan(spread)) or row.spread == spread
    assert row.spread_kind == kind


def test_controls_agree_with_the_reproduction_references():
    """The reproduction gate's published values are the note's rows (drift guard)."""
    ref_path = LITERATURE_CSV.parents[1] / "reference" / "published_scores.yaml"
    ref = yaml.safe_load(ref_path.read_text())
    lit = load_literature()
    rows = {
        ("ppo", "1b"): ("A", "PPO (Matthews 2024)", "1B"),
        ("ppo-rnn", "1b"): ("A", "PPO-RNN (Matthews 2024)", "1B"),
        ("pqn-rnn", "1b"): ("A", "PQN-RNN (Gallici 2025)", "1B"),
        ("ppo", "1m"): ("D", "PPO (Matthews 2024)", "1M"),
        ("ppo-rnn", "1m"): ("D", "PPO-RNN (Matthews 2024)", "1M"),
    }
    for (method, budget), (block, note, lit_budget) in rows.items():
        sel = lit[
            (lit["block"] == block)
            & (lit["note"] == note)
            & (lit["budget"] == lit_budget)
        ]
        assert len(sel) == 1
        assert sel["score_pct"].iloc[0] == ref[method][budget]["score"]


@pytest.mark.parametrize(
    ("score_kind", "budget", "block"),
    [("training", "1b", "A"), ("training", "1m", "D"), ("stochastic", "1b", "A2")],
)
def test_comparable_rows_maps_kind_and_budget_to_one_block(score_kind, budget, block):
    got = comparable_rows(score_kind, budget)
    full = load_literature()
    assert set(got["block"]) == {block}
    assert len(got) == (full["block"] == block).sum()
    assert COMPARABLE_BLOCKS[(score_kind, budget)] == block


def test_comparable_rows_returns_a_copy():
    got = comparable_rows("training", "1b")
    got.loc[:, "score_pct"] = -1.0
    assert (comparable_rows("training", "1b")["score_pct"] > 0).all()


@pytest.mark.parametrize(
    ("score_kind", "budget"),
    [
        ("greedy", "1b"),  # no published greedy evaluation is comparable
        ("greedy", "1m"),
        ("stochastic", "1m"),  # block A2 exists at 1B only
        ("training", "1B"),  # budgets are the lower-case export values
        ("training", "500m"),
        ("eval", "1b"),
    ],
)
def test_comparable_rows_raises_for_any_other_combination(score_kind, budget):
    with pytest.raises(ValueError, match=re.escape(f"({score_kind!r}, {budget!r})")):
        comparable_rows(score_kind, budget)


def test_load_literature_reads_a_given_path(tmp_path):
    path = tmp_path / "lit.csv"
    pd.read_csv(LITERATURE_CSV, dtype=str, keep_default_na=False).head(2).to_csv(
        path, index=False
    )
    assert len(load_literature(path)) == 2


def test_load_literature_rejects_a_changed_column_set(tmp_path):
    path = tmp_path / "lit.csv"
    pd.read_csv(LITERATURE_CSV, dtype=str, keep_default_na=False).drop(
        columns="seeds"
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="seeds"):
        load_literature(path)


def test_load_literature_rejects_an_unknown_spread_kind(tmp_path):
    path = tmp_path / "lit.csv"
    table = pd.read_csv(LITERATURE_CSV, dtype=str, keep_default_na=False)
    table.loc[0, "spread_kind"] = "ci"
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="ci"):
        load_literature(path)


def test_package_reexports_the_public_names():
    import wut_murl_craftax.analysis as analysis

    assert analysis.comparable_rows is comparable_rows
    assert analysis.load_literature is load_literature
    assert analysis.COMPARABLE_BLOCKS is COMPARABLE_BLOCKS


def _parse_score_text(text: str) -> tuple[float, float, str]:
    """Headline score, spread and spread kind of a score cell, by the CSV's rule."""
    score = float(re.match(r"\d+(?:\.\d+)?", text).group())
    stated = re.match(r"\d+(?:\.\d+)? ± (\d+(?:\.\d+)?) (sd|s\.e\.)", text)
    if stated:
        return score, float(stated.group(1)), stated.group(2)
    stated = re.match(r"\d+(?:\.\d+)? \(sd (\d+(?:\.\d+)?)[,;)]", text)
    return (score, float(stated.group(1)), "sd") if stated else (score, np.nan, "none")


def test_score_pct_spread_and_kind_follow_score_text_on_every_row():
    for _, row in load_literature().iterrows():
        score, spread, kind = _parse_score_text(row.score_text)
        assert row.score_pct == score, row.score_text
        assert row.spread_kind == kind, row.score_text
        assert (np.isnan(row.spread) and np.isnan(spread)) or row.spread == spread
    assert load_literature()["spread"].notna().sum() == 11


def test_protocol_is_the_block_s_unless_the_note_says_it_is_unstated():
    """Rule: `n.r.` where the row's source note says the protocol is unstated."""
    lit = load_literature()
    unstated = {
        ("A", "PPO-GTrXL (Hamon 2024)"),
        ("A", "PQN-RNN (Gallici 2025)"),
        ("A2", "SimBa (Lee 2025)"),
        ("D", "EASimulus (Peng 2026)"),
    }
    got = {(r.block, r.note) for r in lit.itertuples() if r.protocol == "n.r."}
    assert got == unstated
    block_protocol = {
        "A": "training-episode return",
        "A2": "separate evaluation of the trained policy",
    }
    for r in lit.itertuples():
        if r.protocol == "n.r.":
            continue
        if r.block in block_protocol:
            assert r.protocol == block_protocol[r.block]
        else:  # block D states its protocol per row
            assert r.protocol in {
                "training-episode return",
                "mean of 100 test episodes at the end of training",
            }
