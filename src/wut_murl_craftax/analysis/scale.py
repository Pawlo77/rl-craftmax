"""The 1M vs 1B angles: rank agreement, advantage over the control, scaling, precision.

Every table is a ``benchmark == "scale"`` frame over the tuned final scores of the two
budgets. Seeds are independent between budgets, so each (budget, method) stratum is
resampled on its own; 1B has 3 seeds, so its intervals are wide and the tables describe,
they do not test.
"""

import itertools
import sqlite3
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sps

from ..evaluation.stats import CI_LEVEL, stratified_bootstrap_ci, t_ci
from . import compare
from .load import base_control

SCALE = "scale"
"""Value of the ``benchmark`` column of the scale tables (their folder name)."""
POWER = 0.8
"""Power behind `precision_table`'s smallest resolvable difference."""
SEEDS_1B = 3
"""Seeds per method at 1B."""
_NAN = float("nan")
Frames = dict[str, pd.DataFrame]


def _finals(frames: Frames) -> dict[str, dict[str, np.ndarray]]:
    """Budget -> method -> tuned final scores of the runs of ``frames``."""
    runs = compare.tuned(frames["runs"])
    return {
        b: {str(m): g["final_score"].to_numpy(float) for m, g in r.groupby("method")}
        for b, r in compare.budgets(runs)
    }


def _strata(
    finals: dict[str, dict[str, np.ndarray]], methods: list[str]
) -> dict[str, list[float]]:
    return {f"{b}:{m}": finals[b][m].tolist() for b in ("1m", "1b") for m in methods}


def _ci(
    strata: dict[str, list[float]],
    statistic: Callable[[dict[str, np.ndarray]], float],
    n: int,
    seed: int,
) -> tuple[float, float, float]:
    ci = stratified_bootstrap_ci(strata, statistic, n, seed)
    return ci.estimate, ci.lo, ci.hi


def _shared(finals: dict[str, dict[str, np.ndarray]]) -> list[str]:
    return sorted(set(finals.get("1m", {})) & set(finals.get("1b", {})))


def rank_agreement(frames: Frames, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """One row: Spearman and Kendall between the 1M and 1B method means, with CIs."""
    finals = _finals(frames)
    methods = _shared(finals)

    def corr(kind: Callable[..., Any], d: dict[str, np.ndarray]) -> float:
        a = [d[f"1m:{m}"].mean() for m in methods]
        b = [d[f"1b:{m}"].mean() for m in methods]
        return float(kind(a, b).statistic)

    strata = _strata(finals, methods)
    s = _ci(strata, lambda d: corr(sps.spearmanr, d), n_boot, seed)
    k = _ci(strata, lambda d: corr(sps.kendalltau, d), n_boot, seed)
    return pd.DataFrame([{
        "benchmark": SCALE,
        "methods": len(methods),
        "spearman": s[0], "spearman_lo": s[1], "spearman_hi": s[2],
        "kendall": k[0], "kendall_lo": k[1], "kendall_hi": k[2],
    }])  # fmt: skip


def _mean_diff(a: str, b: str) -> Callable[[dict[str, np.ndarray]], float]:
    return lambda d: float(d[a].mean() - d[b].mean())


def flipped_pairs(frames: Frames, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Method pairs whose mean order differs between 1M and 1B, with each difference."""
    finals = _finals(frames)
    rows = []
    for a, b in itertools.combinations(_shared(finals), 2):
        diff = {}
        for budget in ("1m", "1b"):
            strata = {k: finals[budget][k].tolist() for k in (a, b)}
            diff[budget] = _ci(strata, _mean_diff(a, b), n_boot, seed)
        if np.sign(diff["1m"][0]) != np.sign(diff["1b"][0]):
            rows.append({
                "benchmark": SCALE, "a": a, "b": b,
                **{f"diff_{k}": v[0] for k, v in diff.items()},
                **{f"diff_{k}_lo": v[1] for k, v in diff.items()},
                **{f"diff_{k}_hi": v[2] for k, v in diff.items()},
                "resolved": all(v[1] > 0 or v[2] < 0 for v in diff.values()),
            })  # fmt: skip
    return pd.DataFrame(rows, columns=FLIPPED_COLUMNS)


FLIPPED_COLUMNS = [
    "benchmark", "a", "b", "diff_1m", "diff_1b", "diff_1m_lo", "diff_1m_hi",
    "diff_1b_lo", "diff_1b_hi", "resolved",
]  # fmt: skip
"""`flipped_pairs`: a - b at each budget and its CI; ``resolved``: both exclude 0."""


def advantage_table(frames: Frames, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Gain over the base control in percent at each budget, and its change.

    ``change`` is the 1B gain minus the 1M gain (the interaction contrast); a CI that
    holds 0 leaves "the 1M advantage survives at 1B" open.
    """
    finals = _finals(frames)
    first = compare.tuned(frames["runs"]).drop_duplicates("method").set_index("method")
    rows = []
    for method in _shared(finals):
        base = base_control({**first.loc[method].to_dict(), "method": method})
        if base == method or base not in finals["1m"] or base not in finals["1b"]:
            continue
        strata = _strata(finals, [method, base])

        def gain(
            d: dict[str, np.ndarray], b: str, method: str = method, base: str = base
        ) -> float:
            return 100 * (d[f"{b}:{method}"].mean() / d[f"{b}:{base}"].mean() - 1)

        g1 = _ci(strata, lambda d: gain(d, "1m"), n_boot, seed)
        gb = _ci(strata, lambda d: gain(d, "1b"), n_boot, seed)
        ch = _ci(strata, lambda d: gain(d, "1b") - gain(d, "1m"), n_boot, seed)
        rows.append({
            "benchmark": SCALE, "method": method, "base": base,
            "gain_1m": g1[0], "gain_1m_lo": g1[1], "gain_1m_hi": g1[2],
            "gain_1b": gb[0], "gain_1b_lo": gb[1], "gain_1b_hi": gb[2],
            "change": ch[0], "change_lo": ch[1], "change_hi": ch[2],
        })  # fmt: skip
    return pd.DataFrame(rows)


def _log_ratio(m: str) -> Callable[[dict[str, np.ndarray]], float]:
    return lambda d: float(np.log(d[f"1b:{m}"].mean() / d[f"1m:{m}"].mean()))


def scaling_table(frames: Frames, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Per method, the log of the 1B to 1M mean ratio with a bootstrap CI."""
    finals = _finals(frames)
    rows = []
    for m in _shared(finals):
        est, lo, hi = _ci(
            _strata(finals, [m]),
            _log_ratio(m),
            n_boot,
            seed,
        )
        rows.append({
            "benchmark": SCALE, "method": m,
            "mean_1m": finals["1m"][m].mean(), "mean_1b": finals["1b"][m].mean(),
            "log_ratio": est, "log_ratio_lo": lo, "log_ratio_hi": hi,
        })  # fmt: skip
    return pd.DataFrame(rows)


def precision_table(frames: Frames) -> pd.DataFrame:
    """CI half-widths at both budgets and the smallest difference 3 seeds resolve.

    ``mde_3_seeds`` is the difference of two 3-seed means that a two-sided Welch test
    at `CI_LEVEL` resolves with power `POWER`, taking the 1M seed sd as the noise.
    """
    finals = _finals(frames)
    z = sps.norm.ppf(0.5 + CI_LEVEL / 2) + sps.norm.ppf(POWER)
    rows = []
    for m in _shared(finals):
        half = {
            b: (lambda i: (i[2] - i[1]) / 2)(t_ci(finals[b][m].tolist()))
            for b in finals
        }
        sd = float(finals["1m"][m].std(ddof=1))
        rows.append({
            "benchmark": SCALE, "method": m, "sd_1m": sd,
            "halfwidth_1m": half["1m"], "halfwidth_1b": half["1b"],
            "mde_3_seeds": float(z * sd * np.sqrt(2 / SEEDS_1B)),
        })  # fmt: skip
    return pd.DataFrame(rows)


def compute_table(frames: Frames) -> pd.DataFrame:
    """Mean final score against wall-clock hours per benchmark and profile.

    ``pareto`` marks the methods no other method of the same benchmark and profile
    beats in both hours and score; a method without timing is never on the front.
    Laptop timings are context only (`compare.CONTEXT_NOTE`).
    """
    hw = compare.hardware_table(dict(frames)).drop_duplicates([
        "benchmark",
        "profile",
        "method",
    ])
    finals = _finals(frames)
    t = pd.DataFrame({
        "benchmark": hw["benchmark"],
        "profile": hw["profile"],
        "method": hw["method"],
        "hours": hw["wall_clock_s"] / 3600,
        "score": [
            finals[b][m].mean()
            for b, m in zip(hw["benchmark"], hw["method"], strict=True)
        ],
        "note": hw["note"],
    }).reset_index(drop=True)
    t["pareto"] = False
    for _, g in t.groupby(["benchmark", "profile"]):
        for i, row in g.iterrows():
            beaten = ((g["hours"] <= row["hours"]) & (g["score"] >= row["score"])) & (
                (g["hours"] < row["hours"]) | (g["score"] > row["score"])
            )
            t.loc[i, "pareto"] = bool(row["hours"] == row["hours"] and not beaten.any())
    return t


def transfer_table(advantage: pd.DataFrame, optimism: pd.DataFrame) -> pd.DataFrame:
    """Each method's 1M tuning optimism beside the change of its advantage at 1B."""
    opt = optimism.drop_duplicates("method")[
        ["method", "tuning_mean", "fresh_mean", "optimism"]
    ]
    t = advantage[["method", "base", "change", "change_lo", "change_hi"]].merge(
        opt, on="method", how="left"
    )
    t.insert(0, "benchmark", SCALE)
    return t


def node_scores(root: Path) -> dict[str, list[float]]:
    """Method -> final scores of the runs in the ``mlruns-node-*.db`` files of ``root``.

    The method is the run name up to its ``-<budget>-`` part; a node store holds one
    run per array task, so the stores are read directly, never merged into the main
    store (their run names equal the laptop runs').
    """
    out: dict[str, list[float]] = {}
    for path in sorted(root.glob("mlruns-node-*.db")):
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        rows = con.execute(
            "select r.name, m.value from runs r join latest_metrics m"
            " on m.run_uuid = r.run_uuid where m.key = 'final_score'"
            " and r.status = 'FINISHED'"
        ).fetchall()
        con.close()
        for name, value in rows:
            out.setdefault(name.split("-1m-")[0], []).append(float(value))
    return out


def bridge_table(
    laptop: Mapping[str, list[float]],
    cluster: Mapping[str, list[float]],
    n_boot: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Final scores of the same controls on the laptop and on the cluster stack.

    ``diff`` is cluster minus laptop with a bootstrap CI over both seed sets;
    ``diff_sd`` is it in units of the pooled seed sd and ``shifted`` marks a CI that
    excludes 0. A shift inside the seed noise lets the 1M vs 1B comparison stand.
    """
    rows = []
    for m in sorted(set(laptop) & set(cluster)):
        a, b = np.asarray(laptop[m], float), np.asarray(cluster[m], float)
        est, lo, hi = _ci(
            {"laptop": a.tolist(), "cluster": b.tolist()},
            lambda d: float(d["cluster"].mean() - d["laptop"].mean()),
            n_boot,
            seed,
        )
        sd = float(np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2))
        rows.append({
            "benchmark": SCALE, "method": m, "seeds_laptop": a.size,
            "seeds_cluster": b.size, "mean_laptop": a.mean(), "mean_cluster": b.mean(),
            "diff": est, "diff_lo": lo, "diff_hi": hi, "diff_sd": est / sd,
            "poi": float(np.mean(b[:, None] > a[None, :])),
            "shifted": bool(lo > 0 or hi < 0),
        })  # fmt: skip
    return pd.DataFrame(rows)
