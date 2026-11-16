"""Inference budget: what k runs per model buy, at a common cost.

The question a practitioner faces is not "is the fusion steadier than one
model?" but "for the seconds I can spend per review, should I repeat one model,
fuse several, or decode greedily once?". This module answers it on the
collected runs, without new inference: a budget of k runs per model is
emulated by every k-subset of the ten run indices, taken at once across all
reviews (k = 1 and 2 give 10 and 45 subsets, k = 5 gives 252; no sampling is
needed).

For each subset the collective ranking is recomputed (2-tuple operator,
two-level hierarchy) and scored by

    margin_top   the winner's beta lead over the runner-up;
    top1_crowd   the share of crowd replicas (experts drawn as clusters, runs
                 held at the subset) that keep the subset's own winner;
    winner_ok    whether the subset's winner is the system's ten-run winner.

The crowd replicas are the corpus's shared ones (`draw_crowd`), common to
every subset, system and table, so differences between budgets are not seed
noise and the ten-run row is the crowd Top-1 reported elsewhere.
Cost is given two ways. Calls per review is exact: k for one model, m x k for a
fusion of m members. Seconds per review is estimated from per-request medians:
the median request latency times k, summed over a fusion's members. For a
fusion, a k-subset is applied by rounds: the same run positions for every
member, not every combination of positions across members. Greedy (T = 0)
rows use the same latency as the sampled collection of that model: decoding
temperature does not change the request path, and no greedy latency was
measured separately.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from itertools import combinations

import numpy as np
import pandas as pd

from .cdm import scale_for
from .stability import (
    DEFAULT_PANEL_REPLICAS,
    DEFAULT_SEED,
    CrowdDraws,
    draw_crowd,
    keeps_winner,
    ranks_of,
)
from .twotuple import beta_from_y

BUDGETS = (1, 2, 3, 5, 10)
N_RUNS = 10


def _run_matrix(runs: pd.DataFrame, review_ids: list[str]) -> np.ndarray:
    """`(n_reviews, N_RUNS)` betas by run index; NaN where a run was rejected."""

    scale = scale_for(str(runs["scheme"].iloc[0]))
    out = np.full((len(review_ids), N_RUNS), np.nan)
    row = {r: i for i, r in enumerate(review_ids)}
    for review_id, run, y in runs[["review_id", "run", "y"]].itertuples(index=False):
        if review_id in row and 0 <= int(run) < N_RUNS:
            out[row[review_id], int(run)] = beta_from_y(y, scale)
    return out


def _subset_values(matrices: list[np.ndarray], subset: tuple[int, ...]) -> np.ndarray:
    """Review values of a (possibly fused) system restricted to `subset` runs.

    Each member averages the runs of the subset that parsed (the available-run
    policy); the fusion averages the members that have a value.
    """

    # A review with no parsed run in the subset is missing for that member;
    # NumPy warns about the empty mean, which is exactly the case intended.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        per_member = np.stack(
            [np.nanmean(m[:, list(subset)], axis=1) for m in matrices], axis=0
        )
        return np.nanmean(per_member, axis=0)


def budget_curve(
    systems: dict[str, pd.DataFrame],
    reference_winners: dict[str, str],
    latency_median: dict[str, float],
    members: dict[str, Sequence[str]],
    budgets: Sequence[int] = BUDGETS,
    replicas: int = DEFAULT_PANEL_REPLICAS,
    seed: int = DEFAULT_SEED,
    draws: CrowdDraws | None = None,
) -> pd.DataFrame:
    """One row per (system, k) with median and 5th-95th percentiles over subsets.

    `systems` maps a row label to the runs it is built from (one scheme; the
    review, expert and alternative columns attached). `members` maps the label
    to the models whose latencies it costs, and `reference_winners` to the
    winner its ten-run, T = 1 counterpart reports.
    """

    first = next(iter(systems.values()))
    reviews = first.drop_duplicates("review_id")
    reviews = reviews[["review_id", "user_id", "alternative"]]
    review_ids = list(reviews["review_id"])
    names = list(dict.fromkeys(reviews["alternative"]))
    index = {n: i for i, n in enumerate(names)}
    indicator = np.zeros((len(names), len(review_ids)))
    for j, alternative in enumerate(reviews["alternative"]):
        indicator[index[alternative], j] = 1.0
    # The corpus's shared crowd replicas, so that a ten-run row equals the
    # crowd Top-1 every other table reports for that system.
    if draws is None:
        draws = draw_crowd(reviews, replicas, seed)
    replicas = draws.replicas
    weights = draws.cluster_counts(
        list(zip(reviews["alternative"], reviews["user_id"], strict=True))
    )

    rows = []
    for label, frame in systems.items():
        models = list(frame["model"].drop_duplicates())
        matrices = [_run_matrix(frame[frame["model"] == m], review_ids) for m in models]
        per_request = sum(latency_median.get(m, np.nan) for m in members[label])
        for k in budgets:
            margins, top1s, agree = [], [], []
            for subset in combinations(range(N_RUNS), k):
                values = _subset_values(matrices, subset)
                present = np.isfinite(values)
                v = np.where(present, values, 0.0)
                full = (indicator @ v) / (indicator @ present)
                ranks = ranks_of(full)[0]
                order = np.argsort(ranks, kind="stable")
                margins.append(full[order[0]] - full[order[1]])
                # A subset tying the ten-run winner for first names it, as
                # `keeps_winner` scores a tie; frame order must not decide.
                reference = names.index(reference_winners[label])
                winner = reference if ranks[reference] == 1 else int(order[0])
                agree.append(winner == reference)

                w = weights * present
                with np.errstate(invalid="ignore", divide="ignore"):
                    replica = ((w * v) @ indicator.T) / (w @ indicator.T)
                replica = np.where(np.isnan(replica), -np.inf, replica)
                # One Top-1 definition throughout: see `keeps_winner`.
                kept = keeps_winner(ranks_of(replica), winner)
                top1s.append(100.0 * float(kept.mean()))
            margins_a, top1_a = np.array(margins), np.array(top1s)
            rows.append(
                {
                    "system": label,
                    "k": k,
                    "subsets": len(margins),
                    "calls_per_review": k * len(members[label]),
                    "cost_s_per_review": round(per_request * k, 3),
                    "top1_crowd": round(float(np.median(top1_a)), 1),
                    "top1_crowd_p5": round(float(np.percentile(top1_a, 5)), 1),
                    "top1_crowd_p95": round(float(np.percentile(top1_a, 95)), 1),
                    "margin_top": round(float(np.median(margins_a)), 5),
                    "margin_top_p5": round(float(np.percentile(margins_a, 5)), 5),
                    "margin_top_p95": round(float(np.percentile(margins_a, 95)), 5),
                    "winner_ok_pct": round(100.0 * float(np.mean(agree)), 1),
                    "replicas": replicas,
                }
            )
    return pd.DataFrame(rows)


def calls_comparison(budget: pd.DataFrame) -> pd.DataFrame:
    """At each number of calls per review: the best single model and the fusion.

    Best is the highest median crowd Top-1 among the sampled single-model rows
    with exactly that many calls; the fusion row is the one whose m x k equals
    it, if any. Greedy rows are kept apart, since one greedy call is not a
    point on the sampled curve.
    """

    sampled = budget[~budget["system"].str.contains("T=0", regex=False)]
    fused = sampled["system"].str.startswith("Fusion")
    rows = []
    for calls in sorted(set(sampled["calls_per_review"])):
        single = sampled[~fused & (sampled["calls_per_review"] == calls)]
        fusion = sampled[fused & (sampled["calls_per_review"] == calls)]
        row = {"calls_per_review": int(calls)}
        if not single.empty:
            best = single.sort_values("top1_crowd", ascending=False).iloc[0]
            row.update(
                single_system=best["system"],
                single_k=int(best["k"]),
                single_top1=best["top1_crowd"],
                single_top=best["margin_top"],
                single_same_winner=best["winner_ok_pct"],
                single_cost_s=best["cost_s_per_review"],
            )
        if not fusion.empty:
            f = fusion.iloc[0]
            row.update(
                fusion_k=int(f["k"]),
                fusion_top1=f["top1_crowd"],
                fusion_top=f["margin_top"],
                fusion_same_winner=f["winner_ok_pct"],
                fusion_cost_s=f["cost_s_per_review"],
            )
        rows.append(row)
    return pd.DataFrame(rows)
