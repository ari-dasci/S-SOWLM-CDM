"""Does the pipeline produce a stable *decision*, and does it still discriminate?

Stability of a label is not on its own evidence of anything: a system that
answers `positive` to every review is perfectly stable and perfectly useless.
Without ground truth the way to tell the two apart is to ask several questions
at once, which is what this module computes.

    decision_stability   Re-run the pipeline on resampled runs. Does the
                         ranking of the alternatives survive? Reported as
                         Kendall's tau against the full-data ranking, the share
                         of replicas reproducing it, and the beta margin
                         between adjacently ranked alternatives -- a ranking
                         held up by a margin of 0.01 is not stable, however
                         reproducible it looks.

    crowd_stability      The same question about a different source of
                         uncertainty. `decision_stability` holds the crowd
                         fixed and resamples the model's runs, so it measures
                         stability *conditional on these 474 reviews*.
                         `crowd_stability` does the opposite: it resamples
                         experts, carrying all of an expert's reviews together,
                         and leaves the runs alone. The two must be reported
                         separately, because a ranking can be robust to
                         inference noise and fragile to who happened to review.

    discriminance        Split the variance of the per-review aggregate into a
                         between-review part (signal) and the noise that
                         survives the operator (measured by resampling, not
                         inherited from the raw runs). Aggregation is supposed
                         to shrink the second without shrinking the first.
                         Collapse shows up here as between-review SD falling.

    mapping_sensitivity  The 2-tuple model places terms at equally spaced
                         indices. That is an assumption, not a measurement.
                         Re-place the terms non-uniformly and report whether
                         the conclusions move.

    operator_scope_contrast
                         Where an operator is allowed to act. The reported
                         pipeline snaps at most at the run-to-review step; this
                         reports what happens when it snaps at every level.

All of them take an `operator` argument, so the aggregation baselines in
`utils.fusion` are evaluated on identical footing.

Every resampled quantity carries a 95% interval. The replica counts are set so
that the Monte-Carlo error is small relative to the differences being claimed:
at 20,000 replicas a percentage is resolved to about half a point, and repeating
the whole analysis under a different seed moves the knife-edge figures by
0.3-0.4 points. An earlier version used 2,000, at which the same figures moved
by up to 2 points between seeds -- enough that a reported headline number was a
property of the seed rather than of the data. The interval is part of the
result, not a decoration.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from .cdm import TIE_DECIMALS, alternative_betas, rank_alternatives, scale_for
from .fusion import OPERATORS, apply_operator
from .twotuple import LinguisticScale, beta_from_y

DEFAULT_SEED = 42
DEFAULT_REPLICAS = 20000
DEFAULT_PANEL_REPLICAS = 5000

Z95 = 1.959963984540054


# --------------------------------------------------------------------------
# Monte-Carlo error
# --------------------------------------------------------------------------


def wilson_interval(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    """95% interval for a proportion, on the resampling count.

    Wilson rather than the normal approximation because the shares reported
    here run to 100%, where the normal interval extends past 1 and collapses to
    zero width. This is Monte-Carlo error -- how precisely the given number of
    replicas pins the resampling distribution -- and not a confidence interval
    for a population parameter.
    """

    if n <= 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denominator = 1.0 + z * z / n
    centre = p + z * z / (2.0 * n)
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return ((centre - half) / denominator, (centre + half) / denominator)


def _mean_interval(values: Sequence[float], z: float = Z95) -> tuple[float, float]:
    """Interval for a mean over replicas, from its Monte-Carlo standard error."""

    array = np.asarray(values, dtype=float)
    if array.size < 2:
        return (float("nan"), float("nan"))
    half = z * float(array.std(ddof=1)) / math.sqrt(array.size)
    mean = float(array.mean())
    return (mean - half, mean + half)


def _percentile_interval(
    values: Sequence[float], lo: float = 2.5, hi: float = 97.5
) -> tuple[float, float]:
    """Percentile interval of a bootstrap distribution."""

    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size < 2:
        return (float("nan"), float("nan"))
    return (float(np.percentile(array, lo)), float(np.percentile(array, hi)))


# --------------------------------------------------------------------------
# The resampling core
# --------------------------------------------------------------------------


@dataclass
class ReviewPanel:
    """One (model, scheme) laid out for resampling.

    The analyses below touch every review thousands of times. Regrouping the
    frame on each pass, or building a DataFrame per replica, dominates the
    runtime by two orders of magnitude, so the group is materialized once as
    parallel arrays and both aggregation levels above the review become a
    single matrix multiply -- see `weight_matrix`.
    """

    model: str
    scheme: str
    scale: LinguisticScale
    review_ids: list[str]
    runs: list[np.ndarray]
    user_ids: list[str]
    entity_ids: list[str]
    alternatives: list[str]

    @property
    def names(self) -> list[str]:
        """Alternatives in first-appearance order, matching `utils.cdm`."""

        return list(dict.fromkeys(self.alternatives))

    def full_values(self, operator: str) -> np.ndarray:
        """The operator applied to each review's runs as collected."""

        fn = OPERATORS[operator]
        return np.array([fn(a, self.scale) for a in self.runs], dtype=float)

    def bootstrap_values(
        self, operator: str, replicas: int, rng: np.random.Generator
    ) -> np.ndarray:
        """`(replicas, n_reviews)` of the operator applied to resampled runs.

        Runs are resampled with replacement within each review, holding the
        reviews themselves fixed: the question is what the model's own
        inference variability does to the decision, not what a different crowd
        would have said. `crowd_stability` asks the other question.
        """

        out = np.empty((replicas, len(self.runs)), dtype=float)
        for j, a in enumerate(self.runs):
            n = len(a)
            draws = a[rng.integers(0, n, (replicas, n))]
            out[:, j] = apply_operator(draws, operator, self.scale)
        return out

    def weight_matrix(self, hierarchy: str) -> np.ndarray:
        """`(n_alternatives, n_reviews)` linear map from reviews to alternatives.

        Both hierarchies are weighted means of the review values, so both are
        linear and both collapse to one matrix. Under `two_level` every review
        of an alternative carries equal weight; under `three_level` an expert's
        reviews are averaged first, so an expert who wrote six reviews of one
        restaurant contributes as much as one who wrote one.
        """

        names = self.names
        index = {name: i for i, name in enumerate(names)}
        matrix = np.zeros((len(names), len(self.review_ids)))

        if hierarchy == "two_level":
            for j, alternative in enumerate(self.alternatives):
                matrix[index[alternative], j] = 1.0
        elif hierarchy == "three_level":
            experts: dict[tuple[str, str], list[int]] = {}
            for j, (alternative, user) in enumerate(
                zip(self.alternatives, self.user_ids, strict=True)
            ):
                experts.setdefault((alternative, user), []).append(j)
            per_alternative: dict[str, int] = {}
            for alternative, _ in experts:
                per_alternative[alternative] = per_alternative.get(alternative, 0) + 1
            for (alternative, _), positions in experts.items():
                weight = 1.0 / (per_alternative[alternative] * len(positions))
                for j in positions:
                    matrix[index[alternative], j] = weight
        else:
            raise ValueError(
                f"hierarchy must be 'two_level' or 'three_level', got {hierarchy!r}"
            )

        totals = matrix.sum(axis=1, keepdims=True)
        return np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals > 0)


def build_panels(runs: pd.DataFrame) -> list[ReviewPanel]:
    """One `ReviewPanel` per (model, scheme) present in `runs`."""

    panels = []
    for (model, scheme), g in runs.groupby(["model", "scheme"], sort=False):
        scale = scale_for(scheme)
        review_ids, arrays, users, entities, alternatives = [], [], [], [], []
        for review_id, rg in g.groupby("review_id", sort=False):
            review_ids.append(str(review_id))
            arrays.append(
                np.array([beta_from_y(y, scale) for y in rg["y"]], dtype=float)
            )
            users.append(rg["user_id"].iloc[0])
            entities.append(rg["entity_id"].iloc[0])
            alternatives.append(rg["alternative"].iloc[0])
        panels.append(
            ReviewPanel(
                model=str(model),
                scheme=str(scheme),
                scale=scale,
                review_ids=review_ids,
                runs=arrays,
                user_ids=users,
                entity_ids=entities,
                alternatives=alternatives,
            )
        )
    return panels


def ranks_of(alternative_values: np.ndarray) -> np.ndarray:
    """Min-ranks, 1 = highest, over the last axis.

    Matches `utils.cdm.rank_alternatives`: ties share the lower rank, and the
    winner is the first alternative in frame order holding rank 1. Ties are
    rare for the lossless operator and routine for the snapping ones, so the
    rule has to agree between the two code paths.
    """

    # Collective values are means of rationals, so two alternatives can tie
    # exactly; summing in a different order (a batched product against a full
    # one) then leaves a difference of one ulp that would break the tie one way
    # in one table and the other way in the next. Compare at TIE_DECIMALS.
    values = np.round(np.atleast_2d(alternative_values), TIE_DECIMALS)
    greater = (values[:, :, None] < values[:, None, :]).sum(axis=2)
    return greater + 1


def _winner_index(ranks: np.ndarray) -> np.ndarray:
    return (ranks == 1).argmax(axis=1)


def keeps_winner(replica_ranks: np.ndarray, winner: int) -> np.ndarray:
    """Per replica: does the reference winner keep first place?

    The one definition of Top-1 used throughout (`utils.budget` and
    `utils.paired` included): a replica keeps the winner when no alternative
    *strictly* beats it, so a replica that ties it for first place keeps it.
    Reading the replica's own winner off frame order instead would score a
    tie as kept or lost depending on where the alternatives happen to sit in
    the corpus file.
    """

    return np.atleast_2d(replica_ranks)[:, winner] == 1


def _min_margin(values: np.ndarray) -> float:
    """Smallest gap between adjacently ranked alternatives."""

    ordered = np.sort(np.asarray(values, dtype=float))[::-1]
    return float(np.min(np.abs(np.diff(ordered)))) if ordered.size > 1 else float("nan")


def _adjacent_margins(values: np.ndarray) -> list[float]:
    """Beta gaps between the alternatives ranked i and i+1, best first."""

    ordered = np.sort(np.asarray(values, dtype=float))[::-1]
    return [round(float(g), 5) for g in -np.diff(ordered)]


def _replica_summary(
    replica_ranks: np.ndarray, reference_ranks: np.ndarray
) -> dict[str, float]:
    """Agreement of each replica's ranking with the full-data ranking."""

    replicas = replica_ranks.shape[0]
    taus = np.empty(replicas)
    for i in range(replicas):
        tau = kendalltau(reference_ranks, replica_ranks[i]).statistic
        taus[i] = 0.0 if np.isnan(tau) else tau

    exact = int(np.all(replica_ranks == reference_ranks, axis=1).sum())
    reference_winner = int(_winner_index(reference_ranks.reshape(1, -1))[0])
    top1 = int(keeps_winner(replica_ranks, reference_winner).sum())

    tau_lo, tau_hi = _mean_interval(taus)
    exact_lo, exact_hi = wilson_interval(exact, replicas)
    top1_lo, top1_hi = wilson_interval(top1, replicas)
    return {
        "tau_mean": round(float(taus.mean()), 4),
        "tau_lo": round(tau_lo, 4),
        "tau_hi": round(tau_hi, 4),
        "tau_min": round(float(taus.min()), 4),
        "exact_pct": round(100.0 * exact / replicas, 1),
        "exact_lo": round(100.0 * exact_lo, 1),
        "exact_hi": round(100.0 * exact_hi, 1),
        "top1_pct": round(100.0 * top1 / replicas, 1),
        "top1_lo": round(100.0 * top1_lo, 1),
        "top1_hi": round(100.0 * top1_hi, 1),
        "replicas": replicas,
    }


# --------------------------------------------------------------------------
# 1. Decision stability under inference noise
# --------------------------------------------------------------------------


def decision_stability(
    runs: pd.DataFrame,
    operator: str = "two_tuple",
    hierarchy: str = "two_level",
    replicas: int = DEFAULT_REPLICAS,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Resample the runs of each review and re-derive the ranking.

    This is stability *conditional on the observed crowd*: the 474 reviews are
    held fixed and only the repeated runs within them are resampled, so what is
    measured is the model's own inference variability. See `crowd_stability`
    for the complementary quantity.

    One row per (model, scheme) with `tau_mean`, `exact_pct`, `top1_pct`, each
    with a 95% Monte-Carlo interval, and, on the full data, `reference_winner`
    (the alternative Top-1 is measured against), `margin_top` (its beta lead
    over the runner-up), `margin_adj_<i>` (the gap between the alternatives
    ranked i and i+1) and `margin_min`, the smallest of those gaps.
    """

    rng = np.random.default_rng(seed)
    results = []
    for panel in build_panels(runs):
        matrix = panel.weight_matrix(hierarchy)

        full = panel.full_values(operator) @ matrix.T
        reference = ranks_of(full)[0]

        replica_values = panel.bootstrap_values(operator, replicas, rng) @ matrix.T
        summary = _replica_summary(ranks_of(replica_values), reference)

        names = panel.names
        winner = names[int(_winner_index(reference.reshape(1, -1))[0])]
        adjacent = _adjacent_margins(full)

        results.append(
            {
                "operator": operator,
                "hierarchy": hierarchy,
                "model": panel.model,
                "scheme": panel.scheme,
                "resampled": "runs",
                **summary,
                "reference_winner": winner,
                "margin_top": adjacent[0] if adjacent else float("nan"),
                **{f"margin_adj_{i}": m for i, m in enumerate(adjacent, start=1)},
                "margin_min": round(_min_margin(full), 5),
            }
        )
    return pd.DataFrame(results)


# --------------------------------------------------------------------------
# 2. Decision stability under crowd sampling
# --------------------------------------------------------------------------

CROWD_MODES = ("experts", "stratified")


@dataclass
class CrowdDraws:
    """One set of crowd replicas, drawn once per corpus and shared.

    Every system of a corpus is scored on the *same* redrawn crowds (common
    random numbers), so a difference between two systems' crowd figures is a
    difference between the systems and not between two Monte-Carlo streams.
    The crowd table, the joint table, the candidate sets and the per-rival gaps
    all read these replicas.

    `experts`     experts drawn with replacement from the whole corpus, each
                  bringing every review they wrote, across alternatives.
    `stratified`  within each alternative, its (alternative, expert) clusters
                  drawn with replacement, keeping the number of clusters per
                  alternative fixed. On a corpus sampled to a fixed number of
                  reviews per alternative this resamples under the sampling
                  design; an expert who reviewed two alternatives is then drawn
                  independently in each.

    `counts[r, u]` is how many times unit `u` (an expert, or a cluster) was
    drawn in replica `r`; `unit_of` maps each (alternative, expert) cluster to
    its unit.
    """

    mode: str
    counts: np.ndarray
    unit_of: dict[tuple[str, str], int]

    @property
    def replicas(self) -> int:
        return int(self.counts.shape[0])

    def cluster_counts(self, keys: Sequence[tuple[str, str]]) -> np.ndarray:
        """`(replicas, len(keys))` multiplicity of each (alternative, expert)."""

        return self.counts[:, [self.unit_of[k] for k in keys]]


def _tally(drawn: np.ndarray, units: int) -> np.ndarray:
    """Per replica, how many times each unit index occurs in `drawn`."""

    replicas = drawn.shape[0]
    counts = np.zeros((replicas, units))
    rows = np.repeat(np.arange(replicas), drawn.shape[1])
    np.add.at(counts, (rows, drawn.ravel()), 1.0)
    return counts


def draw_crowd(
    reviews: pd.DataFrame,
    replicas: int = DEFAULT_REPLICAS,
    seed: int = DEFAULT_SEED,
    mode: str = "experts",
) -> CrowdDraws:
    """Draw the crowd replicas of a corpus once, from its review-level frame.

    Only the `alternative` and `user_id` columns are read; pass the frame with
    every system of the corpus so that every (alternative, expert) cluster has
    a unit. Experts are sorted, so the draw does not depend on row order, and
    under `experts` it is the draw `utils.budget` makes from the same seed.
    """

    pairs = list(
        dict.fromkeys(zip(reviews["alternative"], reviews["user_id"], strict=True))
    )
    rng = np.random.default_rng(seed)
    if mode == "experts":
        experts = sorted({user for _, user in pairs})
        at = {e: i for i, e in enumerate(experts)}
        drawn = rng.integers(0, len(experts), (replicas, len(experts)))
        unit_of = {p: at[p[1]] for p in pairs}
        return CrowdDraws(mode, _tally(drawn, len(experts)), unit_of)
    if mode == "stratified":
        by_alternative: dict[str, list[str]] = {}
        for alternative, user in pairs:
            by_alternative.setdefault(alternative, []).append(user)
        blocks, unit_of, offset = [], {}, 0
        for alternative, users in by_alternative.items():
            users = sorted(users)
            drawn = rng.integers(0, len(users), (replicas, len(users)))
            blocks.append(_tally(drawn, len(users)))
            for i, user in enumerate(users):
                unit_of[(alternative, user)] = offset + i
            offset += len(users)
        return CrowdDraws(mode, np.concatenate(blocks, axis=1), unit_of)
    raise ValueError(f"mode must be one of {CROWD_MODES}, got {mode!r}")


def _crowd_values(
    g: pd.DataFrame, draws: CrowdDraws, hierarchy: str
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Alternatives, their full-data values and `(replicas, m)` crowd values.

    A drawn cluster brings its reviews' sum and count (two-level) or its mean
    with weight one (three-level), times its multiplicity. An alternative none
    of whose clusters was drawn has no value in that replica: NaN.
    """

    values = g["beta"].to_numpy(dtype=float)
    alternatives = list(g["alternative"])
    names = list(dict.fromkeys(alternatives))
    index = {name: i for i, name in enumerate(names)}

    clusters: dict[tuple[str, str], list[int]] = {}
    for j, key in enumerate(zip(alternatives, g["user_id"], strict=True)):
        clusters.setdefault(key, []).append(j)
    keys = list(clusters)

    totals = np.empty(len(keys))
    weights = np.empty(len(keys))
    for i, key in enumerate(keys):
        positions = clusters[key]
        if hierarchy == "two_level":
            totals[i] = values[positions].sum()
            weights[i] = len(positions)
        elif hierarchy == "three_level":
            totals[i] = values[positions].mean()
            weights[i] = 1.0
        else:
            raise ValueError(
                f"hierarchy must be 'two_level' or 'three_level', got {hierarchy!r}"
            )
    to_alternative = np.zeros((len(names), len(keys)))
    to_alternative[[index[k[0]] for k in keys], np.arange(len(keys))] = 1.0

    full = (to_alternative @ totals) / (to_alternative @ weights)
    multiplicity = draws.cluster_counts(keys)
    numerator = (multiplicity * totals) @ to_alternative.T
    denominator = (multiplicity * weights) @ to_alternative.T
    with np.errstate(invalid="ignore", divide="ignore"):
        replica = np.where(denominator > 0, numerator / denominator, np.nan)
    return names, full, replica


def crowd_stability(
    reviews: pd.DataFrame,
    hierarchy: str = "two_level",
    replicas: int = DEFAULT_REPLICAS,
    seed: int = DEFAULT_SEED,
    draws: CrowdDraws | None = None,
) -> pd.DataFrame:
    """Resample the experts and re-derive the ranking, model outputs held fixed.

    Takes a *review-level* frame -- the output of `utils.fusion.fuse_runs` or
    `fuse_models` -- rather than raw runs, so a fused system is scored by
    exactly the same code as an individual one. The `operator` column is
    carried through if present.

    Experts are drawn with replacement and each carries all of their reviews,
    because reviews by one expert are not independent observations of the
    crowd. This isolates the uncertainty that comes from *whose opinions were
    collected*, which run resampling says nothing about. `draws` are the
    corpus's shared replicas (`draw_crowd`); without them they are drawn here
    from `reviews` with `seed`, once for every group.

    `margin_lo`/`margin_hi` are percentiles of the smallest gap between
    alternatives *consecutive in the full-data ranking*, so the quantity is
    signed and goes negative when a replica inverts an adjacent pair. Whether
    the winner is distinguished from its rivals is `rival_gaps`' question, not
    this table's.
    """

    if draws is None:
        draws = draw_crowd(reviews, replicas, seed)
    group_keys = ["model", "scheme"]
    if "operator" in reviews.columns:
        group_keys = ["operator", *group_keys]

    results = []
    for key, g in reviews.groupby(group_keys, sort=False):
        labels = dict(zip(group_keys, key, strict=True))
        _, full, replica_values = _crowd_values(g, draws, hierarchy)
        reference = ranks_of(full)[0]
        order = np.argsort(reference, kind="stable")
        replica_values = np.where(np.isnan(replica_values), -np.inf, replica_values)

        ordered = replica_values[:, order]
        margins = (ordered[:, :-1] - ordered[:, 1:]).min(axis=1)
        margin_lo, margin_hi = _percentile_interval(margins)
        summary = _replica_summary(ranks_of(replica_values), reference)

        results.append(
            {
                **labels,
                "hierarchy": hierarchy,
                "resampled": "experts",
                "crowd_mode": draws.mode,
                "experts": int(g["user_id"].nunique()),
                **summary,
                "margin_min": round(_min_margin(full), 5),
                "margin_lo": round(margin_lo, 5),
                "margin_hi": round(margin_hi, 5),
                "inverts_pct": round(100.0 * float((margins < 0).mean()), 1),
            }
        )
    return pd.DataFrame(results)


def rival_gaps(
    reviews: pd.DataFrame,
    draws: CrowdDraws,
    hierarchy: str = "two_level",
) -> pd.DataFrame:
    """The winner against *every* rival, and each adjacent pair, under the crowd.

    For each system, on the shared crowd replicas: the signed gap between the
    full-data winner and each other alternative (`kind = winner`), and between
    each pair consecutive in the full-data ranking (`kind = adjacent`), with
    its full-data value, 2.5th and 97.5th replica percentiles, and the share of
    replicas in which it is zero or negative.

    The abstention rule reads `gap_lo`: a rival whose 2.5th percentile is at or
    below zero is not distinguished from the winner, a tie counting as not
    distinguished. Checking only the runner-up is not enough, because a rival
    further down can invert more often than the adjacent one when its
    evaluation is more variable.

    These are sensitivity statements under resampling, one pair at a time;
    they are not confidence sets with simultaneous coverage.
    """

    if "operator" in reviews.columns:
        reviews = reviews[reviews["operator"] == "two_tuple"]
    rows = []
    for (model, scheme), g in reviews.groupby(["model", "scheme"], sort=False):
        names, full, replica = _crowd_values(g, draws, hierarchy)
        replica = np.where(np.isnan(replica), -np.inf, replica)
        order = np.argsort(ranks_of(full)[0], kind="stable")
        pairs = [("winner", order[0], j) for j in order[1:]]
        pairs += [("adjacent", a, b) for a, b in pairwise(order)]
        for kind, a, b in pairs:
            gap = replica[:, a] - replica[:, b]
            lo, hi = _percentile_interval(gap)
            rows.append(
                {
                    "model": model,
                    "scheme": scheme,
                    "hierarchy": hierarchy,
                    "crowd_mode": draws.mode,
                    "kind": kind,
                    "upper": names[a],
                    "lower": names[b],
                    "upper_rank": int(np.where(order == a)[0][0]) + 1,
                    "lower_rank": int(np.where(order == b)[0][0]) + 1,
                    "gap": round(float(full[a] - full[b]), 5),
                    "gap_lo": round(lo, 5),
                    "gap_hi": round(hi, 5),
                    "le0_pct": round(100.0 * float((gap <= 0).mean()), 2),
                    "distinguished": bool(lo > 0),
                    "replicas": draws.replicas,
                }
            )
    return pd.DataFrame(rows)


def candidate_sets(gaps: pd.DataFrame) -> pd.DataFrame:
    """One row per system: the alternatives the crowd does not separate.

    The candidate set is the winner plus every rival whose gap to it is not
    distinguished (`rival_gaps`). The partial order applies the same test to
    each adjacent pair: `>` where the pair is distinguished, `~` where it is
    not. `~` is not transitive: A ~ B and B ~ C does not make A ~ C.
    """

    rows = []
    for (model, scheme, mode), g in gaps.groupby(
        ["model", "scheme", "crowd_mode"], sort=False
    ):
        rivals = g[g["kind"] == "winner"].sort_values("lower_rank")
        adjacent = g[g["kind"] == "adjacent"].sort_values("upper_rank")
        kept = rivals[~rivals["distinguished"]]
        closest = rivals.iloc[0]
        order = [adjacent.iloc[0]["upper"]] if len(adjacent) else []
        ranks = ["1"] if len(adjacent) else []
        for r in adjacent.itertuples():
            sign = ">" if r.distinguished else "~"
            order += [sign, r.lower]
            ranks += [sign, str(r.lower_rank)]
        rows.append(
            {
                "model": model,
                "scheme": scheme,
                "crowd_mode": mode,
                "winner": closest["upper"],
                "rivals": len(rivals),
                "rivals_kept": len(kept),
                "candidates": "; ".join(kept["lower"]),
                "candidate_ranks": ", ".join(str(r) for r in kept["lower_rank"]),
                "runner_up": closest["lower"],
                "gap_runner_up": closest["gap"],
                "gap_runner_up_lo": closest["gap_lo"],
                "gap_runner_up_hi": closest["gap_hi"],
                "runner_up_le0_pct": closest["le0_pct"],
                "min_gap_lo": float(rivals["gap_lo"].min()),
                "adjacent_distinguished": int(adjacent["distinguished"].sum()),
                "adjacent_pairs": len(adjacent),
                "partial_order": " ".join(order),
                "partial_order_ranks": " ".join(ranks),
                "replicas": int(closest["replicas"]),
            }
        )
    return pd.DataFrame(rows)


SIMULTANEOUS_METHODS = {True: "max_t", False: "percentile_max"}


def simultaneous_candidates(
    reviews: pd.DataFrame,
    draws: CrowdDraws,
    hierarchy: str = "two_level",
    level: float = 0.95,
    studentized: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A confidence set for the best alternative, from simultaneous intervals.

    Bootstrap max-t over the corpus's shared crowd replicas, for every system:

    1. On each replica b, the difference d*_ij = beta*_i - beta*_j for every
       pair i < j (i ranked above j in the full data), and its full-data
       value d_ij.
    2. se_ij, the standard deviation of d*_ij over the replicas (ddof = 0). A
       pair with se_ij = 0 (the same difference in every replica) is left out
       of the maximum and its interval is [d_ij, d_ij].
    3. T*_b = max_ij |d*_ij,b - d_ij| / se_ij, and c its `level` percentile.
    4. Simultaneous intervals d_ij -/+ c se_ij.
    5. j is a candidate unless some i has d_ij - c se_ij > 0: no alternative
       is simultaneously better than it. The observed winner is always in.

    Coverage: simultaneous `level` coverage over all pairwise differences of
    a system under the participant resampling design used throughout; not
    coverage with respect to a hypothetical population of platform users.

    `studentized=False` is the control variant (`percentile_max`): c is the
    `level` percentile of max_ij |d*_ij - d_ij| and the intervals d_ij -/+ c.

    The pairwise rule (`candidate_sets`) compares the winner with each rival at
    a pointwise 2.5th percentile. With max-t the simultaneous set contains it
    when the differences are roughly symmetric; that is checked on the data
    (`set_size_pairwise`, `pairwise_subset`), not assumed. A replica in which
    an alternative has no reviews cannot bound any difference: its statistic
    is infinite, as `rival_gaps` scores it as an inversion. It does not occur
    on the reported corpora.

    Returns `(pairs, sets)`: one row per pair, one row per system.
    """

    if "operator" in reviews.columns:
        reviews = reviews[reviews["operator"] == "two_tuple"]
    method = SIMULTANEOUS_METHODS[studentized]
    pairwise_sets = candidate_sets(rival_gaps(reviews, draws, hierarchy)).set_index(
        ["model", "scheme"]
    )
    pair_rows, set_rows = [], []
    for (model, scheme), g in reviews.groupby(["model", "scheme"], sort=False):
        names, full, replica = _crowd_values(g, draws, hierarchy)
        order = np.argsort(ranks_of(full)[0], kind="stable")
        pairs = [(a, b) for k, a in enumerate(order) for b in order[k + 1 :]]
        upper = np.array([a for a, _ in pairs])
        lower = np.array([b for _, b in pairs])
        gap_full = full[upper] - full[lower]
        gap = replica[:, upper] - replica[:, lower]
        missing = ~np.isfinite(gap).all(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            se = np.nanstd(gap, axis=0)
        deviation = np.abs(gap - gap_full)
        varies = se > 0
        if studentized:
            scaled = deviation[:, varies] / se[varies]
        else:
            scaled = deviation[:, varies]
        statistic = scaled.max(axis=1) if scaled.shape[1] else np.zeros(len(gap))
        statistic = np.where(missing, np.inf, statistic)
        c = float(np.percentile(statistic, 100.0 * level))
        half = np.where(varies, c * (se if studentized else 1.0), 0.0)
        lo, hi = gap_full - half, gap_full + half
        excluded = lo > 0

        rank_of = {int(a): k + 1 for k, a in enumerate(order)}
        for k, (a, b) in enumerate(pairs):
            pair_rows.append(
                {
                    "model": model,
                    "scheme": scheme,
                    "crowd_mode": draws.mode,
                    "method": method,
                    "i": names[a],
                    "j": names[b],
                    "i_rank": rank_of[int(a)],
                    "j_rank": rank_of[int(b)],
                    "gap_full": round(float(gap_full[k]), 5),
                    "se": round(float(se[k]), 5),
                    "lo_simult": round(float(lo[k]), 5),
                    "hi_simult": round(float(hi[k]), 5),
                    "excluded_by": bool(excluded[k]),
                }
            )
        beaten = {int(b) for (a, b), e in zip(pairs, excluded, strict=True) if e}
        members = [int(a) for a in order if int(a) not in beaten]
        pairwise = pairwise_sets.loc[(model, scheme)]
        pairwise_members = {pairwise["winner"]} | {
            c for c in str(pairwise["candidates"]).split("; ") if c
        }
        set_rows.append(
            {
                "model": model,
                "scheme": scheme,
                "crowd_mode": draws.mode,
                "method": method,
                "c": round(c, 4),
                "n_pairs": len(pairs),
                "winner": names[int(order[0])],
                "candidates": "; ".join(names[a] for a in members),
                "candidate_ranks": ", ".join(str(rank_of[a]) for a in members),
                "set_size": len(members),
                "set_size_pairwise": 1 + int(pairwise["rivals_kept"]),
                "pairwise_subset": pairwise_members <= {names[a] for a in members},
                "alternatives": len(names),
                "replicas": draws.replicas,
            }
        )
    return pd.DataFrame(pair_rows), pd.DataFrame(set_rows)


JOINT_CONDITIONS = ("runs", "experts", "both")


def _joint_inputs(
    runs: pd.DataFrame,
    fusion_members: Sequence[str] | None,
    fusion_label: str,
) -> list[
    tuple[
        str,
        str,
        LinguisticScale,
        list[str],
        list[str],
        list[str],
        list[list[np.ndarray]],
    ]
]:
    """Per system: (model, scheme, scale, review_ids, users, alternatives, runs).

    `runs` holds, per review, one array of run betas per member: one member
    for an individual system, every member that answered for the fused one.
    """

    panels = build_panels(runs)
    out = [
        (
            p.model,
            p.scheme,
            p.scale,
            p.review_ids,
            p.user_ids,
            p.alternatives,
            [[a] for a in p.runs],
        )
        for p in panels
    ]
    if fusion_members is None:
        return out
    for scheme in dict.fromkeys(p.scheme for p in panels):
        members = [
            p for p in panels if p.scheme == scheme and p.model in fusion_members
        ]
        if len(members) < 2:
            continue
        review_ids: list[str] = []
        meta: dict[str, tuple[str, str]] = {}
        arrays: dict[str, list[np.ndarray]] = {}
        for p in members:
            for j, review_id in enumerate(p.review_ids):
                if review_id not in arrays:
                    review_ids.append(review_id)
                    meta[review_id] = (p.user_ids[j], p.alternatives[j])
                    arrays[review_id] = []
                arrays[review_id].append(p.runs[j])
        out.append(
            (
                fusion_label,
                scheme,
                members[0].scale,
                review_ids,
                [meta[r][0] for r in review_ids],
                [meta[r][1] for r in review_ids],
                [arrays[r] for r in review_ids],
            )
        )
    return out


def joint_stability(
    runs: pd.DataFrame,
    operator: str = "two_tuple",
    replicas: int = DEFAULT_REPLICAS,
    seed: int = DEFAULT_SEED,
    fusion_members: Sequence[str] | None = None,
    fusion_label: str = "Fusion",
    batch: int = 1000,
    draws: CrowdDraws | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The ranking under run resampling, crowd resampling, and both at once.

    One code path for the three conditions, so their figures differ only in
    what is resampled:

    `runs`     each review's runs drawn with replacement, crowd held fixed;
    `experts`  experts drawn with replacement as clusters (an expert brings
               every review they wrote), runs held fixed; these are the
               corpus's shared crowd replicas (`draw_crowd`), so this row equals
               `crowd_stability`'s for the same system;
    `both`     experts drawn as clusters *and* the runs of every review drawn,
               within the same replica.

    Two-level hierarchy: a review's weight in its alternative is the number of
    times its author was drawn. For the fused system each member's runs are
    resampled independently and the members' review values are averaged, as
    `fuse_models` does (two_tuple operator only).

    Returns `(summary, intervals)`. `summary` has one row per (model, scheme,
    condition): Top-1 and Exact against the full-data ranking with Wilson
    intervals, the signed smallest margin between alternatives consecutive in
    the full-data order (percentile interval; below zero means a replica
    inverts an adjacent pair), the share of replicas inverting any adjacent
    pair, and `invert_<i>` for the pair ranked i and i+1. `intervals` has one
    row per (model, scheme, condition, alternative) with the full-data value
    and its 2.5th and 97.5th replica percentiles, for the figure.
    """

    if fusion_members is not None and operator != "two_tuple":
        raise ValueError("the fused system is only resampled under two_tuple")

    inputs = _joint_inputs(runs, fusion_members, fusion_label)
    if draws is None:
        # The crowd replicas of `crowd_stability`, from the same seed: one draw
        # per corpus, shared by every system and by both crowd conditions.
        corpus = pd.DataFrame(
            [
                (a, u)
                for *_, users, alternatives, _ in inputs
                for a, u in zip(alternatives, users, strict=True)
            ],
            columns=["alternative", "user_id"],
        )
        draws = draw_crowd(corpus, replicas, seed)
    if draws.replicas != replicas:
        raise ValueError("draws and replicas disagree")
    # Run draws come from their own stream, so adding run resampling to a
    # crowd replica does not change which crowd was drawn.
    rng = np.random.default_rng(seed + 1)
    summary, intervals = [], []
    for model, scheme, scale, review_ids, users, alternatives, arrays in inputs:
        names = list(dict.fromkeys(alternatives))
        index = {name: i for i, name in enumerate(names)}
        n_reviews = len(review_ids)
        indicator = np.zeros((len(names), n_reviews))
        for j, alternative in enumerate(alternatives):
            indicator[index[alternative], j] = 1.0

        experts = sorted(set(users))
        # Multiplicity of each review's (alternative, expert) cluster per
        # replica: under the two-level hierarchy, the review's weight.
        crowd_weights = draws.cluster_counts(
            list(zip(alternatives, users, strict=True))
        )

        fn = OPERATORS[operator]
        full_reviews = np.array(
            [np.mean([fn(a, scale) for a in member_runs]) for member_runs in arrays]
        )
        full = (indicator @ full_reviews) / indicator.sum(axis=1)
        reference = ranks_of(full)[0]
        order = np.argsort(reference, kind="stable")

        for condition in JOINT_CONDITIONS:
            values = np.empty((replicas, len(names)))
            for start in range(0, replicas, batch):
                size = min(batch, replicas - start)
                if condition == "experts":
                    review_values = np.broadcast_to(full_reviews, (size, n_reviews))
                else:
                    review_values = np.empty((size, n_reviews))
                    for j, member_runs in enumerate(arrays):
                        acc = np.zeros(size)
                        for a in member_runs:
                            resampled = a[rng.integers(0, len(a), (size, len(a)))]
                            acc += apply_operator(resampled, operator, scale)
                        review_values[:, j] = acc / len(member_runs)
                if condition == "runs":
                    weights = np.ones((size, n_reviews))
                else:
                    weights = crowd_weights[start : start + size]
                numerator = (weights * review_values) @ indicator.T
                denominator = weights @ indicator.T
                with np.errstate(invalid="ignore", divide="ignore"):
                    values[start : start + size] = numerator / denominator

            # An alternative none of whose authors was drawn has no value; such
            # a replica cannot rank it and is scored as not preserving anything.
            # It does not occur on the reported corpora but is guarded anyway.
            values = np.where(np.isnan(values), -np.inf, values)
            replica_ranks = ranks_of(values)
            stats = _replica_summary(replica_ranks, reference)
            ordered = values[:, order]
            gaps = ordered[:, :-1] - ordered[:, 1:]
            margins = gaps.min(axis=1)
            margin_lo, margin_hi = _percentile_interval(margins[np.isfinite(margins)])
            summary.append(
                {
                    "operator": operator,
                    "hierarchy": "two_level",
                    "model": model,
                    "scheme": scheme,
                    "condition": condition,
                    "reviews": n_reviews,
                    "experts": len(experts),
                    "reference_winner": names[int(order[0])],
                    "top1_pct": stats["top1_pct"],
                    "top1_lo": stats["top1_lo"],
                    "top1_hi": stats["top1_hi"],
                    "exact_pct": stats["exact_pct"],
                    "exact_lo": stats["exact_lo"],
                    "exact_hi": stats["exact_hi"],
                    "margin_min": round(_min_margin(full), 5),
                    "margin_lo": round(margin_lo, 5),
                    "margin_hi": round(margin_hi, 5),
                    "inverts_pct": round(100.0 * float((margins < 0).mean()), 1),
                    **{
                        f"invert_{i}": round(100.0 * float((gap < 0).mean()), 1)
                        for i, gap in enumerate(gaps.T, start=1)
                    },
                    "replicas": replicas,
                }
            )
            finite = np.where(np.isfinite(values), values, np.nan)
            lo = np.nanpercentile(finite, 2.5, axis=0)
            hi = np.nanpercentile(finite, 97.5, axis=0)
            for k, name in enumerate(names):
                intervals.append(
                    {
                        "model": model,
                        "scheme": scheme,
                        "condition": condition,
                        "alternative": name,
                        "rank": int(reference[k]),
                        "beta": round(float(full[k]), 5),
                        "beta_lo": round(float(lo[k]), 5),
                        "beta_hi": round(float(hi[k]), 5),
                    }
                )
    return pd.DataFrame(summary), pd.DataFrame(intervals)


# --------------------------------------------------------------------------
# 3. Discriminance / non-collapse
# --------------------------------------------------------------------------


def discriminance(
    runs: pd.DataFrame,
    operator: str = "two_tuple",
    replicas: int = DEFAULT_PANEL_REPLICAS,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Between-review signal against the noise each operator leaves behind.

    Three spreads, all in beta units, and they answer different questions:

        between_sd   spread of the aggregated per-review evaluation across
                     reviews. Signal: how far apart the pipeline places
                     different reviews. Falls when a pipeline buys stability by
                     refusing to discriminate.

        input_sd     mean spread of the *raw* repeated runs within a review.
                     This is what the operator is given, so it is identical
                     across operators by construction and cannot be used to
                     compare them.

        residual_sd  spread of the *operator's own output* under run
                     resampling, measured by bootstrap and pooled across
                     reviews as a root mean square: the square root of the
                     mean over reviews of the across-replica variance of the
                     operator's output.
                     This is the noise that survives aggregation, and it is the
                     quantity that differs between operators. It is pooled in
                     variance units because it is compared against a variance;
                     `fusion_variance_reduction` deliberately averages standard
                     deviations instead, because there the quantity is a ratio
                     of movements. On a system where most reviews are noiseless
                     and a few are not, the two pooling rules differ by a factor
                     of three, so which one is in use has to be stated.

    `snr` is between_sd / residual_sd and `signal_share` is
    $\\sigma^2_b / (\\sigma^2_b + \\sigma^2_{\\mathrm{res}})$. Deliberately not
    called an ICC: it is a descriptive ratio of two measured spreads, not an
    estimate of a variance component from a fitted random-effects model, and
    the reviews here are not exchangeable draws from one population.

    `distinct_values` has to be read alongside both: an operator that maps
    every review onto the same term scores a perfect `signal_share` of nothing.
    It counts *review-level* values. `distinct_collective` counts the
    alternative-level values the two-level hierarchy derives from them: the
    levels above the review average, so a snapping operator's few review-level
    values still average into distinct collective values. Only an operator
    applied at every level collapses those too (`operator_scope_contrast`).
    """

    rng = np.random.default_rng(seed)
    rows = []
    for panel in build_panels(runs):
        aggregated = panel.full_values(operator)
        input_var = float(np.mean([np.var(a, ddof=0) for a in panel.runs]))
        draws = panel.bootstrap_values(operator, replicas, rng)
        residual_var = float(np.mean(draws.var(axis=0, ddof=0)))

        var_b = float(np.var(aggregated, ddof=0))
        rows.append(
            {
                "operator": operator,
                "model": panel.model,
                "scheme": panel.scheme,
                "between_sd": round(float(np.sqrt(var_b)), 4),
                "input_sd": round(float(np.sqrt(input_var)), 4),
                "residual_sd": round(float(np.sqrt(residual_var)), 4),
                "snr": round(float(np.sqrt(var_b / residual_var)), 3)
                if residual_var > 0
                else float("inf"),
                "signal_share": round(var_b / (var_b + residual_var), 4)
                if var_b + residual_var
                else np.nan,
                "distinct_values": len(set(np.round(aggregated, 9))),
                "distinct_collective": len(
                    set(np.round(aggregated @ panel.weight_matrix("two_level").T, 9))
                ),
                "replicas": replicas,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 5. Sensitivity to how rejected outputs are handled
# --------------------------------------------------------------------------

# How a review whose runs did not all survive parsing is treated.
#
#   available  keep the review, average whatever runs parsed
#   listwise   drop the review entirely unless every run parsed
#
# Neither is neutral. `available` compares reviews aggregated from different
# numbers of runs, and when the rejections are not missing at random it shifts
# the reviews they land on. `listwise` removes those reviews outright, which
# shifts the corpus instead. The study reports both.
MISSING_POLICIES = ("available", "listwise")


def apply_missing_policy(
    runs: pd.DataFrame, policy: str, expected_runs: int = 10
) -> pd.DataFrame:
    """Filter a runs table under one missing-output policy."""

    if policy == "available":
        return runs
    if policy != "listwise":
        raise ValueError(f"policy must be one of {MISSING_POLICIES}, got {policy!r}")

    counts = runs.groupby(["model", "scheme", "review_id"]).size()
    complete = counts[counts >= expected_runs].index
    indexed = runs.set_index(["model", "scheme", "review_id"])
    return indexed.loc[indexed.index.isin(complete)].reset_index()


def rejection_bias(runs: pd.DataFrame, expected_runs: int = 10) -> pd.DataFrame:
    """Is the missingness related to what the review would have scored?

    One row per (model, scheme) that has any rejection, comparing the mean
    normalized value of the reviews that lost runs against the reviews that did
    not, and the share of each alternative's reviews affected. A difference in
    the first, or a spread in the second, is evidence that the rejections are
    not missing at random and therefore that the choice of policy is not a
    formality.
    """

    rows = []
    for (model, scheme), g in runs.groupby(["model", "scheme"], sort=False):
        counts = g.groupby("review_id").size()
        short = set(counts[counts < expected_runs].index)
        if not short:
            continue
        affected = g[g["review_id"].isin(short)]
        rest = g[~g["review_id"].isin(short)]
        unique = g[["review_id", "alternative"]].drop_duplicates()
        unique = unique.assign(is_short=unique["review_id"].isin(short))
        per_alt = unique.groupby("alternative")["is_short"].mean()
        rows.append(
            {
                "model": model,
                "scheme": scheme,
                "reviews_affected": len(short),
                "mean_y_affected": round(
                    float(affected.groupby("review_id")["y"].mean().mean()), 4
                ),
                "mean_y_rest": round(
                    float(rest.groupby("review_id")["y"].mean().mean()), 4
                ),
                "min_alt_rate": round(100 * float(per_alt.min()), 1),
                "max_alt_rate": round(100 * float(per_alt.max()), 1),
            }
        )
    return pd.DataFrame(rows)


def missing_data_sensitivity(
    runs: pd.DataFrame,
    operator: str = "two_tuple",
    hierarchy: str = "two_level",
    expected_runs: int = 10,
    fusion_members: Sequence[str] | None = None,
    fusion_label: str = "Fusion",
) -> pd.DataFrame:
    """Ranking and margin under each missing-output policy.

    Same shape as `mapping_sensitivity`: the question is whether a modelling
    choice the analyst makes without comment moves the decision.

    With `fusion_members`, the fused system is added. Under `available` it is
    the fusion every other table reports. Under `listwise` a review enters only
    if every member returned all `expected_runs` runs for it, so the fused
    system and each member are scored on complete evidence alone.
    """

    from .fusion import fuse_models, fuse_runs

    rows = []

    def record(policy, model, scheme, names, values, reviews):
        ranks = ranks_of(values)[0]
        order = np.argsort(ranks, kind="stable")
        rows.append(
            {
                "policy": policy,
                "operator": operator,
                "hierarchy": hierarchy,
                "model": model,
                "scheme": scheme,
                "reviews": reviews,
                "winner": names[int(order[0])],
                "order": " > ".join(names[int(i)] for i in order),
                "margin_min": round(_min_margin(values), 5),
            }
        )

    for policy in MISSING_POLICIES:
        filtered = apply_missing_policy(runs, policy, expected_runs)
        for panel in build_panels(filtered):
            values = panel.full_values(operator) @ panel.weight_matrix(hierarchy).T
            record(
                policy,
                panel.model,
                panel.scheme,
                panel.names,
                values,
                len(panel.review_ids),
            )

        if fusion_members is None:
            continue
        members = filtered[filtered["model"].isin(fusion_members)]
        fused = fuse_models(
            fuse_runs(members, operator), operator, fusion_members, fusion_label
        )
        if policy == "listwise":
            fused = fused[fused["n_models"] == members["model"].nunique()]
        for scheme, g in fused.groupby("scheme", sort=False):
            if scheme == "SCORE" and operator != "two_tuple":
                continue
            panel = ReviewPanel(
                model=fusion_label,
                scheme=str(scheme),
                scale=scale_for(str(scheme)),
                review_ids=list(g["review_id"]),
                runs=[np.array([b]) for b in g["beta"]],
                user_ids=list(g["user_id"]),
                entity_ids=list(g["entity_id"]),
                alternatives=list(g["alternative"]),
            )
            values = np.array(g["beta"], dtype=float) @ panel.weight_matrix(hierarchy).T
            record(policy, fusion_label, scheme, panel.names, values, len(g))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 6. Sensitivity to the linguistic mapping
# --------------------------------------------------------------------------

# Term placements on the index domain. `uniform` is what the 2-tuple model
# assumes and what every other table in the study uses. The other two keep the
# endpoints and the ordering fixed and move the interior terms, which is the
# minimal way to ask whether equal spacing is doing the work: `compressed`
# pushes the intensifiers towards the centre (adjacent categories close
# together, extremes far out), `stretched` does the opposite.
PLACEMENTS: dict[str, dict[int, tuple[float, ...]]] = {
    "uniform": {
        4: (0.0, 1.0, 2.0, 3.0, 4.0),
        6: (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
    },
    "compressed": {
        4: (0.0, 1.6, 2.0, 2.4, 4.0),
        6: (0.0, 2.0, 2.6, 3.0, 3.4, 4.0, 6.0),
    },
    "stretched": {
        4: (0.0, 0.5, 2.0, 3.5, 4.0),
        6: (0.0, 0.4, 1.2, 3.0, 4.8, 5.6, 6.0),
    },
}


def _replace_positions(
    betas: np.ndarray, scale: LinguisticScale, placement: str
) -> np.ndarray:
    """Move each term from its uniform index to its placed position.

    Only meaningful for values sitting on term indices, i.e. ordinal schemes
    before aggregation. Non-integral values are left alone.
    """

    positions = PLACEMENTS[placement][scale.g]
    out = betas.copy()
    for index, position in enumerate(positions):
        out[np.isclose(betas, index)] = position
    return out


def mapping_sensitivity(
    runs: pd.DataFrame,
    operator: str = "two_tuple",
    hierarchy: str = "two_level",
    fusion_members: Sequence[str] | None = None,
    fusion_label: str = "Fusion",
) -> pd.DataFrame:
    """Rankings under each term placement, and their agreement with uniform.

    Only the ordinal schemes are placed: SCORE outputs do not sit on term
    indices, so moving the terms would not be well defined for them.

    With `fusion_members`, a row for the fused system is added per scheme and
    placement: each member's review value is computed under the placement, the
    operator fuses them across members review by review (as `fuse_models`
    does), and the fused review values are aggregated like any system's.
    """

    fn = OPERATORS[operator]
    rows = []
    # (scheme, placement) -> review_id -> (alternative, user_id, member values)
    fused: dict[tuple[str, str], dict[str, tuple[str, str, list[float]]]] = {}
    scales: dict[str, LinguisticScale] = {}

    def record(model, scheme, placement, names, ranks, reference_ranks):
        tau = (
            1.0
            if reference_ranks is None
            else float(kendalltau(reference_ranks, ranks).statistic)
        )
        order = np.argsort(ranks, kind="stable")
        rows.append(
            {
                "operator": operator,
                "hierarchy": hierarchy,
                "model": model,
                "scheme": scheme,
                "placement": placement,
                "winner": names[int(order[0])],
                "order": " > ".join(names[int(i)] for i in order),
                "tau_vs_uniform": round(tau, 6),
            }
        )

    for panel in build_panels(runs):
        if panel.scheme == "SCORE":
            continue
        matrix = panel.weight_matrix(hierarchy)
        names = panel.names
        is_member = fusion_members is not None and panel.model in fusion_members
        scales[panel.scheme] = panel.scale

        reference_ranks: np.ndarray | None = None
        for placement in PLACEMENTS:
            values = np.array(
                [
                    fn(_replace_positions(a, panel.scale, placement), panel.scale)
                    for a in panel.runs
                ],
                dtype=float,
            )
            if is_member:
                slot = fused.setdefault((panel.scheme, placement), {})
                for j, review_id in enumerate(panel.review_ids):
                    entry = slot.setdefault(
                        review_id, (panel.alternatives[j], panel.user_ids[j], [])
                    )
                    entry[2].append(float(values[j]))
            ranks = ranks_of(values @ matrix.T)[0]
            record(panel.model, panel.scheme, placement, names, ranks, reference_ranks)
            if reference_ranks is None:
                reference_ranks = ranks

    references: dict[str, np.ndarray] = {}
    for (scheme, placement), slot in fused.items():
        scale = scales[scheme]
        review_ids = list(slot)
        panel = ReviewPanel(
            model=fusion_label,
            scheme=scheme,
            scale=scale,
            review_ids=review_ids,
            runs=[np.array(slot[r][2]) for r in review_ids],
            user_ids=[slot[r][1] for r in review_ids],
            entity_ids=["" for _ in review_ids],
            alternatives=[slot[r][0] for r in review_ids],
        )
        # The member values are already placed; the fusion only averages them.
        values = np.array([fn(list(a), scale) for a in panel.runs], dtype=float)
        ranks = ranks_of(values @ panel.weight_matrix(hierarchy).T)[0]
        record(
            fusion_label, scheme, placement, panel.names, ranks, references.get(scheme)
        )
        references.setdefault(scheme, ranks)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 7. Sensitivity to where the operator is allowed to act
# --------------------------------------------------------------------------


def operator_scope_contrast(
    reviews_by_operator: dict[str, pd.DataFrame],
    hierarchy: str = "two_level",
) -> pd.DataFrame:
    """The collective ranking when the operator acts at one level or at all.

    `reviews_by_operator` maps an operator name to the review-level frame it
    produced (see `utils.fusion.fuse_runs`). Under scope `review` the levels
    above the review take an arithmetic mean of beta, which is the reported
    pipeline; under `all_levels` the same operator is applied again at
    review -> expert and expert -> alternative, so the collective evaluation
    itself lands on a term.

    The distinction is not cosmetic and is the reason the reported pipeline has
    to say where it snaps: an operator applied at every level cannot express a
    difference smaller than one term, so alternatives that a study would want
    ranked come back tied.
    """

    rows = []
    for operator, reviews in reviews_by_operator.items():
        for scope in ("review", "all_levels"):
            snap = None if scope == "review" else OPERATORS[operator]
            ranked = rank_alternatives(alternative_betas(reviews, hierarchy, snap=snap))
            for (model, scheme), g in ranked.groupby(["model", "scheme"], sort=False):
                values = g["beta"].to_numpy()
                rows.append(
                    {
                        "operator": operator,
                        "scope": scope,
                        "hierarchy": hierarchy,
                        "model": model,
                        "scheme": scheme,
                        "winner": g.iloc[0]["alternative"],
                        "order": " > ".join(g["alternative"]),
                        "distinct_values": len(set(np.round(values, 9))),
                        "ties": int(len(values) - len(set(np.round(values, 9)))),
                        "margin_min": round(_min_margin(values), 5),
                    }
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Tie rule of the majority operator
# --------------------------------------------------------------------------


def _modal(values: Sequence[float], upper: bool) -> float:
    """Modal term of `values`, ties broken towards the upper or lower term."""

    terms = np.floor(np.asarray(values, dtype=float) + 0.5)
    uniques, counts = np.unique(terms, return_counts=True)
    modes = uniques[counts == counts.max()]
    return float(modes.max() if upper else modes.min())


def tiebreak_sensitivity(
    runs: pd.DataFrame,
    fusion_members: Sequence[str] | None = None,
    fusion_label: str = "Fusion",
) -> pd.DataFrame:
    """What the majority operator's tie rule decides, per system.

    `majority` breaks a tie between modal terms towards the lower term, a
    stated convention. This recomputes every review under the opposite rule
    (towards the upper term) and reports how many reviews have a tied mode, how
    many change value, and the winner, order and smallest margin of the
    collective ranking (two-level hierarchy) under each rule. The fused system
    takes the mode of its members' review values under the same rule.
    """

    rows = []
    per_review: dict[str, dict[bool, dict[str, float]]] = {}
    meta: dict[str, str] = {}
    for panel in build_panels(runs):
        values = {
            upper: {
                r: _modal(a, upper)
                for r, a in zip(panel.review_ids, panel.runs, strict=True)
            }
            for upper in (False, True)
        }
        meta.update(zip(panel.review_ids, panel.alternatives, strict=True))
        if fusion_members is not None and panel.model in fusion_members:
            per_review[panel.model] = values
        tied = sum(_modal(a, False) != _modal(a, True) for a in panel.runs)
        rows.append(_tiebreak_row(panel.model, panel.scheme, values, meta, tied))
    if fusion_members is not None and per_review:
        scheme = str(runs["scheme"].iloc[0])
        # In corpus order, not set order: a set's iteration order changes with
        # the process's string hashing, and with tied alternatives the order
        # of first appearance decides which is listed first.
        shared = set.intersection(*(set(v[False]) for v in per_review.values()))
        first = next(iter(per_review.values()))[False]
        common = [r for r in first if r in shared]
        values = {}
        for upper in (False, True):
            values[upper] = {
                r: _modal([v[upper][r] for v in per_review.values()], upper)
                for r in common
            }
        tied = sum(values[False][r] != values[True][r] for r in common)
        rows.append(_tiebreak_row(fusion_label, scheme, values, meta, None, tied))
    return pd.DataFrame(rows)


def _tiebreak_row(
    model: str,
    scheme: str,
    values: dict[bool, dict[str, float]],
    alternative_of: dict[str, str],
    tied: int | None,
    changed: int | None = None,
) -> dict[str, object]:
    """One row of `tiebreak_sensitivity`: both rules side by side."""

    out: dict[str, object] = {"model": model, "scheme": scheme}
    lower, upper = values[False], values[True]
    out["reviews_tied"] = tied if tied is not None else np.nan
    out["reviews_changed"] = (
        changed if changed is not None else sum(lower[r] != upper[r] for r in lower)
    )
    for rule, v in (("lower", lower), ("upper", upper)):
        frame = pd.DataFrame(
            {"alternative": [alternative_of[r] for r in v], "beta": list(v.values())}
        )
        betas = frame.groupby("alternative", sort=False)["beta"].mean()
        ordered = betas.round(TIE_DECIMALS).sort_values(ascending=False, kind="stable")
        out[f"winner_{rule}"] = ordered.index[0]
        out[f"order_{rule}"] = " > ".join(ordered.index)
        out[f"margin_{rule}"] = round(_min_margin(ordered.to_numpy()), 5)
    out["same_winner"] = out["winner_lower"] == out["winner_upper"]
    out["same_order"] = out["order_lower"] == out["order_upper"]
    return out
