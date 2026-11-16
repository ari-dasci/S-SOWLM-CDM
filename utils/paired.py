"""Paired comparisons between systems and the variance of the winner's lead.

    python -m utils.paired london
    python -m utils.paired madrid

Everything is computed on the corpus's shared participant replicas (20,000,
seed 42: the draws of `utils.cdm_analyze`), under LABEL-5, the 2-tuple operator
and the two-level hierarchy. Written into the profile's `out_dir`:

    cdm_variance_lead.csv   per system: the winner's lead, its standard
                            deviation over the participant replicas (`se`), the
                            ratio var_inf(k) / var_part, the smallest k that
                            brings it under 10% and 5%, and the normal
                            approximations of crowd Top-1.
    cdm_paired.csv          per comparison A against B: the difference of
                            leads on the same redrawn crowds with its
                            percentile interval, and the difference of crowd
                            Top-1 with a nested-bootstrap standard error.
    cdm_star_paired.csv     differences in review-level correlation with the
                            reviewers' stars, paired over the redrawn crowds.
    cdm_star_gaps.csv       the stars as a system: each pairwise gap of the
                            star evaluation minus the same gap of the fusion.

**var_inf(k)** is the variance of the lead, participants fixed, when every
review receives k runs drawn with replacement from its ten collected runs
(rejected runs discarded; fusion members drawn independently). The lead is a
fixed weighted average of review means, so it is sum_j c_j^2 s_j^2 / k exactly,
with c_j the weight of review j in the lead and s_j^2 the variance of one run
of that review. `mc_over_exact_min`/`_max` check it by Monte Carlo over
k = 1, 2, 3, 5, 10, and `mc_over_exact_accepted` at ten runs under the design
of the run-resampling tables (each member's accepted runs redrawn).

**Top-1 differences.** The spread of a Top-1 difference over the replicas only
measures Monte Carlo error. Its sampling error comes from a nested bootstrap:
the first 2,000 shared crowds are outer replicas, each receives 1,000 inner
redraws of its participants, and every configuration's Top-1 is recomputed
against its own winner in the outer crowd (the median over run subsets for
budget and greedy rows). The standard error is the SD of the outer differences,
less the inner Monte Carlo variance where it can be estimated, and the interval
is the difference -/+ 1.96 SE.
"""

from __future__ import annotations

import argparse
import math
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, norm

from .budget import N_RUNS, _run_matrix, _subset_values
from .cdm import TIE_DECIMALS, attach_entities, review_position
from .cdm_analyze import FUSION_LABEL, SCHEME
from .fusion import fuse_models, fuse_runs
from .profiles import Profile, load_profile
from .runs import load_runs
from .stability import DEFAULT_REPLICAS, DEFAULT_SEED, draw_crowd, ranks_of
from .validity import star_frame

F = FUSION_LABEL
STARS = "Stars"
THRESHOLDS = (0.10, 0.05)
MC_BUDGETS = (1, 2, 3, 5, 10)
MC_SEED = 43
NESTED_SEED = 4242
OUTER, INNER = 2000, 1000

# What is compared. A configuration is a system name, optionally followed by
# ` k=<runs>` (every k-subset of its ten runs), ` T=0` (each greedy execution
# in turn) or ` majority` (review-level majority voting). The third element
# names the pair of alternatives whose gap is compared when the two systems do
# not share their leading pair; None compares the gap of A's leading pair.
ZENITH, MONTES = "Zenith Brunch & Cocktails - Valverde", "Los Montes de Galicia"
COMPARISONS = {
    "london": [
        ("GPT-5.2", "Gemma 4 E4B", None),
        ("GPT-5.2", "Ministral 3 8B", None),
        ("GPT-5.2", "Gemma 4 31B", None),
        ("GPT-5.2", "Ornith 35B", None),
        ("GPT-5.2", "Qwen 3.6 35B", None),
        ("GPT-5.2", F, None),
        ("Gemma 4 31B", F, None),
        ("Ornith 35B", F, None),
        ("Qwen 3.6 35B", F, None),
        (F, "Gemma 4 E4B", None),
        (F, "Ministral 3 8B", None),
        (f"{F} majority", F, None),
        ("Gemma 4 31B k=5", f"{F} k=1", None),
        ("Gemma 4 31B", f"{F} k=2", None),
        (F, f"{F} k=1", None),
    ],
    "madrid": [
        ("Qwen 3.6 35B", "Gemma 4 E4B", (ZENITH, MONTES)),
        ("Qwen 3.6 35B", "Ministral 3 8B", (ZENITH, MONTES)),
        ("Qwen 3.6 35B", F, (ZENITH, MONTES)),
        (F, "Gemma 4 E4B", None),
        (F, "Ministral 3 8B", None),
        (f"{F} majority", F, None),
        ("Qwen 3.6 35B k=1", F, None),
        ("Qwen 3.6 35B k=3", f"{F} k=1", None),
        (F, f"{F} k=1", None),
        ("Gemma 4 E4B T=0", "Gemma 4 E4B", None),
        ("Ministral 3 8B T=0", "Ministral 3 8B", None),
        ("Qwen 3.6 35B T=0", "Qwen 3.6 35B", None),
        (f"{F} T=0", F, None),
    ],
}
STAR_CORRELATIONS = {
    "london": [
        ("Gemma 4 31B", "GPT-5.2"),
        ("Ornith 35B", "GPT-5.2"),
        ("Qwen 3.6 35B", "GPT-5.2"),
        (F, "GPT-5.2"),
        (F, "Ornith 35B"),
    ],
    "madrid": [("Qwen 3.6 35B", F)],
}


class Corpus:
    """One corpus laid out for resampling: its reviews in a fixed order, each
    system's run matrix, and the shared participant replicas."""

    def __init__(self, profile: Profile, replicas: int = DEFAULT_REPLICAS):
        self.profile = profile
        runs = attach_entities(load_runs(profile), profile.data_csv)
        self.runs = runs[runs["scheme"] == SCHEME]
        self.members = profile.members
        self.systems = [*profile.model_order, F]
        frame = fuse_runs(self.runs, "two_tuple")
        self.with_fusion = pd.concat(
            [frame, fuse_models(frame, "two_tuple", self.members, F)],
            ignore_index=True,
        )
        first = self.with_fusion[self.with_fusion["model"] == self.systems[0]]
        self.reviews = first[["review_id", "user_id", "alternative"]].reset_index(
            drop=True
        )
        self.review_ids = list(self.reviews["review_id"])
        self.names = list(dict.fromkeys(self.reviews["alternative"]))
        self.n_alt = len(self.names)
        self.alt_index = np.array(
            [self.names.index(a) for a in self.reviews["alternative"]]
        )
        self.draws = draw_crowd(self.with_fusion, replicas, DEFAULT_SEED)
        keys = zip(self.reviews["alternative"], self.reviews["user_id"], strict=True)
        self.unit = np.array([self.draws.unit_of[k] for k in keys])
        self.n_units = self.draws.counts.shape[1]
        self.matrices = {m: self.run_matrix(m, self.runs) for m in profile.model_order}

    def run_matrix(self, model: str, runs: pd.DataFrame) -> np.ndarray:
        """`(n_reviews, 10)` betas by run index, NaN where a run was rejected."""

        return _run_matrix(runs[runs["model"] == model], self.review_ids)

    def member_matrices(self, system: str) -> list[np.ndarray]:
        members = self.members if system == F else [system]
        return [self.matrices[m] for m in members]

    def review_values(self, system: str, operator: str = "two_tuple") -> np.ndarray:
        """Ten-run review values of a system, aligned with `review_ids`."""

        frame = self.with_fusion
        if operator != "two_tuple":
            frame = fuse_runs(self.runs, operator)
            frame = pd.concat(
                [frame, fuse_models(frame, operator, self.members, F)],
                ignore_index=True,
            )
        values = frame[frame["model"] == system].set_index("review_id")["beta"]
        return values.reindex(self.review_ids).to_numpy(dtype=float)

    def star_values(self) -> np.ndarray:
        stars = star_frame(self.profile.data_csv).set_index("position")["beta_star"]
        positions = [review_position(r) for r in self.review_ids]
        return stars.loc[positions].to_numpy(dtype=float)

    def configuration(self, name: str) -> np.ndarray:
        """`(n_reviews, S)` review values of a named configuration, one column
        per run subset (S = 1 for a ten-run system)."""

        if name == STARS:
            return self.star_values()[:, None]
        if name.endswith(" majority"):
            return self.review_values(name.removesuffix(" majority"), "majority")[
                :, None
            ]
        if " k=" in name:
            system, k = name.rsplit(" k=", 1)
            matrices = self.member_matrices(system)
            return np.stack(
                [
                    _subset_values(matrices, s)
                    for s in combinations(range(N_RUNS), int(k))
                ],
                axis=1,
            )
        if name.endswith(" T=0"):
            system = name.removesuffix(" T=0")
            greedy = load_runs(load_profile(f"{self.profile.name}_t0"))
            greedy = attach_entities(greedy, self.profile.data_csv)
            members = self.members if system == F else [system]
            matrices = [self.run_matrix(m, greedy) for m in members]
            return np.stack(
                [_subset_values(matrices, (p,)) for p in range(N_RUNS)], axis=1
            )
        return self.review_values(name)[:, None]

    # -- alternative values under participant weights ----------------------

    def unit_sums(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per resampling unit, the sum and count of each alternative's review
        values: `(n_units, S * n_alt)` each. A review without a value is
        dropped from its alternative's mean."""

        present = np.isfinite(values)
        v = np.where(present, values, 0.0)
        columns = values.shape[1]
        num = np.zeros((self.n_units, columns, self.n_alt))
        den = np.zeros((self.n_units, columns, self.n_alt))
        np.add.at(num, (self.unit, slice(None), self.alt_index), v)
        np.add.at(den, (self.unit, slice(None), self.alt_index), present.astype(float))
        shape = (self.n_units, columns * self.n_alt)
        return num.reshape(shape), den.reshape(shape)

    def alternative_values(
        self, counts: np.ndarray, num: np.ndarray, den: np.ndarray
    ) -> np.ndarray:
        """`(len(counts), S, n_alt)` alternative values when unit u is drawn
        `counts[., u]` times; -inf where an alternative has no review."""

        with np.errstate(invalid="ignore", divide="ignore"):
            out = (counts @ num) / (counts @ den)
        out = np.where(np.isfinite(out), out, -np.inf)
        return out.reshape(counts.shape[0], -1, self.n_alt)


def winner_runner(values: np.ndarray) -> tuple[int, int]:
    """Winner and runner-up as the pipeline orders them."""

    order = np.argsort(ranks_of(values)[0], kind="stable")
    return int(order[0]), int(order[1])


def kept(values: np.ndarray, winner: np.ndarray) -> np.ndarray:
    """`values` (B, S, n_alt), `winner` (S,) -> (B, S): no alternative
    strictly beats the winner (a tie keeps it, as `keeps_winner`)."""

    r = np.round(values, TIE_DECIMALS)
    w = r[:, np.arange(r.shape[1]), winner]
    return ~(r > w[:, :, None]).any(axis=2)


# --------------------------------------------------------------------------
# Variance of the winner's lead
# --------------------------------------------------------------------------


def _run_variance(matrices: list[np.ndarray]) -> np.ndarray:
    """Per review, the variance of its value under one requested run.

    sum_i (10 / n_i) s_i^2 / m^2 over the m members with an accepted run, with
    s_i^2 the variance (ddof = 0) of member i's n_i accepted runs: one
    requested run yields n_i / 10 accepted runs on average.
    """

    out = np.zeros(matrices[0].shape[0])
    for j in range(len(out)):
        total, members = 0.0, 0
        for matrix in matrices:
            accepted = matrix[j][np.isfinite(matrix[j])]
            if accepted.size:
                members += 1
                total += accepted.var(ddof=0) * N_RUNS / accepted.size
        out[j] = total / members**2
    return out


def _resampled_lead(
    matrices: list[np.ndarray],
    indicator: np.ndarray,
    pair: tuple[int, int],
    k: int | None,
    rng: np.random.Generator,
    replicas: int,
) -> np.ndarray:
    """The lead in `replicas` redraws of each review's runs, participants fixed.

    With `k`, k of the ten run slots are drawn with replacement and the
    rejected ones dropped (k requested runs). With `k=None`, each member's
    accepted runs are redrawn, as many as it has: the design of the
    run-resampling tables.
    """

    values = np.empty((replicas, matrices[0].shape[0]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for j in range(values.shape[1]):
            per_member = []
            for m in matrices:
                if k is not None:
                    drawn = m[j][rng.integers(0, N_RUNS, (replicas, k))]
                    per_member.append(np.nanmean(drawn, axis=1))
                elif (accepted := m[j][np.isfinite(m[j])]).size:
                    n = accepted.size
                    per_member.append(
                        accepted[rng.integers(0, n, (replicas, n))].mean(axis=1)
                    )
            values[:, j] = (
                per_member[0]
                if len(per_member) == 1
                else np.nanmean(np.stack(per_member, axis=1), axis=1)
            )
    present = np.isfinite(values)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = (np.where(present, values, 0.0) @ indicator) / (present @ indicator)
    return means[:, pair[0]] - means[:, pair[1]]


def variance_of_lead(corpus: Corpus) -> pd.DataFrame:
    indicator = np.zeros((len(corpus.review_ids), corpus.n_alt))
    indicator[np.arange(len(corpus.review_ids)), corpus.alt_index] = 1.0
    values = np.stack([corpus.review_values(s) for s in corpus.systems], axis=1)
    num, den = corpus.unit_sums(values)
    full = corpus.alternative_values(np.ones((1, corpus.n_units)), num, den)[0]
    replica = corpus.alternative_values(corpus.draws.counts, num, den)
    rng = np.random.default_rng(MC_SEED)

    rows = []
    for si, system in enumerate(corpus.systems):
        w, r = winner_runner(full[si])
        lead = float(full[si, w] - full[si, r])
        var_part = float((replica[:, si, w] - replica[:, si, r]).var(ddof=0))
        se = math.sqrt(var_part)
        top1 = float(kept(replica[:, [si], :], np.array([w])).mean())

        # Normal approximations of crowd Top-1: against the runner-up alone,
        # and jointly against every rival.
        rivals = [j for j in range(corpus.n_alt) if j != w]
        gaps = replica[:, si, [w]] - replica[:, si, rivals]
        joint = multivariate_normal(
            mean=-(full[si, w] - full[si, rivals]),
            cov=np.cov(gaps, rowvar=False, ddof=0),
            allow_singular=True,
        ).cdf(np.zeros(len(rivals)))

        # Weight of each review in the lead, then var_inf(1).
        weights = np.zeros(len(corpus.review_ids))
        for alternative, sign in ((w, 1.0), (r, -1.0)):
            members = np.isfinite(values[:, si]) & (corpus.alt_index == alternative)
            weights[members] = sign / members.sum()
        matrices = corpus.member_matrices(system)
        var_1 = float((weights**2 * _run_variance(matrices)).sum())

        # Monte Carlo check of var_inf(k); None is the accepted-run design.
        mc = []
        for k in (*MC_BUDGETS, None):
            lead_k = _resampled_lead(
                matrices, indicator, (w, r), k, rng, corpus.draws.replicas
            )
            mc.append(float(np.nanvar(lead_k)) / (var_1 / (k or N_RUNS)))
        mc_accepted = mc.pop()
        rows.append(
            {
                "system": system,
                "winner": corpus.names[w],
                "runner_up": corpus.names[r],
                "lead": round(lead, 5),
                "se": round(se, 5),
                "var_part": var_part,
                "var_inf_1": var_1,
                **{
                    f"ratio_k{k}_pct": round(100 * var_1 / k / var_part, 2)
                    for k in (1, 3, 10)
                },
                **{
                    f"k_star_{round(100 * t)}pct": max(
                        1, math.ceil(var_1 / (t * var_part) - 1e-12)
                    )
                    for t in THRESHOLDS
                },
                "mc_over_exact_min": round(min(mc), 3),
                "mc_over_exact_max": round(max(mc), 3),
                "mc_over_exact_accepted": round(mc_accepted, 3),
                "normal_top1_pct": round(100 * float(norm.cdf(lead / se)), 1),
                "multivariate_top1_pct": round(100 * float(joint), 1),
                "top1_pct": round(100 * top1, 1),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Paired comparisons
# --------------------------------------------------------------------------


def _interval(values: np.ndarray) -> tuple[float, float]:
    return (
        round(float(np.percentile(values, 2.5)), 3),
        round(float(np.percentile(values, 97.5)), 3),
    )


def paired_comparisons(
    corpus: Corpus,
    comparisons: list[tuple[str, str, tuple[str, str] | None]],
    outer: int = OUTER,
    inner: int = INNER,
) -> pd.DataFrame:
    names = list(dict.fromkeys(n for a, b, _ in comparisons for n in (a, b)))
    sums = {n: corpus.unit_sums(corpus.configuration(n)) for n in names}
    num = np.concatenate([sums[n][0] for n in names], axis=1)
    den = np.concatenate([sums[n][1] for n in names], axis=1)
    bounds = np.cumsum([0, *(sums[n][0].shape[1] for n in names)])

    def split(counts: np.ndarray) -> dict[str, np.ndarray]:
        """Alternative values of every configuration: name -> (B, S, n_alt)."""

        flat = corpus.alternative_values(counts, num, den).reshape(len(counts), -1)
        return {
            n: flat[:, a:b].reshape(len(counts), -1, corpus.n_alt)
            for n, a, b in zip(names, bounds[:-1], bounds[1:], strict=True)
        }

    full = {n: v[0] for n, v in split(np.ones((1, corpus.n_units))).items()}
    replica = split(corpus.draws.counts)
    winners = {n: np.array([winner_runner(row)[0] for row in full[n]]) for n in names}
    retained = {n: kept(replica[n], winners[n]) for n in names}
    top1 = {n: float(np.median(retained[n].mean(axis=0))) for n in names}

    # Nested bootstrap of each configuration's Top-1.
    rng = np.random.default_rng(NESTED_SEED)
    size = int(corpus.draws.counts[0].sum())
    outer_counts = corpus.draws.counts[:outer]
    outer_values = split(outer_counts)
    outer_winner = {
        n: np.argmax(np.round(outer_values[n], TIE_DECIMALS), axis=-1) for n in names
    }
    single = [
        (a, b) for a, b, _ in comparisons if len(full[a]) == 1 and len(full[b]) == 1
    ]
    outer_top1 = {n: np.empty(outer) for n in names}
    inner_mc = {pair: np.empty(outer) for pair in single}
    for o in range(outer):
        draws = rng.multinomial(size, outer_counts[o] / size, size=inner).astype(float)
        values = split(draws)
        inside = {n: kept(values[n], outer_winner[n][o]) for n in names}
        for n in names:
            outer_top1[n][o] = np.median(inside[n].mean(axis=0))
        for a, b in single:
            d = inside[a][:, 0].astype(float) - inside[b][:, 0].astype(float)
            inner_mc[a, b][o] = d.var(ddof=1) / inner

    rows = []
    for a, b, named in comparisons:
        difference = top1[a] - top1[b]
        spread = (outer_top1[a] - outer_top1[b]).var(ddof=1)
        if (a, b) in inner_mc:
            spread -= float(inner_mc[a, b].mean())
        se = math.sqrt(max(spread, 0.0))
        row = {
            "A": a,
            "B": b,
            "top1_A": round(100 * top1[a], 1),
            "top1_B": round(100 * top1[b], 1),
            "top1_diff": round(100 * difference, 1),
            "top1_se": round(100 * se, 1),
            "top1_lo": round(100 * max(difference - 1.96 * se, -1), 1),
            "top1_hi": round(100 * min(difference + 1.96 * se, 1), 1),
        }
        if (a, b) in inner_mc:
            # Ten-run systems: the lead, on the same redrawn crowds.
            paired = retained[a][:, 0].astype(float) - retained[b][:, 0]
            row["top1_mc_se"] = round(
                100 * float(paired.std(ddof=1)) / math.sqrt(len(paired)), 2
            )
            pair_a, pair_b = winner_runner(full[a][0]), winner_runner(full[b][0])
            pairs = {"lead": pair_a}
            if named is not None and pair_a != pair_b:
                pairs = {
                    "lead": tuple(corpus.names.index(x) for x in named),
                    "lead_alt": pair_a,
                }
            row["same_pair"] = pair_a == pair_b
            for key, (i, j) in pairs.items():
                gap = replica[a][:, 0, i] - replica[a][:, 0, j]
                gap -= replica[b][:, 0, i] - replica[b][:, 0, j]
                observed = (full[a][0, i] - full[a][0, j]) - (
                    full[b][0, i] - full[b][0, j]
                )
                lo, hi = _interval(gap)
                row[f"{key}_pair"] = f"{corpus.names[i]} - {corpus.names[j]}"
                row[f"{key}_diff"] = round(float(observed), 4)
                row[f"{key}_lo"], row[f"{key}_hi"] = lo, hi
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# The reviewers' stars
# --------------------------------------------------------------------------


def star_correlations(corpus: Corpus, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Differences in review-level Pearson correlation with the stars, each
    correlation weighted by the multiplicities of the redrawn crowd."""

    stars = corpus.star_values()
    weights = corpus.draws.counts[:, corpus.unit]

    def correlation(x: np.ndarray) -> tuple[np.ndarray, float]:
        ok = np.isfinite(x)
        w, xx, yy = weights[:, ok], x[ok], stars[ok]
        total = w.sum(axis=1)
        mx, my = (w @ xx) / total, (w @ yy) / total
        cov = (w @ (xx * yy)) / total - mx * my
        vx = (w @ (xx * xx)) / total - mx**2
        vy = (w @ (yy * yy)) / total - my**2
        return cov / np.sqrt(vx * vy), float(np.corrcoef(xx, yy)[0, 1])

    rows = []
    for a, b in pairs:
        ra, ra_full = correlation(corpus.review_values(a))
        rb, rb_full = correlation(corpus.review_values(b))
        rows.append(
            {
                "A": a,
                "B": b,
                "r_A": round(ra_full, 4),
                "r_B": round(rb_full, 4),
                "diff": round(ra_full - rb_full, 4),
                "lo": round(float(np.percentile(ra - rb, 2.5)), 4),
                "hi": round(float(np.percentile(ra - rb, 97.5)), 4),
            }
        )
    return pd.DataFrame(rows)


def star_gaps(corpus: Corpus) -> pd.DataFrame:
    """Each pairwise gap of the star evaluation minus the fusion's, pairs in
    the order of the star ranking."""

    values = np.stack([corpus.star_values(), corpus.review_values(F)], axis=1)
    num, den = corpus.unit_sums(values)
    full = corpus.alternative_values(np.ones((1, corpus.n_units)), num, den)[0]
    replica = corpus.alternative_values(corpus.draws.counts, num, den)
    order = np.argsort(-np.round(full[0], TIE_DECIMALS), kind="stable")
    rows = []
    for i, j in combinations(order, 2):
        difference = (replica[:, 0, i] - replica[:, 0, j]) - (
            replica[:, 1, i] - replica[:, 1, j]
        )
        lo, hi = _interval(difference)
        gap_stars, gap_fusion = full[0, i] - full[0, j], full[1, i] - full[1, j]
        rows.append(
            {
                "i": corpus.names[i],
                "j": corpus.names[j],
                "gap_stars": round(float(gap_stars), 4),
                "gap_fusion": round(float(gap_fusion), 4),
                "diff": round(float(gap_stars - gap_fusion), 4),
                "lo": lo,
                "hi": hi,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("profile", choices=sorted(COMPARISONS))
    args = parser.parse_args()

    profile = load_profile(args.profile)
    corpus = Corpus(profile)
    outputs = {
        "cdm_variance_lead.csv": variance_of_lead(corpus),
        "cdm_paired.csv": paired_comparisons(corpus, COMPARISONS[profile.name]),
        "cdm_star_paired.csv": star_correlations(
            corpus, STAR_CORRELATIONS[profile.name]
        ),
        "cdm_star_gaps.csv": star_gaps(corpus),
    }
    pd.set_option("display.width", 250)
    for name, table in outputs.items():
        table.to_csv(profile.out_dir / name, index=False)
        print(f"\n{profile.out_dir / name}")
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
