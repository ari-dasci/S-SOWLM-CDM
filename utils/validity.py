"""Convergent validity against the rating the review's own author gave.

Every other analysis in this study is internal: it asks whether the pipeline
agrees with itself under resampling and under the analyst's choices. None of it
can rule out the possibility that the pipeline is stably measuring nothing, and
without an external anchor that objection cannot be answered.

`data/TripR-2020Large_FullReviews.csv` carries one. `REVIEW_RATING_GLOBAL` is
the 1--5 star rating the reviewer attached to the review they wrote. It is not
an annotation we produced, it is not a third party's judgement of the text, and
it was never used to build any other number in this study.

**What is compared, and on what scale.** The star scale and the LABEL-5
vocabulary are both 5-point ordinal scales, so the comparison is made on the
term-index domain the 2-tuple model is already defined on, with no rescaling
and no free parameters:

    star rating   1  2  3  4  5
    beta_star     0  1  2  3  4
    LABEL-5       very_negative  negative  neutral  positive  very_positive
    beta          0              1         2        3         4

`beta_star = rating - 1`. One star is one term, and a difference of 0.5 in
$\\beta$ means the same thing on both sides. That exact correspondence is why
LABEL-5 is the headline comparison and why a signed bias in $\\beta$ units is
reportable rather than only a correlation. LABEL-7 has a different $g$ and
SCORE has no term set, so those two are compared on the normalized $[-1,1]$
domain instead, where `y_star = (rating - 3) / 2`.

**What this is not.** The star is an overall rating of a visit; the LABEL-5
answer is the polarity of the review text. They are related constructs elicited
from different sources, not the same quantity measured twice, and they can
legitimately differ -- a diner may write warmly about the food and still deduct
a star for the bill. Agreement is therefore evidence that the pipeline tracks
what the crowd meant, and disagreement is not evidence that the pipeline is
wrong. Nothing here establishes a correct ranking, and no system is scored
against another on closeness to the stars: a system that reproduced the star
ranking exactly would not thereby be the better system, since the star ranking
carries its own near-ties (see `star_alternatives`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr

from .cdm import TIE_DECIMALS, review_position, scale_for
from .stability import _min_margin, ranks_of
from .twotuple import SCALE_5

RATING_COLUMN = "REVIEW_RATING_GLOBAL"


def star_frame(data_csv: Path) -> pd.DataFrame:
    """One row per review position: the author's rating on both domains."""

    source = pd.read_csv(data_csv)
    rating = source[RATING_COLUMN].to_numpy()
    return pd.DataFrame(
        {
            "position": np.arange(len(source)),
            "alternative": source["ENTITY_NAME"].to_numpy(),
            "user_id": source["USER_ID"].to_numpy(),
            "rating": rating,
            # One star is one term; see the module docstring.
            "beta_star": rating - 1.0,
            "y_star": (rating - 3.0) / 2.0,
        }
    )


def _attach_star(reviews: pd.DataFrame, stars: pd.DataFrame) -> pd.DataFrame:
    """Join review-level evaluations to their author's rating, by position."""

    position = reviews["review_id"].map(review_position)
    indexed = stars.set_index("position")
    out = reviews.copy()
    out["rating"] = indexed["rating"].reindex(position).to_numpy()
    out["beta_star"] = indexed["beta_star"].reindex(position).to_numpy()
    out["y_star"] = indexed["y_star"].reindex(position).to_numpy()
    return out


def review_agreement(reviews: pd.DataFrame, stars: pd.DataFrame) -> pd.DataFrame:
    """Per-review agreement between the pipeline and the author's own rating.

    One row per (model, scheme). Correlations are reported for every scheme on
    whichever domain is common to it and the stars. The columns that require
    the two scales to coincide -- signed bias, mean absolute deviation and the
    share of reviews landing on the author's own term -- are reported for
    LABEL-5 only, and are left empty elsewhere, because a 7-point label and a
    5-point star do not have a shared unit to be biased by.
    """

    joined = _attach_star(reviews, stars)

    rows = []
    for (model, scheme), g in joined.groupby(["model", "scheme"], sort=False):
        g = g.dropna(subset=["rating"])
        if len(g) < 3:
            continue
        scale = scale_for(scheme)
        # Comparable domain: term indices where the scales coincide, the
        # normalized [-1, 1] domain otherwise.
        aligned = scheme == "LABEL-5" and scale is SCALE_5
        theirs = g["beta_star"] if aligned else g["y_star"]
        ours = (
            g["beta"]
            if aligned
            else g["beta"].map(lambda b, s=scale: 2.0 * b / s.g - 1.0)
        )

        row = {
            "model": model,
            "scheme": scheme,
            "n": len(g),
            "domain": "beta (5-point)" if aligned else "y [-1,1]",
            "pearson_r": round(float(pearsonr(ours, theirs)[0]), 4),
            "spearman_rho": round(float(spearmanr(ours, theirs)[0]), 4),
        }
        if aligned:
            difference = ours.to_numpy() - theirs.to_numpy()
            row["bias_beta"] = round(float(difference.mean()), 4)
            row["mad_beta"] = round(float(np.abs(difference).mean()), 4)
            # The pipeline's aggregate is continuous, so "same term" means the
            # 2-tuple rounds onto the term the author chose. Delta rounds halves
            # up, so a review at beta = k - 0.5 lands on term k: |d| < 0.5 would
            # miss it.
            term = np.floor(ours.to_numpy() + 0.5)
            row["same_term_pct"] = round(
                100.0 * float((term == theirs.to_numpy()).mean()), 1
            )
        rows.append(row)
    return pd.DataFrame(rows)


def star_alternatives(
    stars: pd.DataFrame, hierarchy: str = "two_level"
) -> pd.DataFrame:
    """The collective ranking the crowd's own star ratings produce.

    Built with the same hierarchy as the LLM rankings so the two are
    comparable: `two_level` averages the ratings of an alternative directly,
    `three_level` averages within expert first.
    """

    if hierarchy == "two_level":
        level = stars
    elif hierarchy == "three_level":
        level = stars.groupby(["alternative", "user_id"], as_index=False)[
            "beta_star"
        ].mean()
    else:
        raise ValueError(
            f"hierarchy must be 'two_level' or 'three_level', got {hierarchy!r}"
        )

    out = (
        level.groupby("alternative", as_index=False)
        .agg(beta=("beta_star", "mean"), n=("beta_star", "size"))
        .sort_values("beta", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    out["hierarchy"] = hierarchy
    # Means of integer ratings tie exactly (Madrid: 50 reviews each); a tie
    # shares the lower rank, as in `utils.cdm.rank_alternatives`.
    out["rank"] = (
        out["beta"].round(TIE_DECIMALS).rank(ascending=False, method="min").astype(int)
    )
    return out


def ranking_agreement(
    alternatives: pd.DataFrame, scheme: str, hierarchy: str, stars: pd.DataFrame
) -> pd.DataFrame:
    """Kendall tau between each system's collective ranking and the stars'.

    Reported with both margins side by side and in the same units, which is the
    point of the exercise: a disagreement across a gap neither side resolves is
    not a disagreement about anything.
    """

    star_rank = star_alternatives(stars, hierarchy)
    order = list(star_rank["alternative"])
    star_margin = _min_margin(star_rank["beta"].to_numpy())

    sub = alternatives[
        (alternatives["scheme"] == scheme) & (alternatives["hierarchy"] == hierarchy)
    ]
    rows = []
    for model, g in sub.groupby("model", sort=False):
        values = g.set_index("alternative")["beta"]
        if not set(order) <= set(values.index):
            continue
        ours = values.reindex(order).to_numpy()
        ranks = ranks_of(ours)[0]
        # tau-b against the star ranks, ties included: breaking the stars' ties
        # by row order would make tau depend on it.
        tau = kendalltau(star_rank["rank"].to_numpy(), ranks).statistic
        winner = order[int(np.argsort(ranks, kind="stable")[0])]
        rows.append(
            {
                "model": model,
                "scheme": scheme,
                "hierarchy": hierarchy,
                "winner": winner,
                "star_winner": order[0],
                "same_winner": winner == order[0],
                "tau_vs_stars": round(float(tau), 4),
                "margin_min": round(_min_margin(ours), 5),
                "star_margin_min": round(star_margin, 5),
            }
        )
    return pd.DataFrame(rows)
