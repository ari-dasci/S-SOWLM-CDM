"""Crowd decision-making aggregation: runs -> review -> expert -> alternative.

The runs table is keyed by `review_id`, the review folder the runner wrote.
That folder name is `review-<i>` where `i` is the row position in the source
CSV, so the expert (`USER_ID`) and the alternative (`ENTITY_ID`) behind every
run are recoverable by position. `attach_entities` does that join and checks it
against the `uid` the runner recorded, so a mismatched CSV fails loudly instead
of silently reassigning reviews to the wrong restaurant.

Position, not `uid`, is the join key on purpose: `uid` is
`USER_ID + "_" + ENTITY_ID` and is not unique (474 review folders carry 402
distinct uids, because some experts reviewed the same restaurant more than
once). Joining on it would merge those reviews.

Two hierarchies are built, and the study reports both:

    two_level    runs -> review -> alternative
    three_level  runs -> review -> expert -> alternative

They differ in how much weight an expert carries. Under `two_level` an expert
who wrote six reviews of one restaurant counts six times; under `three_level`
their reviews are averaged first, so every expert counts once per alternative.

Aggregation above the review is a plain arithmetic mean on the continuous
$\\beta$ domain unless a `snap` callable is supplied, in which case it is
applied at every level instead. The reported pipeline does not snap: the
operators of `utils.fusion` are compared at the run-to-review step, and the
levels above carry $\\beta$ forward. `snap` exists so that the alternative --
collapsing onto a term at *every* level, which is what an applied study using
majority vote throughout would do -- can be reported rather than assumed away;
see `utils.stability.operator_scope_contrast`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd

from .twotuple import (
    SCALE_BY_SCHEME,
    LinguisticScale,
    delta,
    format_tuple,
)


def review_position(review_id: str) -> int:
    """`review-12` -> 12, the row position in the source CSV."""

    _, _, number = review_id.partition("-")
    return int(number)


def attach_entities(runs: pd.DataFrame, data_csv: Path) -> pd.DataFrame:
    """Add expert and alternative columns to a runs table.

    Raises if the recorded `uid` disagrees with the row the position points at,
    which is what happens when the runs were collected against a different
    input file.
    """

    # Identifiers are text: the UniOvi corpus has numeric restaurant ids, which
    # pandas would otherwise read as integers and fail to join on.
    source = pd.read_csv(data_csv, dtype={"USER_ID": str, "ENTITY_ID": str})
    position = runs["review_id"].map(review_position)

    out_of_range = position[(position < 0) | (position >= len(source))]
    if len(out_of_range):
        raise ValueError(
            f"{len(out_of_range)} runs point past the end of {data_csv.name} "
            f"({len(source)} rows); highest position {out_of_range.max()}"
        )

    runs = runs.copy()
    runs["user_id"] = source["USER_ID"].to_numpy()[position]
    runs["entity_id"] = source["ENTITY_ID"].to_numpy()[position]
    runs["alternative"] = source["ENTITY_NAME"].to_numpy()[position]

    if "uid" in runs.columns:
        expected = runs["user_id"] + "_" + runs["entity_id"]
        wrong = runs.loc[expected != runs["uid"]]
        if len(wrong):
            sample = wrong.iloc[0]
            raise ValueError(
                f"{len(wrong)} runs disagree with {data_csv.name}: "
                f"{sample['review_id']} records uid {sample['uid']!r} but that "
                f"position holds {expected.loc[wrong.index[0]]!r}. The runs "
                "were collected against a different input file."
            )

    return runs


def scale_for(scheme: str) -> LinguisticScale:
    """The term set a scheme's 2-tuples live on.

    SCORE has no term set of its own. It is expressed on the 5-point grid so
    that a continuous baseline can be compared against LABEL-5 in the same
    linguistic domain.
    """

    return SCALE_BY_SCHEME.get(scheme, SCALE_BY_SCHEME["LABEL-5"])


Snap = Callable[[Sequence[float], LinguisticScale], float]


def _collapse(
    frame: pd.DataFrame,
    keys: list[str],
    count_col: str,
    out_count: str,
    snap: Snap | None,
) -> pd.DataFrame:
    """Group `frame` by `keys` and reduce `beta` to one value per group."""

    if snap is None:
        return (
            frame.groupby(keys, sort=False)
            .agg(beta=("beta", "mean"), **{out_count: (count_col, "size")})
            .reset_index()
        )

    scheme_at = keys.index("scheme")
    rows = []
    for key, g in frame.groupby(keys, sort=False):
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "beta": snap(list(g["beta"]), scale_for(key[scheme_at])),
                out_count: len(g),
            }
        )
    return pd.DataFrame(rows)


def expert_betas(reviews: pd.DataFrame, snap: Snap | None = None) -> pd.DataFrame:
    """Level 2 of the three-level hierarchy: reviews of one expert about one
    alternative, aggregated to one beta."""

    return _collapse(
        reviews,
        ["model", "scheme", "alternative", "entity_id", "user_id"],
        "review_id",
        "n_reviews",
        snap,
    )


def alternative_betas(
    reviews: pd.DataFrame,
    hierarchy: str = "two_level",
    snap: Snap | None = None,
) -> pd.DataFrame:
    """Collective evaluation of each alternative, per model and scheme.

    `hierarchy` is `two_level` (reviews averaged directly) or `three_level`
    (reviews averaged within expert first, then experts averaged). `snap`
    replaces the arithmetic mean at every level above the review; the default
    of `None` is the reported pipeline.
    """

    if hierarchy == "two_level":
        level = reviews
        count_col = "review_id"
    elif hierarchy == "three_level":
        level = expert_betas(reviews, snap)
        count_col = "user_id"
    else:
        raise ValueError(
            f"hierarchy must be 'two_level' or 'three_level', got {hierarchy!r}"
        )

    out = _collapse(level, ["model", "scheme", "alternative"], count_col, "n", snap)
    out["hierarchy"] = hierarchy
    out["two_tuple"] = [
        format_tuple(delta(b, scale_for(s)), scale_for(s))
        for b, s in zip(out["beta"], out["scheme"], strict=True)
    ]
    return out


# Collective values closer than this many decimals are a tie: exact ties are
# real (means of rationals) and must not be broken by summation order.
TIE_DECIMALS = 9


def rank_alternatives(alternatives: pd.DataFrame) -> pd.DataFrame:
    """Rank 1 = highest collective evaluation, within each model and scheme."""

    out = alternatives.copy()
    out["rank"] = (
        out["beta"]
        .round(TIE_DECIMALS)
        .groupby([out["model"], out["scheme"], out["hierarchy"]], sort=False)
        .rank(ascending=False, method="min")
        .astype(int)
    )
    return out.sort_values(
        ["model", "scheme", "hierarchy", "rank"], kind="stable"
    ).reset_index(drop=True)
