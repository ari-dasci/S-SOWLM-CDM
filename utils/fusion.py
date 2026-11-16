"""Aggregation operators, applied at the two fusion points of the pipeline.

    within-model   the repeated runs of one review, one model  -> one value
    across-model   that review's per-model values              -> one value

Every operator sees exactly the same runs, so differences between them are
attributable to the operator and nothing else. That is the whole point: the
claim under test is that 2-tuple aggregation buys something over the simpler
alternatives, and without a common input there is no way to tell.

A note on what actually distinguishes the 2-tuple operator, because it is
easy to overstate. On the term-index domain the 2-tuple arithmetic mean *is*
the arithmetic mean: `two_tuple` and the mean of the betas are the same
number. What the 2-tuple adds is that the number survives. It is carried
forward as a continuous beta and only ever rendered as a pair
$(s_i, \\alpha)$, whereas the operators an applied study would otherwise reach
for -- majority vote, mean-then-round, median-then-round -- collapse onto a
term at every step. So the operators below split into two families, and the
comparison measures what the collapse costs:

    lossless   two_tuple            keeps beta continuous
    snapping   majority, mean_round, median_round

A further distinction matters for what the study is entitled to claim.
`continuous_mean` is the arithmetic mean on beta reported as a number;
`two_tuple` is the same number rendered as $(s_i, \alpha)$. They are the same
arithmetic, so every numerical benefit reported for the lossless operator is a
benefit of *not snapping*, not of the 2-tuple model as such. What the 2-tuple
model contributes is that the surviving value can still be stated in the
linguistic vocabulary the crowd was asked to use, which a bare 3.17 cannot.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from .cdm import scale_for
from .twotuple import LinguisticScale, _round_half_up, beta_from_y

Operator = Callable[[Sequence[float], LinguisticScale], float]


def op_continuous_mean(betas: Sequence[float], scale: LinguisticScale) -> float:
    """The arithmetic mean on beta, reported as a bare number.

    Present so that the study does not have to claim more for the 2-tuple model
    than the data support. Every stability, margin, resolution and residual
    noise figure attributed to `two_tuple` below is produced by *this*
    function: `op_two_tuple` calls it and changes nothing numerical. What the
    2-tuple adds is the rendering, not the arithmetic, and separating the two
    lets each be credited with what it does.
    """

    return float(np.mean(betas))


def op_two_tuple(betas: Sequence[float], scale: LinguisticScale) -> float:
    """The same mean, carried forward to be rendered through Delta.

    Identical to `op_continuous_mean` by construction, and a test pins the two
    to each other over the whole corpus. The distinction is representational:
    this value is expressed as a pair $(s_i, \alpha)$, which is what makes it
    reportable in the vocabulary the crowd was asked to use.
    """

    return op_continuous_mean(betas, scale)


def op_mean_round(betas: Sequence[float], scale: LinguisticScale) -> float:
    """Arithmetic mean snapped back onto a term. The classic lossy average."""

    return float(_round_half_up(float(np.mean(betas))))


def op_median_round(betas: Sequence[float], scale: LinguisticScale) -> float:
    """Median snapped back onto a term."""

    return float(_round_half_up(float(np.median(betas))))


def op_majority(betas: Sequence[float], scale: LinguisticScale) -> float:
    """Modal term. Ties break towards the lower term index, deterministically.

    Ties are common at 10 runs, so the rule has to be stated: `Counter` orders
    by first insertion, which would make the result depend on run order.
    """

    terms = [_round_half_up(b) for b in betas]
    counts = Counter(terms)
    best = max(counts.values())
    return float(min(term for term, n in counts.items() if n == best))


OPERATORS: dict[str, Operator] = {
    "continuous_mean": op_continuous_mean,
    "two_tuple": op_two_tuple,
    "mean_round": op_mean_round,
    "median_round": op_median_round,
    "majority": op_majority,
}

# `continuous_mean` is not listed: it is numerically identical to `two_tuple`
# and would add a duplicate row to every table. It exists to be named in the
# text and pinned by a test.
LOSSLESS = ("two_tuple",)
SNAPPING = ("mean_round", "median_round", "majority")

# Where an operator is allowed to act.
#
#   review      run -> review only; every level above carries beta forward as a
#               continuous value. This is the reported pipeline.
#   all_levels  the operator also replaces the mean at review -> expert and
#               expert -> alternative, so the collective evaluation itself
#               lands on a term. This is what an applied study that used
#               majority vote "throughout" would actually be doing, and it is
#               reported as a contrast rather than left implicit.
SCOPES = ("review", "all_levels")


def apply_operator(
    samples: np.ndarray, operator: str, scale: LinguisticScale
) -> np.ndarray:
    """Row-wise operator over a `(replicas, n_runs)` matrix of betas.

    Same semantics as the scalar operators above, including the tie rule of
    `op_majority`; `tests/test_cdm_pipeline.py` pins the two implementations to
    each other. The vectorized form exists because the resampling analyses
    evaluate an operator millions of times, and the scalar one makes thousands
    of replicas cost minutes instead of seconds.
    """

    if operator in ("two_tuple", "continuous_mean"):
        return samples.mean(axis=1)
    if operator == "mean_round":
        return np.floor(samples.mean(axis=1) + 0.5)
    if operator == "median_round":
        return np.floor(np.median(samples, axis=1) + 0.5)
    if operator == "majority":
        terms = np.floor(samples + 0.5).astype(int)
        # Counts per term, then argmax -- which returns the *first* maximum,
        # i.e. the lowest term index, matching op_majority's tie rule.
        counts = (terms[:, :, None] == np.arange(scale.g + 1)[None, None, :]).sum(
            axis=1
        )
        return counts.argmax(axis=1).astype(float)
    raise ValueError(f"unknown operator {operator!r}; expected one of {[*OPERATORS]}")


def fuse_runs(runs: pd.DataFrame, operator: str = "two_tuple") -> pd.DataFrame:
    """Within-model fusion: (model, scheme, review) -> one beta.

    `runs` must already carry the expert and alternative columns; see
    `utils.cdm.attach_entities`.
    """

    fn = OPERATORS[operator]
    rows = []
    for (model, scheme, review_id), g in runs.groupby(
        ["model", "scheme", "review_id"], sort=False
    ):
        scale = scale_for(scheme)
        betas = [beta_from_y(y, scale) for y in g["y"]]
        rows.append(
            {
                "operator": operator,
                "model": model,
                "scheme": scheme,
                "review_id": review_id,
                "user_id": g["user_id"].iloc[0],
                "entity_id": g["entity_id"].iloc[0],
                "alternative": g["alternative"].iloc[0],
                "beta": fn(betas, scale),
                "n_runs": len(betas),
            }
        )
    return pd.DataFrame(rows)


def fuse_models(
    reviews: pd.DataFrame,
    operator: str = "two_tuple",
    models: Sequence[str] | None = None,
    label: str = "Fusion",
) -> pd.DataFrame:
    """Across-model fusion: the per-model betas of a review -> one beta.

    Returns a frame shaped like its input, with `model` set to `label`, so a
    fused system can be appended to the per-model frame and carried through the
    rest of the pipeline as if it were another model.

    `models` restricts which systems take part; the default is every model
    present. The reported fusion uses the open-weight systems only -- folding
    the proprietary reference into the ensemble would make it a different
    claim.
    """

    fn = OPERATORS[operator]
    if models is not None:
        reviews = reviews[reviews["model"].isin(models)]

    rows = []
    for (scheme, review_id), g in reviews.groupby(["scheme", "review_id"], sort=False):
        scale = scale_for(scheme)
        rows.append(
            {
                "operator": operator,
                "model": label,
                "scheme": scheme,
                "review_id": review_id,
                "user_id": g["user_id"].iloc[0],
                "entity_id": g["entity_id"].iloc[0],
                "alternative": g["alternative"].iloc[0],
                "beta": fn(list(g["beta"]), scale),
                "n_models": len(g),
            }
        )
    return pd.DataFrame(rows)
