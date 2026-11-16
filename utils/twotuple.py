"""The 2-tuple linguistic representation model (Herrera & Martinez).

A linguistic assessment is a pair $(s_i, \\alpha)$: a term $s_i$ from an ordered
set $S = \\{s_0, \\dots, s_g\\}$ and a symbolic translation
$\\alpha \\in [-0.5, 0.5)$ giving the displacement from that term. The pair is
equivalent to a single value $\\beta \\in [0, g]$, so aggregation happens on
$\\beta$ and comes back as a 2-tuple with no rounding loss.

    delta(beta)      beta -> (s_i, alpha)     i = round(beta), alpha = beta - i
    delta_inv(t)     (s_i, alpha) -> beta     beta = i + alpha

Two domains are in play and mixing them is the easy mistake to make. The rest
of this package normalizes every scheme onto $[-1, 1]$ (`utils.parsing`), while
the 2-tuple model is defined on the term index $[0, g]$. `beta_from_y` and
`y_from_beta` convert between them; nothing here consumes a normalized value
without converting first.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .parsing import LABEL_5_VALUES, LABEL_7_VALUES


@dataclass(frozen=True)
class TwoTuple:
    """A linguistic 2-tuple: term index plus symbolic translation."""

    index: int
    alpha: float

    def __post_init__(self) -> None:
        if not -0.5 <= self.alpha < 0.5:
            raise ValueError(
                f"alpha must lie in [-0.5, 0.5), got {self.alpha!r}. "
                "Build 2-tuples with delta(), which enforces this."
            )


@dataclass(frozen=True)
class LinguisticScale:
    """An ordered term set $S = \\{s_0, \\dots, s_g\\}$.

    `terms[0]` is the most negative label, matching the ordinal convention in
    `utils.parsing` (ORDINAL_5/ORDINAL_7 assign 0 to `very_negative`).
    """

    name: str
    terms: tuple[str, ...]

    @property
    def g(self) -> int:
        """Index of the last term. The term set has $g + 1$ elements."""

        return len(self.terms) - 1

    def label(self, index: int) -> str:
        return self.terms[index]


def _titlecase(label: str) -> str:
    return " ".join(word.capitalize() for word in label.split("_"))


# LABEL_5_VALUES and LABEL_7_VALUES run positive -> negative; the term set runs
# s_0 = most negative -> s_g = most positive, so they are reversed here. Both
# scales are derived from `utils.parsing`, which stays the single source of
# truth for the vocabularies.
SCALE_5 = LinguisticScale(
    name="LABEL-5",
    terms=tuple(_titlecase(label) for label in reversed(LABEL_5_VALUES)),
)
SCALE_7 = LinguisticScale(
    name="LABEL-7",
    terms=tuple(_titlecase(label) for label in reversed(LABEL_7_VALUES)),
)

SCALE_BY_SCHEME: dict[str, LinguisticScale] = {
    "LABEL-5": SCALE_5,
    "LABEL-7": SCALE_7,
}


def _round_half_up(x: float) -> int:
    """Round to nearest, halves away from zero on the positive side.

    Not Python's `round`, which is half-to-even: at beta = i + 0.5 that lands
    on i for even i, making alpha = +0.5 and putting the tuple outside the
    half-open [-0.5, 0.5) the model requires. Rounding halves up always sends
    beta = i + 0.5 to i + 1 with alpha = -0.5, which is inside it.

    beta is confined to [0, g] here, so the negative-half case never arises.
    """

    return math.floor(x + 0.5)


def delta(beta: float, scale: LinguisticScale) -> TwoTuple:
    """$\\Delta: [0, g] \\to S \\times [-0.5, 0.5)$."""

    if not -1e-9 <= beta <= scale.g + 1e-9:
        raise ValueError(f"beta {beta!r} outside [0, {scale.g}] for {scale.name}")
    beta = min(max(beta, 0.0), float(scale.g))
    index = _round_half_up(beta)
    return TwoTuple(index=index, alpha=beta - index)


def delta_inv(tuple_: TwoTuple) -> float:
    """$\\Delta^{-1}(s_i, \\alpha) = i + \\alpha = \\beta$."""

    return tuple_.index + tuple_.alpha


def beta_from_y(y: float, scale: LinguisticScale) -> float:
    """Normalized $[-1, 1]$ value to term index $[0, g]$.

    Inverse of `utils.parsing.normalize_label5` / `normalize_label7`, which map
    ordinal to $(\\text{ordinal} - g/2)/(g/2)$.
    """

    return (y + 1.0) * scale.g / 2.0


def y_from_beta(beta: float, scale: LinguisticScale) -> float:
    """Term index $[0, g]$ back to the normalized $[-1, 1]$ domain."""

    return beta * 2.0 / scale.g - 1.0


def two_tuple_mean(betas: Sequence[float], scale: LinguisticScale) -> TwoTuple:
    """The 2-tuple arithmetic mean: $\\Delta(\\frac{1}{n}\\sum \\beta_k)$.

    This is the aggregation operator the study uses at every fusion point. It
    is exact on the continuous $\\beta$ domain, which is the property that
    distinguishes it from majority voting: no intermediate result is snapped to
    a term before the final $\\Delta$.
    """

    if not betas:
        raise ValueError("two_tuple_mean of an empty sequence")
    return delta(sum(betas) / len(betas), scale)


def format_tuple(tuple_: TwoTuple, scale: LinguisticScale, places: int = 2) -> str:
    """Render as `(Positive, +0.12)`, the form used in the tables."""

    alpha = f"{tuple_.alpha:+.{places}f}"
    # A negative translation that rounds to zero is not printed as -0.00.
    if alpha.startswith("-") and float(alpha) == 0.0:
        alpha = alpha[1:]
    return f"({scale.label(tuple_.index)}, {alpha})"
