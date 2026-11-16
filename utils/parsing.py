"""Parsing policy: from a raw model reply to a value on [-1, 1].

A score must be a single number already inside [-1, 1]; out-of-range values are
rejected, not clamped. An ordinal reply goes through two stages, in order:

1. **Unwrap.** Strip markdown emphasis, a leading ``Sentiment:`` prefix and
   backslash escapes, then require an exact vocabulary match. This is pure
   formatting normalization.
2. **Extract.** Failing that, look for every canonical label inside the
   reply. Exactly one category present: accepted. None: rejected as
   ``no_label``. Two or more distinct categories (``neutral to
   very_positive``): rejected as ``ambiguous``, never resolved by guessing.

Stage 2 matches **longest label first** and anchors on word boundaries, so
``very_positive`` is always preferred over ``positive``, and a matched label is
consumed before shorter ones are searched, so ``very positive`` counts as one
category, not two. Tests pin both.

This module is the single source of truth for the label vocabularies. The
lenient parser in `experiments.parsing` only feeds the console while runs are
being collected.
"""

from __future__ import annotations

import re
from collections.abc import Callable

LABEL_5_VALUES = ["very_positive", "positive", "neutral", "negative", "very_negative"]
LABEL_7_VALUES = [
    "extremely_positive",
    "very_positive",
    "positive",
    "neutral",
    "negative",
    "very_negative",
    "extremely_negative",
]

# Labels are sorted from positive to negative; ordinal 0 is very negative.
ORDINAL_5 = {
    label: abs(i - (len(LABEL_5_VALUES) - 1)) for i, label in enumerate(LABEL_5_VALUES)
}
ORDINAL_7 = {
    label: abs(i - (len(LABEL_7_VALUES) - 1)) for i, label in enumerate(LABEL_7_VALUES)
}


def normalize_label(raw: str) -> str:
    return raw.strip().strip("*").strip().lower().replace(" ", "_").replace("-", "_")


def normalize_label7(ordinal: int) -> float:
    """Map ordinal [0..6] to [-1..1]."""

    return (ordinal - 3) / 3


def normalize_label5(ordinal: int) -> float:
    """Map ordinal [0..4] to [-1..1]."""

    return (ordinal - 2) / 2


def normalize_score(score: float) -> float:
    """Scores are already on [-1, 1]."""

    return float(score)


def parse_score(raw: str | None) -> float | None:
    """The reply must be a single number already inside [-1, 1].

    More than one number is rejected rather than resolved by position: a reply
    such as "on a scale from -1 to 1, I'd say 0.8" would otherwise be read as
    -1. Every one of the 14,220 reported SCORE outputs carries exactly one
    number, so this rejects nothing that was previously accepted.
    """

    if not raw:
        return None
    numbers = re.findall(r"-?\d+\.?\d*", raw)
    if len(numbers) != 1:
        return None
    value = float(numbers[0])
    return value if -1.0 <= value <= 1.0 else None


# A reply may open with "Sentiment:", "**Sentiment:**", "sentiment :" and so on.
_PREFIX = re.compile(r"^\W*sentiment\W*:\s*", re.IGNORECASE)


def unwrap(raw: str) -> str:
    """Remove formatting that carries no meaning: emphasis, prefix, escapes."""

    text = raw.strip()
    text = text.replace("*", "").replace("\\", "").replace("`", "")
    text = _PREFIX.sub("", text).strip()
    return text.strip(" .:;\"'")


def _sorted_labels(labels: list[str]) -> list[str]:
    """Vocabulary longest first.

    The ordering is the whole safety property of stage 2. `very_positive`
    contains `positive`, so scanning in vocabulary order finds the weaker label
    inside the stronger one and drops the intensifier.
    """

    return sorted(labels, key=len, reverse=True)


def classify_label(raw: str | None, labels: list[str]) -> tuple[str | None, str]:
    """Canonical label of a reply and why it was accepted or rejected.

    Returns ``(label, "exact")`` or ``(label, "embedded")`` when exactly one
    category is present, ``(None, "no_label")`` when none is, and
    ``(None, "ambiguous")`` when two or more distinct categories are. A reply
    such as ``neutral to **very_positive**`` names two points of the scale;
    picking either would be a guess, so it is rejected rather than read.
    """

    if not raw:
        return None, "no_label"

    # Stage 1: formatting only.
    unwrapped = normalize_label(unwrap(raw))
    if unwrapped in labels:
        return unwrapped, "exact"

    # Stage 2: every category that appears inside a longer reply.
    #
    # Not `normalize_label` here: that maps spaces to underscores, so a whole
    # sentence collapses into one underscore-joined token and no word boundary
    # survives to anchor against. Instead both sides are reduced to
    # space-separated words, which makes "very positive", "very_positive" and
    # "very-positive" the same string, and the boundary meaningful.
    #
    # Longest label first, and each match is consumed (blanked) before the
    # shorter labels are searched, so the `positive` inside `very positive` is
    # not counted as a second category.
    haystack = f" {_words(unwrap(raw))} "
    found: list[str] = []
    for label in _sorted_labels(labels):
        pattern = r"\s+".join(re.escape(w) for w in label.split("_"))
        regex = re.compile(rf"(?<=\s){pattern}(?=\s)")
        if regex.search(haystack):
            found.append(label)
            haystack = regex.sub(lambda m: " " * len(m.group(0)), haystack)
    if len(found) == 1:
        return found[0], "embedded"
    if not found:
        return None, "no_label"
    return None, "ambiguous"


def extract_label(raw: str | None, labels: list[str]) -> str | None:
    """Canonical label of a reply, or None if none or several are present."""

    return classify_label(raw, labels)[0]


def _words(text: str) -> str:
    """Lowercase, every non-alphanumeric run collapsed to a single space."""

    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def parse_label7(raw: str | None) -> tuple[float | None, str | None]:
    return _parse(raw, LABEL_7_VALUES, ORDINAL_7, normalize_label7)


def parse_label5(raw: str | None) -> tuple[float | None, str | None]:
    return _parse(raw, LABEL_5_VALUES, ORDINAL_5, normalize_label5)


def _parse(
    raw: str | None,
    labels: list[str],
    ordinal_map: dict[str, int],
    norm_fn: Callable[[int], float],
) -> tuple[float | None, str | None]:
    label = extract_label(raw, labels)
    if label is None:
        return None, None
    return norm_fn(ordinal_map[label]), label


# Reported scheme -> (experiment directory inside the archives, parser).
SCHEMES: dict[str, tuple[str, Callable]] = {
    "SCORE": ("single_score_v2_t1", parse_score),
    "LABEL-7": ("single_label_7c_t1", parse_label7),
    "LABEL-5": ("single_label_5c_t1", parse_label5),
}

# The temperature contrast. Same prompt, same vocabulary, same parser, same
# reviews: only the sampling temperature differs, so a difference between the
# two collections is attributable to sampling and nothing else. LABEL-5 only --
# the question it answers is whether the collapse onto one term survives greedy
# decoding, and that question is asked of the headline scheme.
SCHEMES_GREEDY: dict[str, tuple[str, Callable]] = {
    "LABEL-5": ("single_label_5c_t0", parse_label5),
}
