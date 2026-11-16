"""Tests for the lenient collection-side parser (`experiments.parsing`).

The parser behind the reported numbers is `utils.parsing`, covered at the end.
These tests import the shipped code directly. An earlier version of this file
kept a hand-copied snapshot of the parser, which had already drifted from the
original and could not catch a regression.
"""

import pytest

from experiments.parsing import ResponseType, parse_response


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("0.5", 0.5),
        ("-0.5", -0.5),
        ("The sentiment score is 0.75", 0.75),
        (".8", 0.8),
        # Out of range is clamped here; the strict parser rejects instead.
        ("1.4", 1.0),
        ("-3", -1.0),
        # Only the first number is read, then clamped; no /10 rescaling on
        # this path, so "8/10" becomes 1.0 rather than 0.8.
        ("8/10", 1.0),
    ],
)
def test_score_parsing_is_lenient(raw, expected):
    parsed = parse_response(raw, ResponseType.SCORE)

    assert parsed.valid
    assert parsed.value == pytest.approx(expected)


def test_score_parsing_rejects_text_without_a_number():
    parsed = parse_response("no numeric answer here", ResponseType.SCORE)

    assert not parsed.valid


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("very_positive", "very_positive"),
        ("**neutral**", "neutral"),
        # Matching is by substring, longest label first, so surrounding prose
        # is tolerated and does not shadow the more specific label.
        ("I would say very_positive overall", "very_positive"),
    ],
)
def test_label_5_substring_matching(raw, expected):
    parsed = parse_response(raw, ResponseType.LABEL_5)

    assert parsed.valid
    assert parsed.value == expected


@pytest.mark.parametrize("raw", ["Very Positive", "very-positive"])
def test_label_5_does_not_normalize_separators(raw):
    """Separator variants silently degrade to the shorter label.

    The canonical form uses underscores; with a space or hyphen the
    "very_positive" needle no longer matches, so the substring scan falls
    through to plain "positive". This is precisely the laxity that keeps this
    parser out of the paper. `utils.parsing.extract_label` normalizes
    separators first and returns the intended label.
    """

    parsed = parse_response(raw, ResponseType.LABEL_5)

    assert parsed.valid
    assert parsed.value == "positive"


def test_label_5_rejects_unknown_label():
    parsed = parse_response("excellent", ResponseType.LABEL_5)

    assert not parsed.valid


def test_lenient_and_reported_parsers_disagree_as_documented():
    """The boundary the two parsing modules are built around."""

    from utils.parsing import LABEL_5_VALUES, extract_label, parse_score

    # Out-of-range score: clamped by the collection parser, rejected by the
    # one behind the reported numbers.
    assert parse_response("1.4", ResponseType.SCORE).value == 1.0
    assert parse_score("1.4") is None

    # A label outside the vocabulary is rejected; separators are normalized.
    assert extract_label("excellent", LABEL_5_VALUES) is None
    assert extract_label("very-positive", LABEL_5_VALUES) == "very_positive"


# --- Replies naming more than one category --------------------------------
#
# The thirteen locally served Ministral 3 8B replies that name more than one
# category. The earlier longest-first extractor returned the first label it
# found; each of these must now be rejected as ambiguous.

from utils.parsing import (  # noqa: E402
    LABEL_5_VALUES,
    LABEL_7_VALUES,
    classify_label,
    extract_label,
)

AMBIGUOUS_MINISTRAL = [
    # Ministral 3 8B local, single_label_7c_t1, review-22, run 1; previously read as very_positive
    pytest.param(
        '**neutral** (Note: The text does contain elements of enjoyment and pleasure, but the mention of expense ("not cheap") introduces some reservation or pragmatic consideration.) Corrected with the appropriate single label:\nVery_positive',
        LABEL_7_VALUES,
        id="7c_t1-review-22-r1",
    ),
    # Ministral 3 8B local, single_label_7c_t1, review-24, run 9; previously read as positive
    pytest.param(
        '**negative**\n(Though there are positive aspects ("service was good," "view...excellent"), the inclusion of a critique ("went a few years back and had..."—implying disappointment in current experience compared to past) leans toward an overall negative tone.)*\n\n*(Correction: Upon re-reading, the phrasing implies a nostalgic but neutral/comparative sentiment with more positive emphasis. The correct label is **neutral**.)*\n\n---\n**neutral**\n(The text describes a past experience favorably while subtly suggesting the current setting fell short. The focus remains on the past’s positives, but the critique (“went back”) introduces ambiguity—too mixed for clear positive/negative. No extreme bias.)\n---',
        LABEL_7_VALUES,
        id="7c_t1-review-24-r9",
    ),
    # Ministral 3 8B local, single_label_7c_t1, review-49, run 8; previously read as very_positive
    pytest.param(
        "**neutral** (correction: after reconsideration based on context, the more accurate label is **very_positive**)",
        LABEL_7_VALUES,
        id="7c_t1-review-49-r8",
    ),
    # Ministral 3 8B local, single_label_7c_t1, review-128, run 6; previously read as positive
    pytest.param(
        '**neutral** (with slight positive undertones)\n\n*(The response leans slightly positive but includes enough objective details—such as "a bit of secret" and comparative mentions—to keep the overall tone neutral rather than definitively uplifting.)*\n\n**Final label: neutral.**',
        LABEL_7_VALUES,
        id="7c_t1-review-128-r6",
    ),
    # Ministral 3 8B local, single_label_7c_t1, review-190, run 0; previously read as very_positive
    pytest.param(
        "neutral to **very_positive**", LABEL_7_VALUES, id="7c_t1-review-190-r0"
    ),
    # Ministral 3 8B local, single_label_7c_t1, review-271, run 4; previously read as extremely_positive
    pytest.param(
        '**neutral** (The text leans slightly toward positive sentiment, but it lacks strong emotional intensity to justify labels like "very_positive" or "extremely_positive")',
        LABEL_7_VALUES,
        id="7c_t1-review-271-r4",
    ),
    # Ministral 3 8B local, single_label_7c_t1, review-300, run 2; previously read as positive
    pytest.param(
        '**Sentiment: Neutral**\n\n*(Rationale: While the text has many positive notes, the inclusion of significant issues (tough octopus and steak) and critical comparisons ("very disappointing," "could get... so wrong") skew it from being primarily positive.)*',
        LABEL_7_VALUES,
        id="7c_t1-review-300-r2",
    ),
    # Ministral 3 8B local, single_label_7c_t1, review-300, run 4; previously read as positive
    pytest.param(
        "**Sentiment: neutral**\n\n*(Note: The review mixes **mixed sentiments**—positive experiences outweigh the negative, but there is a **notable disappointment (the steak)**, keeping it from being clearly positive.)*",
        LABEL_7_VALUES,
        id="7c_t1-review-300-r4",
    ),
    # Ministral 3 8B local, single_label_7c_t1, review-323, run 4; previously read as very_negative
    pytest.param(
        "**Sentiment:** **very_negative** (due to significant dissatisfaction despite positive aspects)",
        LABEL_7_VALUES,
        id="7c_t1-review-323-r4",
    ),
    # Ministral 3 8B local, single_label_7c_t1, review-334, run 5; previously read as very_positive
    pytest.param(
        '**neutral** *(but leaning toward **very_positive**)*\n\n*Note:* While "neutral" technically includes both positive and negative components (e.g., "always looked after" vs. implied no negatives), the overall context overwhelmingly skews positive. The strict single-label requirement defaults to **"neutral"** for balance, though **very_positive** would be more accurate if forced to choose. If you\'d prefer a purely positive label in this case: **very_positive**.',
        LABEL_7_VALUES,
        id="7c_t1-review-334-r5",
    ),
    # Ministral 3 8B local, single_label_7c_t1, review-389, run 6; previously read as very_positive
    pytest.param(
        "neutral is incorrect here since the text clearly indicates positive sentiment.\nThe appropriate classification for the given text is **very_positive**.",
        LABEL_7_VALUES,
        id="7c_t1-review-389-r6",
    ),
    # Ministral 3 8B local, single_label_5c_t1, review-205, run 6; previously read as negative
    pytest.param(
        "**neutral**\n(Overall praise with minor negative detail)",
        LABEL_5_VALUES,
        id="5c_t1-review-205-r6",
    ),
    # Ministral 3 8B local, single_label_5c_t1, review-249, run 1; previously read as positive
    pytest.param(
        "**neutral**\n(Though slightly positive with qualified appreciation, the overall tone is tempered.)",
        LABEL_5_VALUES,
        id="5c_t1-review-249-r1",
    ),
]


@pytest.mark.parametrize("raw,labels", AMBIGUOUS_MINISTRAL)
def test_cdm_parser_rejects_replies_naming_two_categories(raw, labels):
    assert classify_label(raw, labels) == (None, "ambiguous")
    assert extract_label(raw, labels) is None


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("very_positive", ("very_positive", "exact")),
        ("**Very Positive.**", ("very_positive", "exact")),
        (r"Sentiment: \*\*negative\*\*", ("negative", "exact")),
        # One category, written twice: not ambiguous.
        (
            "I would say very_positive overall, very positive",
            ("very_positive", "embedded"),
        ),
        # The `positive` inside `very positive` is consumed, not a second label.
        ("The label is very positive.", ("very_positive", "embedded")),
        ("**Excellent.**", (None, "no_label")),
        ("", (None, "no_label")),
    ],
)
def test_cdm_parser_reasons(raw, expected):
    assert classify_label(raw, LABEL_5_VALUES) == expected
