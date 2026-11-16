"""Lenient parsing for the console while runs are being collected.

It salvages what it can, clamping out-of-range scores (``"1.4"`` -> ``1.0``)
and matching labels by substring, so the progress lines stay readable mid-run.
It is not the parser behind the reported numbers: those go through
`utils.parsing`, which reads the raw reply stored in every run record.
"""

import re
from dataclasses import dataclass
from enum import Enum

from utils.parsing import LABEL_5_VALUES, LABEL_7_VALUES


class ResponseType(str, Enum):
    LABEL_5 = "label_5"
    LABEL_7 = "label_7"
    SCORE = "score"


@dataclass
class ParsedResponse:
    """Parsed and validated response from LLM."""

    raw: str
    response_type: ResponseType
    value: str | float
    valid: bool = True
    error: str | None = None

    @property
    def display(self) -> str:
        """Formatted display string with Rich markup."""
        if not self.valid:
            return f"[red]INVALID: {self.raw[:50]}...[/red]"

        match self.response_type:
            case ResponseType.LABEL_5 | ResponseType.LABEL_7:
                return self._colored_label(str(self.value))
            case ResponseType.SCORE:
                return self._colored_score(float(self.value))
        return str(self.value)

    def _colored_label(self, label: str) -> str:
        """Defining colors for the sentiment labels to display into the terminal.
        There are up to 9 labels defined."""
        colors = {
            "extremely_positive": "bold bright_green",
            "very_positive": "bold green",
            "positive": "green",
            "slightly_positive": "cyan",
            "neutral": "yellow",
            "slightly_negative": "orange3",
            "negative": "red",
            "very_negative": "bold red",
            "extremely_negative": "bold bright_red",
        }
        color = colors.get(label.lower(), "white")
        return f"[{color}]{label}[/{color}]"

    def _colored_score(self, score: float) -> str:
        if score >= 0.2:
            color = "green"
        elif score >= -0.2:
            color = "yellow"
        else:
            color = "red"
        return f"[{color}]{score:.2f}[/{color}]"


# --- Main Parser ---


def parse_response(raw: str, response_type: ResponseType) -> ParsedResponse:
    """Parse LLM response based on expected type."""
    raw_clean = raw.strip()
    raw_lower = raw_clean.lower()

    match response_type:
        case ResponseType.LABEL_5:
            for label in sorted(LABEL_5_VALUES, key=len, reverse=True):
                if label in raw_lower:
                    return ParsedResponse(raw, response_type, label)
            return ParsedResponse(
                raw,
                response_type,
                raw_lower,
                valid=False,
                error=f"Expected one of {LABEL_5_VALUES}",
            )

        case ResponseType.LABEL_7:
            for label in sorted(LABEL_7_VALUES, key=len, reverse=True):
                if label in raw_lower:
                    return ParsedResponse(raw, response_type, label)
            return ParsedResponse(
                raw,
                response_type,
                raw_lower,
                valid=False,
                error=f"Expected one of {LABEL_7_VALUES}",
            )

        case ResponseType.SCORE:
            score = _extract_single_score(raw_lower)
            if score is not None:
                return ParsedResponse(raw, response_type, score)
            return ParsedResponse(
                raw,
                response_type,
                0.0,
                valid=False,
                error="Could not parse score (expected -1-1)",
            )

    return ParsedResponse(
        raw, response_type, raw_clean, valid=False, error="Unknown response type"
    )


# --- Score Parsing ---


def _extract_single_score(raw: str) -> float | None:
    """Extract a single score from text, clamped to [-1, 1].

    Clamping is the lenient policy; `utils.parsing.parse_score` rejects
    out-of-range values instead. See this module's docstring.
    """

    match = re.search(r"-?(?:\d+\.?\d*|\.\d+)", raw)
    if match:
        return max(-1.0, min(1.0, float(match.group())))
    return None
