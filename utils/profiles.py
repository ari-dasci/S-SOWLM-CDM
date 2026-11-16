"""The collections the paper reports: `config/profiles.yaml`, as objects.

Every stage resolves its corpus, archives, output directory and model roster
through a profile, so the archives that get extracted and the rows that get
rendered come from the same list.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml

from .parsing import SCHEMES, SCHEMES_GREEDY

ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "config" / "profiles.yaml"


@dataclass(frozen=True)
class ModelEntry:
    """One reported system."""

    raw: str
    """Model id as the runner wrote it, colon included (`gemma4:e4b`)."""

    display: str
    """Name shown in tables and figures, and the `model` column of the runs CSV."""

    runs_csv: str
    """Filename of the long-format runs CSV built from this model's archives."""


@dataclass(frozen=True)
class Profile:
    name: str
    description: str
    models: tuple[ModelEntry, ...]

    data_csv: Path
    """The corpus the runs were collected against. Reviews join to
    alternatives by row position, so the analysis must read the same file the
    runner read."""

    raw_dir: Path
    """Directory of the raw archives, one subdirectory per model."""

    out_dir: Path
    """Where the runs CSVs and every derived file of this profile go."""

    greedy: bool = False
    """Whether this is the T = 0 collection (LABEL-5 only)."""

    reference: str = ""
    """Display name of the hosted reference system, reported beside the local
    ones and kept out of the fusion. Empty when the corpus has none."""

    stratified: bool = False
    """Also resample participants within each alternative: the sampling design
    of a corpus drawn to a fixed number of reviews per alternative."""

    @property
    def members(self) -> list[str]:
        """The fused panel: every reported system but the hosted reference."""

        return [m for m in self.model_order if m != self.reference]

    @property
    def model_order(self) -> list[str]:
        """Display names, in reported order."""

        return [m.display for m in self.models]

    @property
    def schemes(self) -> dict[str, tuple[str, Callable]]:
        """Scheme -> (archive name, parser) for this collection."""

        return SCHEMES_GREEDY if self.greedy else SCHEMES

    @property
    def scheme_dirs(self) -> tuple[str, ...]:
        """Archive names this profile reads, in reported order."""

        return tuple(dirname for dirname, _ in self.schemes.values())

    def archive(self, raw_model: str, scheme_dir: str) -> Path:
        """Path of one archive. A colon is not portable in a path, so on disk
        it is a dash: `gemma4:e4b` lives in `gemma4-e4b/`."""

        return self.raw_dir / raw_model.replace(":", "-") / f"{scheme_dir}.tar.gz"


def load_profiles(config_path: Path = CONFIG_PATH) -> dict[str, Profile]:
    with open(config_path) as f:
        raw: dict[str, dict] = yaml.safe_load(f) or {}

    return {
        name: Profile(
            name=name,
            description=(spec.get("description") or "").strip(),
            models=tuple(ModelEntry(**entry) for entry in spec["models"]),
            data_csv=ROOT / spec["data_csv"],
            raw_dir=ROOT / spec["raw_dir"],
            out_dir=ROOT / spec["out_dir"],
            greedy=bool(spec.get("greedy", False)),
            reference=spec.get("reference", ""),
            stratified=bool(spec.get("stratified", False)),
        )
        for name, spec in raw.items()
    }


def load_profile(name: str, config_path: Path = CONFIG_PATH) -> Profile:
    profiles = load_profiles(config_path)
    if name not in profiles:
        available = ", ".join(sorted(profiles))
        raise ValueError(f"Unknown profile {name!r}. Available: {available}")
    return profiles[name]
