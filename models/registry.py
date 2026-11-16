from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TypedDict

import yaml

from backends import BackendType


def _load_model_families(config_path: Path) -> type[Enum]:
    """Generate ModelFamily enum from YAML keys."""
    with open(config_path) as f:
        config: dict[str, object] = yaml.safe_load(f)

    return Enum("ModelFamily", {k: k for k in config}, type=str)


# Generate at module load time
CONFIG_PATH = Path(__file__).parent.parent / "config" / "models.yaml"
ModelFamily = _load_model_families(CONFIG_PATH)


class BackendIds(TypedDict):
    ollama: str
    openai: str


class FamilyConfig(TypedDict):
    sizes: dict[str, BackendIds]
    default_size: str


class FamilyInfo(TypedDict):
    sizes: list[str]
    default: str


@dataclass
class ModelSpec:
    family: ModelFamily
    size: str
    backend_ids: dict[BackendType, str]

    @property
    def canonical_name(self) -> str:
        return f"{self.family.value}:{self.size}"

    def get_id(self, backend: BackendType) -> str:
        # ValueError, not the bare KeyError: callers catch it to turn an
        # unavailable pairing into a message listing the backends that do work.
        if backend not in self.backend_ids:
            available = ", ".join(b.value for b in self.backend_ids)
            raise ValueError(
                f"{self.canonical_name} has no {backend.value} id. "
                f"Available: {available}"
            )
        return self.backend_ids[backend]


class ModelRegistry:
    def __init__(self, config_path: Path) -> None:
        self._models: dict[str, ModelSpec] = {}
        self._families: dict[ModelFamily, FamilyInfo] = {}
        self._load(config_path)

    def _load(self, config_path: Path) -> None:
        with open(config_path) as f:
            config: dict[str, FamilyConfig] = yaml.safe_load(f)

            for family_name, family_config in config.items():
                family = ModelFamily(family_name)
                sizes = list(family_config["sizes"])
                default = family_config["default_size"]

                self._families[family] = {"sizes": sizes, "default": default}

                for size, backend in family_config["sizes"].items():
                    backend_ids: dict[BackendType, str] = {}
                    for backend_name, model_id in backend.items():
                        backend_ids[BackendType(backend_name)] = str(model_id)

                    spec = ModelSpec(family, size, backend_ids)

                    self._models[spec.canonical_name] = spec

    def available_sizes(self, family: ModelFamily) -> list[str]:
        return self._families[family]["sizes"]

    def get(self, family: ModelFamily | str, size: str | None = None) -> ModelSpec:
        if isinstance(family, str):
            family = ModelFamily(family)

        if size is None:
            size = self._families[family]["default"]

        canonical = f"{family.value}:{size}"
        if canonical not in self._models:
            available = self._families[family]["sizes"]
            raise ValueError(
                f"Size {size} not available for {family.value}. "
                f"Choose from: {available}"
            )
        return self._models[canonical]
