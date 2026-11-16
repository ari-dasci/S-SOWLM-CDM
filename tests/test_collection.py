"""Tests for the collection side: registry, backends, parsing, run order.

The reproduction pipeline never touches this half of the repository, so nothing
else pins its behaviour.
"""

import json
from pathlib import Path

import pytest

from backends import BackendType
from experiments.experiments import load_experiments
from models.registry import ModelRegistry
from utils.parsing import parse_score

CONFIG_DIR = Path("config")


@pytest.fixture(scope="module")
def registry():
    return ModelRegistry(CONFIG_DIR / "models.yaml")


# ----- model registry ---------------------------------------------------


def test_unavailable_backend_raises_value_error(registry):
    """The runner catches ValueError to report which backends do work.

    `get_id` used to raise KeyError, so that branch never ran and the user got
    a traceback instead of the message.
    """

    spec = registry.get("gemma4", "e4b")
    assert BackendType("openai") not in spec.backend_ids

    with pytest.raises(ValueError) as excinfo:
        spec.get_id(BackendType("openai"))

    # The message has to name the backends that are actually configured.
    assert "ollama" in str(excinfo.value)


def test_available_backend_returns_its_id(registry):
    spec = registry.get("gemma4", "e4b")
    assert spec.get_id(BackendType("ollama")) == "gemma4:e4b"


def test_unknown_size_raises_value_error(registry):
    with pytest.raises(ValueError):
        registry.get("gemma4", "does-not-exist")


# ----- max_tokens reaches the backend -----------------------------------


class RecordingBackend:
    """Captures what `Experiment.run` passes down."""

    def __init__(self):
        self.calls = []

    def chat(self, model, messages, temperature, max_tokens=None, think=None):
        self.calls.append(
            {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "think": think,
            }
        )
        return "positive"

    def pull(self, model):
        pass


def test_max_tokens_is_passed_to_the_backend():
    """It sits in every reported experiment config; it must not be inert."""

    experiments = load_experiments(CONFIG_DIR / "experiments.yaml")
    exp = experiments["single_label_5c_t1"]
    assert exp.config.max_tokens == 2048

    backend = RecordingBackend()
    exp.run(backend, model_id="m", review="Great food.", review_idx="u_1")

    assert backend.calls[0]["max_tokens"] == 2048
    assert backend.calls[0]["temperature"] == exp.config.temperature


def test_think_is_passed_to_the_backend():
    """Every reported experiment pins it off; it must reach the provider.

    Left inert, a reasoning-capable model would silently generate a trace and
    the collection would no longer be comparable with the archived ones.
    """

    experiments = load_experiments(CONFIG_DIR / "experiments.yaml")
    for name, exp in experiments.items():
        assert exp.config.think is False, f"{name} does not pin think"

    backend = RecordingBackend()
    experiments["single_label_5c_t1"].run(
        backend, model_id="m", review="Great food.", review_idx="u_1"
    )
    assert backend.calls[0]["think"] is False


def test_backends_translate_max_tokens_to_their_own_spelling():
    """Neither provider calls it max_tokens, so the name has to be converted."""

    import inspect

    from backends.ollama import OllamaBackend
    from backends.openai import OpenAIBackend

    ollama_src = inspect.getsource(OllamaBackend.chat)
    assert "num_predict" in ollama_src

    openai_src = inspect.getsource(OpenAIBackend.chat)
    assert "max_output_tokens" in openai_src


# ----- score parsing ----------------------------------------------------


def test_a_lone_score_parses():
    assert parse_score("0.8") == 0.8
    assert parse_score("-0.5") == -0.5
    assert parse_score("0") == 0.0


def test_a_reply_with_two_numbers_is_rejected():
    """Taking the first number would read the scale bound as the answer."""

    assert parse_score("on a scale from -1 to 1, I'd say 0.8") is None
    assert parse_score("0.8 (confidence 0.9)") is None


def test_out_of_range_is_rejected_not_clamped():
    assert parse_score("1.5") is None
    assert parse_score("-2") is None


# ----- run ordering -----------------------------------------------------


def test_runs_are_read_in_numeric_order(tmp_path):
    """run-10 sorts before run-2 lexically; the archives are read numerically."""

    import tarfile
    from dataclasses import replace

    from utils.profiles import load_profile
    from utils.runs import iter_records

    review = tmp_path / "scheme" / "review-0"
    review.mkdir(parents=True)
    for n in range(12):
        payload = {"run_number": n, "review_idx": "u_1", "history": []}
        (review / f"run-{n}.json").write_text(json.dumps(payload))
    (tmp_path / "model").mkdir()
    with tarfile.open(tmp_path / "model" / "scheme.tar.gz", "w:gz") as tar:
        tar.add(tmp_path / "scheme", arcname="scheme")

    profile = replace(load_profile("london"), raw_dir=tmp_path)
    seen = [r["run_number"] for _, r in iter_records(profile, "model", "scheme")]
    assert seen == list(range(12))
