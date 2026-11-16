import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import Field
from pydantic.dataclasses import dataclass

from backends import Backend
from experiments.parsing import ParsedResponse, ResponseType, parse_response


@dataclass
class ExperimentConfig:
    """Configuration loaded from YAML."""

    description: str
    system_prompt: str
    user_template: str
    response_type: ResponseType = ResponseType.LABEL_5
    temperature: float = 0.0
    max_tokens: int = 2048
    think: bool | None = None
    """Reasoning trace on reasoning-capable models; None leaves the default.

    Recorded in every run JSON, so a collection carries the setting it was
    gathered under. See `backends.base.Backend.chat` for why it matters.
    """
    workers: int | None = None
    """Concurrent requests the runner had in flight; None means one.

    Set by the runner, not by the YAML. Latency recorded under concurrency
    includes queueing at the server and is not a per-request figure.
    """

    draft_num_predict: int | None = None
    """Speculative-decoding draft length passed to Ollama; None leaves the
    model's own default. 0 disables the draft model.

    Set by the runner (`--draft-num-predict`), not by the YAML. Since the
    September 2026 tag update `gemma4:e4b` ships a draft model whose loading
    aborts under Ollama 0.31.1; the April collection had none.
    """

    def to_dict(self):
        # Runner-set fields are recorded only when set, so records collected
        # before they existed keep the exact config shape they were written
        # with.
        runner_set = ("workers", "draft_num_predict")
        return {
            k: v
            for k, v in asdict(self).items()
            if k not in runner_set or v is not None
        }


@dataclass
class ChatResult:
    """A single chat"""

    experiment_name: str
    experiment_config: ExperimentConfig
    model_id: str
    messages: list[dict[str, str]]
    response: str
    parsed: ParsedResponse
    # Metadata
    run_number: int = 0
    review_idx: str = ""
    input_file: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    latency_s: float | None = None
    """Wall-clock seconds for the backend call that produced `response`.

    Measured around `Backend.chat` in `Experiment.run`, so it covers request,
    generation and response, which is what a deployment latency figure should
    report. `None` on collections gathered before this was instrumented; the
    reader in `utils.runs` treats a missing key that way too.
    """

    def to_dict(self) -> dict[str, Any]:
        result = {
            "experiment": {
                "name": self.experiment_name,
                "config": self.experiment_config.to_dict(),
            },
            "model": self.model_id,
            "run_number": self.run_number,
            "review_idx": self.review_idx,
            "input_file": self.input_file,
            "timestamp": self.timestamp,
            "latency_s": self.latency_s,
            "history": [
                *self.messages,
                {"role": "assistant", "content": self.response},
            ],
        }
        return result

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class Experiment:
    """A runnable experiment."""

    name: str
    config: ExperimentConfig

    def build_message(self, review: str) -> list[dict[str, str]]:
        """Build the system and user messages for one review."""

        content = self.config.user_template.format(review=review)
        return [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": content},
        ]

    def run(
        self,
        backend: Backend,
        model_id: str,
        review: str,
        review_idx: str,
        run_number: int = 0,
        input_file: str = "",
    ) -> ChatResult:
        """Run the experiment on a single review."""
        messages = self.build_message(review)
        # perf_counter, not time(): a monotonic clock, so an NTP step mid-run
        # cannot produce a negative or inflated latency.
        started = time.perf_counter()
        response = backend.chat(
            model_id,
            messages,
            self.config.temperature,
            self.config.max_tokens,
            self.config.think,
            **(
                {}
                if self.config.draft_num_predict is None
                else {"draft_num_predict": self.config.draft_num_predict}
            ),
        )
        latency_s = time.perf_counter() - started

        parsed = parse_response(
            response.strip(),
            self.config.response_type,
        )

        return ChatResult(
            experiment_name=self.name,
            experiment_config=self.config,
            model_id=model_id,
            messages=messages,
            response=response.strip(),
            parsed=parsed,
            review_idx=review_idx,
            run_number=run_number,
            input_file=input_file,
            latency_s=latency_s,
        )


def load_experiments(*config_paths: Path):
    """Load experiment definitions, merging several YAML files.

    Later files win on name collisions, and paths that do not exist are
    skipped, so an extra file of your own can be dropped in beside the shipped
    `experiments.yaml` without touching the runner.
    """

    raw: dict[str, Any] = {}
    for config_path in config_paths:
        if not Path(config_path).is_file():
            continue
        with open(config_path) as f:
            raw.update(cast(dict[str, Any], yaml.safe_load(f) or {}))

    return {
        name: Experiment(name=name, config=ExperimentConfig(**config_dict))
        for name, config_dict in raw.items()
    }
