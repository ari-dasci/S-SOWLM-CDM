"""The concurrent runner writes the same collection as the sequential one.

`--workers` only changes how many requests are in flight. The set of files,
their layout under `--results-dir` and the shape of each record must not
depend on it, or a collection's structure would depend on how fast it ran.
"""

import json
import logging
import threading
import time

import pandas as pd

from experiments import load_experiments
from sentiment_prompt_runner import _CONFIG_DIR, _run_single_experiment


class FakeBackend:
    """Answers every request with a fixed label, slowly enough to overlap."""

    def __init__(self):
        self.in_flight = 0
        self.peak = 0
        self._lock = threading.Lock()

    def chat(self, model, messages, temperature, max_tokens, think):
        with self._lock:
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
        time.sleep(0.005)
        with self._lock:
            self.in_flight -= 1
        return "positive"


def _collect(root, workers):
    exp = load_experiments(_CONFIG_DIR / "experiments.yaml")["single_label_5c_t1"]
    df = pd.DataFrame(
        {
            "REVIEW_BODY": [f"review text {i}" for i in range(20)],
            "USER_ID": [f"u{i}" for i in range(20)],
            "ENTITY_ID": ["e1"] * 20,
        }
    )
    backend = FakeBackend()
    _run_single_experiment(
        exp,
        backend,
        "fake:1b",
        df,
        2,
        "input.csv",
        logging.getLogger("test"),
        results_dir=root,
        workers=workers,
    )
    files = sorted(p.relative_to(root) for p in root.rglob("*.json"))
    return files, backend.peak


def _shape(record):
    if isinstance(record, dict):
        return {k: _shape(v) for k, v in record.items() if k not in ("timestamp",)}
    if isinstance(record, list):
        return [_shape(v) for v in record]
    return type(record).__name__


def test_four_workers_write_the_same_files_as_one(tmp_path):
    seq_files, seq_peak = _collect(tmp_path / "seq", 1)
    par_files, par_peak = _collect(tmp_path / "par", 4)

    assert len(seq_files) == 40
    assert seq_files == par_files
    assert seq_peak == 1
    assert par_peak > 1

    for rel in seq_files:
        a = json.loads((tmp_path / "seq" / rel).read_text())
        b = json.loads((tmp_path / "par" / rel).read_text())
        assert _shape(a) == _shape(b)
        assert a["review_idx"] == b["review_idx"]
        assert a["run_number"] == b["run_number"]
        assert a["history"] == b["history"]
        assert a["experiment"]["config"]["workers"] == 1
        assert b["experiment"]["config"]["workers"] == 4


def test_a_resumed_concurrent_collection_only_fills_the_gaps(tmp_path):
    files, _ = _collect(tmp_path, 4)
    victim = tmp_path / files[7]
    victim.unlink()
    kept = (tmp_path / files[0]).stat().st_mtime_ns

    again, _ = _collect(tmp_path, 4)

    assert again == files
    assert (tmp_path / files[0]).stat().st_mtime_ns == kept
