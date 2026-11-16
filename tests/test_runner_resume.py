"""Tests for the resume guard in the collection runner.

Runs are stored under `review-<i>`, where `i` is the row position in the input
dataframe rather than anything derived from the review itself. Resuming an
interrupted collection is the intended use of that layout; pointing a rerun at a
different `--input` is not, and used to mix two reviews into one directory.
"""

import json

import pytest
import typer

from sentiment_prompt_runner import _check_resume

REVIEW_A = "UID_2069FDC1AB8797BF77F40C541B350543_d680501"
REVIEW_B = "UID_B9DF2EA33AC2F73A8D19D9BC59B81738_d680501"


def _write_run(tmp_path, review_idx):
    path = tmp_path / "run-0.json"
    path.write_text(json.dumps({"review_idx": review_idx, "history": []}))
    return str(path)


def test_resuming_the_same_review_is_allowed(tmp_path):
    """The ordinary case: an interrupted run continues where it stopped."""

    _check_resume(_write_run(tmp_path, REVIEW_A), REVIEW_A)


def test_a_different_review_in_the_slot_is_refused(tmp_path):
    """A rerun with another input maps a different review onto `review-<i>`."""

    path = _write_run(tmp_path, REVIEW_A)
    with pytest.raises(typer.BadParameter) as excinfo:
        _check_resume(path, REVIEW_B)

    # The message has to name both reviews; the whole point is telling the user
    # which collection the directory already holds.
    assert REVIEW_A in str(excinfo.value)
    assert REVIEW_B in str(excinfo.value)


def test_a_run_without_review_idx_is_allowed(tmp_path):
    """Grouped experiments store no `review_idx`; they key by group name."""

    path = tmp_path / "run-0.json"
    path.write_text(json.dumps({"history": []}))
    _check_resume(str(path), REVIEW_A)


def test_published_layout_resumes_cleanly():
    """`review-<i>` matches row `i` of the dataset in the reported collection.

    If it did not, the guard would reject a resume of the paper's own runs.
    """

    import tarfile

    import pandas as pd

    df = pd.read_csv("data/TripR-2020Large_FullReviews.csv")
    expected = [f"{r.USER_ID}_{r.ENTITY_ID}" for r in df.itertuples()]

    checked = 0
    with tarfile.open("raw/london/gemma4-e4b/single_label_7c_t1.tar.gz") as tf:
        for member in tf.getmembers():
            if not member.name.endswith("run-0.json"):
                continue
            folders = [p for p in member.name.split("/") if p.startswith("review-")]
            if not folders:
                continue
            index = int(folders[0].split("-")[1])
            stored = json.load(tf.extractfile(member))["review_idx"]
            assert stored == expected[index], folders[0]
            checked += 1

    assert checked == len(df)
