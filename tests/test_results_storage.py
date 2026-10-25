"""Tests for results_storage.py: saving results as lines in a .jsonl file."""

import json

from model_selection_lab.models.call_result import RunResult
from model_selection_lab.runs.results_storage import append_result, make_record

SUCCESS = RunResult(
    text="Revenue grew 17%\n- bullet",
    input_tokens=120,
    output_tokens=40,
    latency_s=1.5,
    cost_usd=0.0002,
)
FAILURE = RunResult(
    text="", input_tokens=0, output_tokens=0, latency_s=0.0, cost_usd=0.0, error="timeout"
)


def read_lines(path):
    """Read a .jsonl file back as a list of dictionaries, one per line."""
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def test_make_record_copies_labels_and_result():
    record = make_record("slide_drafting", "easy_01", "some-model", SUCCESS)

    assert record.task_name == "slide_drafting"
    assert record.input_id == "easy_01"
    assert record.model_key == "some-model"
    assert record.text == SUCCESS.text
    assert record.cost_usd == SUCCESS.cost_usd
    assert record.error is None
    assert record.timestamp  # not empty


def test_append_result_adds_one_line_per_record(tmp_path):
    path = tmp_path / "results" / "run.jsonl"  # the "results" folder does not exist yet

    append_result(path, make_record("slide_drafting", "easy_01", "model-a", SUCCESS))
    append_result(path, make_record("slide_drafting", "easy_01", "model-b", FAILURE))

    lines = read_lines(path)
    assert len(lines) == 2
    assert lines[0]["model_key"] == "model-a"
    assert lines[0]["text"] == SUCCESS.text
    assert lines[0]["input_tokens"] == 120
    assert lines[1]["model_key"] == "model-b"
    assert lines[1]["error"] == "timeout"
    assert lines[0]["truncated"] is False
