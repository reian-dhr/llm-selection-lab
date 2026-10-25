"""Tests for check.py: the whole chain, with fake models and temporary folders. No network."""

from model_selection_lab.check import list_checks, load_check, run_check
from model_selection_lab.models.call_result import RunResult

TEMPLATE = "Write {{NUMBER_OF_INPUTS}} inputs for: {{DESCRIPTION}}"

DRAFTED = """name: short_summary
understood:
  goal: Summarise a text.
open_questions: []
description: |
  One sentence, every fact kept.
instruction: |
  Summarise the text below in one sentence.
inputs:
  - id: easy_01
    difficulty: easy
    input: |
      The cat sat on the mat.
    reference: null
"""


def fake_run(model_key, prompt, max_tokens):
    """The helper gets the drafting prompt and returns a task; every other model answers."""
    if prompt.startswith("Write "):
        text = DRAFTED
    else:
        text = f"answer from {model_key}"
    return RunResult(text=text, input_tokens=10, output_tokens=5, latency_s=0.1, cost_usd=0.001)


def test_preview_drafts_the_test_but_calls_no_compared_model(tmp_path):
    outcome = run_check(
        "Summarise texts.", preview_only=True, run=fake_run, template=TEMPLATE,
        tasks_folder=tmp_path / "tasks", results_folder=tmp_path / "results",
    )

    assert outcome.task.name.startswith("short_summary_")
    assert outcome.results_path is None
    assert outcome.max_cost_usd > 0
    assert not (tmp_path / "results").exists()


def test_full_check_runs_every_model_and_recommends(tmp_path):
    outcome = run_check(
        "Summarise texts.", run=fake_run, template=TEMPLATE,
        tasks_folder=tmp_path / "tasks", results_folder=tmp_path / "results",
    )

    assert list(outcome.summary.index) == outcome.model_keys  # one row per compared model
    assert (outcome.summary["calls"] == 1).all()
    assert any(line.startswith("Cheapest:") for line in outcome.recommendation)


def test_a_past_check_can_be_listed_and_loaded_back(tmp_path):
    tasks, results = tmp_path / "tasks", tmp_path / "results"
    first = run_check(
        "Summarise texts.", run=fake_run, template=TEMPLATE,
        tasks_folder=tasks, results_folder=results,
    )

    checks = list_checks(tasks_folder=tasks, results_folder=results)
    assert [check["name"] for check in checks] == [first.task.name]
    assert checks[0]["goal"] == "Summarise a text."
    assert checks[0]["results_path"] == first.results_path

    loaded = load_check(checks[0]["task_path"], results_folder=results)
    assert loaded.results_path == first.results_path
    assert list(loaded.summary.index) == first.model_keys
