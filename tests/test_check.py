"""Tests for check.py: the whole chain, with fake models and temporary folders. No network."""

import os

from model_selection_lab.check import latest_run, list_checks, load_check, run_check, run_path_for
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


def test_a_run_file_made_by_the_helper_is_found_as_the_latest_run(tmp_path):
    path = run_path_for("short_summary_20261107_100000", tmp_path)
    tmp_path.mkdir(exist_ok=True)
    open(path, "w").close()

    assert latest_run("short_summary_20261107_100000", tmp_path) == path


def test_the_latest_run_is_the_newest_file_whatever_its_name_style(tmp_path):
    # An older file named with the time only (21:30), a newer one named with date and time.
    old = tmp_path / "task_x_run_213000.jsonl"
    new = tmp_path / "task_x_run_20261107_100000.jsonl"
    old.write_text("")
    new.write_text("")
    os.utime(old, (1_000, 1_000))
    os.utime(new, (2_000, 2_000))

    assert latest_run("task_x", tmp_path) == str(new)


def test_models_that_break_a_code_check_are_counted_and_left_out(tmp_path):
    drafted_with_rule = DRAFTED.replace(
        "inputs:", "checks:\n  must_include: [\"answer from haiku\"]\ninputs:", 1
    )

    def run(model_key, prompt, max_tokens):
        result = fake_run(model_key, prompt, max_tokens)
        if prompt.startswith("Write "):
            result.text = drafted_with_rule
        return result

    outcome = run_check(
        "Summarise texts.", run=run, template=TEMPLATE,
        tasks_folder=tmp_path / "tasks", results_folder=tmp_path / "results",
    )

    assert outcome.summary.loc["haiku-5-5", "rules_broken"] == 0
    others = outcome.summary.drop(index="haiku-5-5")
    assert (others["rules_broken"] == 1).all()
    assert any("broken rules" in line for line in outcome.recommendation)


JUDGE_VERDICT = '{"chosen": "A", "reason": "best", "critiques": {}, "refused": []}'


def judge_aware_run(model_key, prompt, max_tokens):
    """Like fake_run, but the judge (it gets THE ANSWERS in its prompt) returns a verdict."""
    result = fake_run(model_key, prompt, max_tokens)
    if "Answer A:" in prompt:
        result.text = JUDGE_VERDICT
    return result


def test_a_check_asks_the_judge_and_reports_quality(tmp_path):
    outcome = run_check(
        "Summarise texts.", run=judge_aware_run, template=TEMPLATE, judge_template="{{ANSWERS}}",
        tasks_folder=tmp_path / "tasks", results_folder=tmp_path / "results",
        verdicts_folder=tmp_path / "verdicts",
    )

    assert outcome.judge_error == ""
    assert outcome.summary["judge_wins"].sum() == 1  # one input, one winner
    assert any(line.startswith("Best quality:") for line in outcome.recommendation)


def test_a_busy_judge_leaves_the_check_without_quality_but_does_not_fail_it(tmp_path):
    def busy_judge_run(model_key, prompt, max_tokens):
        result = fake_run(model_key, prompt, max_tokens)
        if "Answer A:" in prompt:
            result.error = "RateLimitError: busy"
        return result

    outcome = run_check(
        "Summarise texts.", run=busy_judge_run, template=TEMPLATE, judge_template="{{ANSWERS}}",
        tasks_folder=tmp_path / "tasks", results_folder=tmp_path / "results",
        verdicts_folder=tmp_path / "verdicts",
    )

    assert "busy" in outcome.judge_error
    assert "judge_wins" not in outcome.summary.columns
    assert any(line.startswith("Cheapest:") for line in outcome.recommendation)


def test_a_check_saves_a_report_and_the_critic_confirms_it(tmp_path):
    outcome = run_check(
        "Summarise texts.", run=judge_aware_run, template=TEMPLATE, judge_template="{{ANSWERS}}",
        tasks_folder=tmp_path / "tasks", results_folder=tmp_path / "results",
        verdicts_folder=tmp_path / "verdicts",
    )

    assert outcome.report_path.endswith(".md")
    assert "## Per model" in open(outcome.report_path, encoding="utf-8").read()
    assert outcome.critic[0].startswith("Critic: all")
