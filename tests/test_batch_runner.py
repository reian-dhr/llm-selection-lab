"""Tests for batch_runner.py: the batch loop. A fake run_model is used, so no network."""

import json
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

from model_selection_lab.models.call_result import RunResult
from model_selection_lab.models.model_registry import ModelInfo
from model_selection_lab.runs.batch_runner import build_prompt, estimate_max_cost, run_task
from model_selection_lab.runs.results_storage import load_reusable_answers
from model_selection_lab.runs.task_loader import Task, TaskInput
from model_selection_lab.settings import (
    MAX_ANSWER_TOKENS,
    THINKING_EXTRA_TOKENS,
    TOKENS_PER_WORD_ESTIMATE,
)

TASK = Task(
    name="tiny_task",
    description="A good answer is short.",
    instruction="Summarise the facts in one sentence.",
    inputs=[
        TaskInput(id="easy_01", difficulty="easy", input="The cat sat.", reference=None),
        TaskInput(id="hard_01", difficulty="hard", input="The dog ran.", reference=None),
    ],
)


def fake_run(model_key, prompt, max_tokens):
    """Stand-in for run_model: answers at once, and crashes for the model "broken"."""
    if model_key == "broken":
        raise KeyError(model_key)
    return RunResult(
        text=f"answer from {model_key}",
        input_tokens=10,
        output_tokens=5,
        latency_s=0.1,
        cost_usd=0.001,
    )


def read_lines(path):
    """Read a .jsonl file back as a list of dictionaries, one per line."""
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def test_build_prompt_contains_instruction_and_input():
    prompt = build_prompt("Summarise the facts.", "The cat sat.")

    assert "Summarise the facts." in prompt
    assert "The cat sat." in prompt


def test_build_prompt_without_input_sends_the_instruction_alone():
    assert build_prompt("Write a haiku about rain.", "") == "Write a haiku about rain."


def test_run_task_saves_one_line_per_input_and_model(tmp_path):
    path = tmp_path / "run.jsonl"

    run_task(TASK, ["model-a", "model-b"], path, run=fake_run)

    lines = read_lines(path)
    pairs = {(line["input_id"], line["model_key"]) for line in lines}
    assert len(lines) == 4  # answers arrive in any order when calls run together
    assert pairs == {
        ("easy_01", "model-a"),
        ("easy_01", "model-b"),
        ("hard_01", "model-a"),
        ("hard_01", "model-b"),
    }
    assert {line["task_name"] for line in lines} == {"tiny_task"}
    assert {line["text"] for line in lines} == {"answer from model-a", "answer from model-b"}


def test_run_task_continues_after_a_failed_call(tmp_path):
    path = tmp_path / "run.jsonl"

    run_task(TASK, ["broken", "model-a"], path, run=fake_run)

    lines = read_lines(path)
    assert len(lines) == 4  # the 2 failed calls are saved too, and the run went on
    broken = [line for line in lines if line["model_key"] == "broken"]
    good = [line for line in lines if line["model_key"] == "model-a"]
    assert len(broken) == 2 and all("KeyError" in line["error"] for line in broken)
    assert len(good) == 2 and all(line["error"] is None for line in good)


def test_run_task_stops_when_spending_reaches_the_cap(tmp_path, capsys):
    path = tmp_path / "run.jsonl"

    # Each fake call costs $0.001, so a cap of $0.002 allows exactly 2 of the 4 calls.
    run_task(
        TASK, ["model-a", "model-b"], path, run=fake_run, max_cost_usd=0.002, parallel_calls=1
    )

    assert len(read_lines(path)) == 2
    assert "STOPPED" in capsys.readouterr().out


def test_estimate_max_cost_assumes_full_length_answers():
    models = {
        "plain": SimpleNamespace(
            model_id="plain-id", reasoning=False,
            input_price_per_mtok=1.0, output_price_per_mtok=10.0,
        ),
        "thinker": SimpleNamespace(
            model_id="thinker-id", reasoning=True,
            input_price_per_mtok=1.0, output_price_per_mtok=10.0,
        ),
    }
    input_tokens = sum(
        len(build_prompt(TASK.instruction, task_input.input).split()) * TOKENS_PER_WORD_ESTIMATE
        for task_input in TASK.inputs
    )
    input_cost = input_tokens / 1_000_000 * 1.0
    plain_output_cost = len(TASK.inputs) * MAX_ANSWER_TOKENS / 1_000_000 * 10.0
    thinker_output_cost = (
        len(TASK.inputs) * (MAX_ANSWER_TOKENS + THINKING_EXTRA_TOKENS) / 1_000_000 * 10.0
    )

    assert estimate_max_cost(TASK, ["plain"], models) == pytest.approx(
        input_cost + plain_output_cost
    )
    # A model that thinks is assumed to use its whole thinking allowance too.
    assert estimate_max_cost(TASK, ["thinker"], models) == pytest.approx(
        input_cost + thinker_output_cost
    )


def test_a_second_run_reuses_saved_answers_and_makes_no_call(tmp_path):
    calls = []

    def counting_run(model_key, prompt, max_tokens):
        calls.append(model_key)
        return fake_run(model_key, prompt, max_tokens)

    run_task(TASK, ["model-a", "model-b"], tmp_path / "tiny_task_1.jsonl", run=counting_run)
    assert len(calls) == 4

    reusable = load_reusable_answers(tmp_path, "tiny_task", max_age_days=30)
    second_path = tmp_path / "tiny_task_2.jsonl"
    run_task(TASK, ["model-a", "model-b"], second_path, run=counting_run, reusable=reusable)

    assert len(calls) == 4  # no new call
    assert len(read_lines(second_path)) == 4  # but the new file is complete
    assert estimate_max_cost(TASK, ["model-a"], {}, reusable=reusable) == 0.0


def test_a_new_model_is_the_only_one_called(tmp_path):
    calls = []

    def counting_run(model_key, prompt, max_tokens):
        calls.append(model_key)
        return fake_run(model_key, prompt, max_tokens)

    run_task(TASK, ["model-a"], tmp_path / "tiny_task_1.jsonl", run=fake_run)
    reusable = load_reusable_answers(tmp_path, "tiny_task", max_age_days=30)

    second_path = tmp_path / "tiny_task_2.jsonl"
    run_task(TASK, ["model-a", "model-c"], second_path, run=counting_run, reusable=reusable)

    assert calls == ["model-c", "model-c"]  # one call per input, for the new model only


def test_saved_answers_are_not_reused_when_the_prompt_changed_or_they_failed(tmp_path):
    run_task(TASK, ["model-a", "broken"], tmp_path / "tiny_task_1.jsonl", run=fake_run)
    reusable = load_reusable_answers(tmp_path, "tiny_task", max_age_days=30)

    # The failed calls of "broken" are not kept.
    assert {key[1] for key in reusable} == {"model-a"}
    # Too old: nothing is reusable.
    assert load_reusable_answers(tmp_path, "tiny_task", max_age_days=-1) == {}

    # Same task with a reworded instruction: the saved answers no longer match.
    changed = Task(
        name=TASK.name,
        description=TASK.description,
        instruction="Summarise the facts in two sentences.",
        inputs=TASK.inputs,
    )
    calls = []

    def counting_run(model_key, prompt, max_tokens):
        calls.append(model_key)
        return fake_run(model_key, prompt, max_tokens)

    second_path = tmp_path / "tiny_task_2.jsonl"
    run_task(changed, ["model-a"], second_path, run=counting_run, reusable=reusable)

    assert len(calls) == 2


def test_saved_answers_are_not_reused_when_the_model_behind_a_key_changed(tmp_path):
    old = ModelInfo(
        provider="anthropic", model_id="model-v1", input_price_per_mtok=1.0,
        output_price_per_mtok=1.0, docs_url="", date_checked="", reasoning=False,
    )
    new = replace(old, model_id="model-v2")  # the same key now points at a newer model
    run_task(TASK, ["m"], tmp_path / "tiny_task_1.jsonl", run=fake_run, models={"m": old})
    reusable = load_reusable_answers(tmp_path, "tiny_task", max_age_days=30)
    calls = []

    def counting_run(model_key, prompt, max_tokens):
        calls.append(model_key)
        return fake_run(model_key, prompt, max_tokens)

    second_path = tmp_path / "tiny_task_2.jsonl"
    run_task(TASK, ["m"], second_path, run=counting_run, reusable=reusable, models={"m": new})

    assert len(calls) == 2  # both inputs asked again, nothing reused
    assert estimate_max_cost(TASK, ["m"], {"m": new}, reusable=reusable) > 0


def test_run_task_saves_the_model_id_and_prices_with_each_answer(tmp_path):
    model = ModelInfo(
        provider="anthropic", model_id="model-v1", input_price_per_mtok=0.5,
        output_price_per_mtok=2.0, docs_url="", date_checked="", reasoning=False,
    )
    path = tmp_path / "run.jsonl"

    run_task(TASK, ["m"], path, run=fake_run, models={"m": model})

    line = read_lines(path)[0]
    assert line["model_id"] == "model-v1"
    assert (line["input_price_per_mtok"], line["output_price_per_mtok"]) == (0.5, 2.0)


def test_calls_run_at_the_same_time(tmp_path):
    # Each call waits for another one to arrive. That only works if two run together.
    meeting = threading.Barrier(2, timeout=5)

    def waiting_run(model_key, prompt, max_tokens):
        meeting.wait()
        return fake_run(model_key, prompt, max_tokens)

    path = tmp_path / "run.jsonl"

    run_task(TASK, ["model-a"], path, run=waiting_run, parallel_calls=2)  # 2 inputs, 1 model

    assert [line["error"] for line in read_lines(path)] == [None, None]


def test_a_cap_reached_early_stops_calls_that_have_not_started(tmp_path, capsys):
    # One worker, so each call starts after the previous one finished and was counted.
    path = tmp_path / "run.jsonl"

    run_task(TASK, ["model-a", "model-b"], path, run=fake_run, max_cost_usd=0.001, parallel_calls=1)

    assert len(read_lines(path)) == 1
    assert "STOPPED" in capsys.readouterr().out


def test_run_task_saves_the_exact_prompt_sent_with_each_answer(tmp_path):
    path = tmp_path / "run.jsonl"

    run_task(TASK, ["model-a"], path, run=fake_run)

    prompts = {line["input_id"]: line["prompt"] for line in read_lines(path)}
    assert prompts["easy_01"] == build_prompt(TASK.instruction, "The cat sat.")
