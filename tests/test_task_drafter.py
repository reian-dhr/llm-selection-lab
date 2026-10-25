"""Tests for runs/task_drafter.py: turning a description into a task. No network."""

import pytest

from model_selection_lab.models.call_result import RunResult
from model_selection_lab.runs.task_drafter import (
    build_drafter_prompt,
    draft_task,
    safe_file_name,
    save_drafted_task,
)
from model_selection_lab.runs.task_loader import load_task

TEMPLATE = "Write {{NUMBER_OF_INPUTS}} inputs for this description: {{DESCRIPTION}}"

GOOD_ANSWER = """name: follow-up email
understood:
  goal: Write a follow-up email after a meeting.
  audience: A client.
  output_format: Subject and body.
  constraints: none stated
  assumptions: none
open_questions:
  - How formal should the tone be?
description: |
  A good email states the purpose first.
instruction: |
  Write a follow-up email from the notes below.
inputs:
  - id: easy_01
    difficulty: easy
    input: |
      Meeting on budget; next step is a call on Friday.
    reference: null
"""


def answer(text):
    """A successful fake call returning the given text."""
    return RunResult(text=text, input_tokens=10, output_tokens=10, latency_s=0.1, cost_usd=0.001)


def scripted_run(texts, prompts_seen):
    """A fake run_model that plays back answers in order and records the prompts it got."""
    remaining = list(texts)

    def run(model_key, prompt, max_tokens):
        prompts_seen.append(prompt)
        return answer(remaining.pop(0))

    return run


def test_build_drafter_prompt_fills_both_placeholders():
    prompt = build_drafter_prompt(TEMPLATE, "  Draft emails.  ", 3)

    assert prompt == "Write 3 inputs for this description: Draft emails."


def test_draft_task_reads_a_valid_answer_even_inside_code_fences():
    prompts = []
    run = scripted_run(["```yaml\n" + GOOD_ANSWER + "```"], prompts)

    data, task, cost = draft_task("Draft emails.", run=run, template=TEMPLATE)

    assert task.instruction.startswith("Write a follow-up email")
    assert task.inputs[0].id == "easy_01"
    assert data["open_questions"] == ["How formal should the tone be?"]
    assert len(prompts) == 1
    assert cost == pytest.approx(0.001)


def test_draft_task_retries_once_with_the_problem_explained():
    prompts = []
    missing_instruction = GOOD_ANSWER.replace("instruction:", "instructions:")
    run = scripted_run([missing_instruction, GOOD_ANSWER], prompts)

    data, task, cost = draft_task("Draft emails.", run=run, template=TEMPLATE)

    assert len(prompts) == 2
    assert "missing the field 'instruction'" in prompts[1]
    assert cost == pytest.approx(0.002)


def test_draft_task_gives_up_after_two_invalid_answers():
    prompts = []
    run = scripted_run(["not: [valid", "just a sentence"], prompts)

    with pytest.raises(ValueError, match="did not write a valid task"):
        draft_task("Draft emails.", run=run, template=TEMPLATE)


def test_safe_file_name_keeps_only_simple_characters():
    assert safe_file_name("Follow-up Email / v2!") == "follow_up_email_v2"
    assert safe_file_name("???") == "task"


def test_saved_draft_loads_as_a_normal_task(tmp_path):
    data, _, _ = draft_task("Draft emails.", run=scripted_run([GOOD_ANSWER], []), template=TEMPLATE)

    path = save_drafted_task(data, folder=tmp_path)

    task = load_task(path)
    assert task.name.startswith("follow_up_email_")
    assert path.endswith(f"{task.name}.yaml")  # the file is named after the task
    assert task.inputs[0].input.startswith("Meeting on budget")


def test_context_is_added_to_the_description():
    prompt = build_drafter_prompt(TEMPLATE, "Draft emails.", 2, context="The budget is 40 euros.")

    assert "Draft emails." in prompt
    assert "Context provided by the person" in prompt
    assert "The budget is 40 euros." in prompt


def test_saved_draft_writes_long_texts_as_blocks(tmp_path):
    data, _, _ = draft_task("Draft emails.", run=scripted_run([GOOD_ANSWER], []), template=TEMPLATE)

    text = open(save_drafted_task(data, folder=tmp_path), encoding="utf-8").read()

    assert "instruction: |" in text
    assert "  input: |" in text


def test_draft_task_reads_a_json_answer_with_colons_in_texts():
    json_answer = (
        '{"name": "slide", "understood": {"output_format": "Slide: a title: then bullets"},'
        ' "open_questions": [], "description": "Title: states the takeaway.",'
        ' "instruction": "Draft a slide.\\nFormat: TITLE then BULLETS.",'
        ' "inputs": [{"id": "easy_01", "difficulty": "easy", "input": "Facts: A, B.",'
        ' "reference": null}]}'
    )

    data, task, _ = draft_task("Slides.", run=scripted_run([json_answer], []), template=TEMPLATE)

    assert data["understood"]["output_format"] == "Slide: a title: then bullets"
    assert task.instruction == "Draft a slide.\nFormat: TITLE then BULLETS."
