"""Tests for task_loader.py: loading a task YAML file."""

import pytest

from model_selection_lab.runs.task_loader import find_task_file, load_task, parse_task

VALID_YAML = """
name: tiny_task
description: A good answer is short.
instruction: Summarise the text in one sentence.
inputs:
  - id: easy_01
    difficulty: easy
    input: The cat sat on the mat.
    reference: null
"""

# Same file without the "instruction" field.
MISSING_INSTRUCTION_YAML = """
name: tiny_task
description: A good answer is short.
inputs:
  - id: easy_01
    difficulty: easy
    input: The cat sat on the mat.
    reference: null
"""


def test_load_task_valid_file(tmp_path):
    path = tmp_path / "task.yaml"
    path.write_text(VALID_YAML, encoding="utf-8")

    task = load_task(path)

    assert task.name == "tiny_task"
    assert task.instruction == "Summarise the text in one sentence."
    assert len(task.inputs) == 1
    assert task.inputs[0].id == "easy_01"
    assert task.inputs[0].difficulty == "easy"
    assert task.inputs[0].input == "The cat sat on the mat."
    assert task.inputs[0].reference is None


def test_load_task_missing_field_names_it(tmp_path):
    path = tmp_path / "task.yaml"
    path.write_text(MISSING_INSTRUCTION_YAML, encoding="utf-8")

    with pytest.raises(ValueError, match="missing the field 'instruction'"):
        load_task(path)


def test_find_task_file_looks_in_each_folder(tmp_path):
    (tmp_path / "generated").mkdir()
    (tmp_path / "generated" / "drafted_one.yaml").write_text(VALID_YAML, encoding="utf-8")
    folders = (str(tmp_path), str(tmp_path / "generated"))

    assert find_task_file("drafted_one", folders).endswith("generated/drafted_one.yaml")
    with pytest.raises(FileNotFoundError, match="No task file named missing.yaml"):
        find_task_file("missing", folders)


def test_reference_difficulty_and_checks_are_optional(tmp_path):
    task = parse_task(
        {"name": "t", "description": "d", "instruction": "i", "inputs": [{"id": "a", "input": "x"}]},
        "test",
    )

    assert task.checks == {}
    assert task.inputs[0].checks == {}
    assert task.inputs[0].reference is None and task.inputs[0].difficulty == ""


def test_checks_are_read_for_the_task_and_for_each_input():
    task = parse_task(
        {
            "name": "t", "description": "d", "instruction": "i",
            "checks": {"max_words": 50},
            "inputs": [{"id": "a", "input": "x", "checks": {"must_include": ["17%"]}}],
        },
        "test",
    )

    assert task.checks == {"max_words": 50}
    assert task.inputs[0].checks == {"must_include": ["17%"]}


def test_a_misspelt_check_in_a_task_is_an_error_naming_the_input():
    data = {
        "name": "t", "description": "d", "instruction": "i",
        "inputs": [{"id": "a", "input": "x", "checks": {"max_word": 5}}],
    }

    with pytest.raises(ValueError, match="Input 1"):
        parse_task(data, "test")
