"""Tests for task_loader.py: loading a task YAML file."""

import pytest

from model_selection_lab.runs.task_loader import find_task_file, load_task

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
