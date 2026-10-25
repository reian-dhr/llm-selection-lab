"""Task: one type of work to test the models on, loaded from a YAML file.

A task file (see examples/tasks/example_summary.yaml) holds the instruction given to every
model and a list of inputs (the test cases). This file turns that YAML text into
Python objects, so the rest of the code can write task.instruction instead of
digging through dictionaries.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class TaskInput:
    """One test case of a task."""

    id: str  # short unique name, used to label the saved results
    difficulty: str  # "easy" or "hard"
    input: str  # the facts or text the model has to work with
    reference: str | None  # an ideal answer, None if we have not written one


@dataclass
class Task:
    """One type of work, with the instruction and all its test cases."""

    name: str
    description: str  # what a good answer looks like, used later by the judge
    instruction: str  # what every model is told
    inputs: list[TaskInput]


TASK_FIELDS = ("name", "description", "instruction", "inputs")
INPUT_FIELDS = ("id", "difficulty", "input", "reference")


def check_fields(data: dict, fields: tuple, label: str) -> None:
    """Raise a clear error if one of the expected fields is not in data."""
    for field in fields:
        if field not in data:
            raise ValueError(f"{label} is missing the field '{field}'")


TASK_FOLDERS = ("tasks", "tasks/generated")


def find_task_file(task_name: str, folders: tuple = TASK_FOLDERS) -> str:
    """Return the path of the task file with this name, written by hand or drafted.

    Raises FileNotFoundError with a clear message if no folder has it.
    """
    for folder in folders:
        path = Path(folder) / f"{task_name}.yaml"
        if path.exists():
            return str(path)
    raise FileNotFoundError(f"No task file named {task_name}.yaml in {', '.join(folders)}")


def load_task(path: str) -> Task:
    """Read a task YAML file and return it as a Task.

    Raises ValueError naming the field if the file or one of its inputs is incomplete.
    """
    with open(path, encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return parse_task(data, path)


def parse_task(data, path: str) -> Task:
    """Check the content of a task (as read from YAML) and return it as a Task.

    path: where the content comes from, only used in error messages.
    Also used for tasks written by a model, so it checks the overall shape as well.
    """
    if not isinstance(data, dict):
        raise ValueError(f"Task file {path} is not a set of named fields")
    check_fields(data, TASK_FIELDS, f"Task file {path}")
    if not isinstance(data["inputs"], list) or not data["inputs"]:
        raise ValueError(f"Task file {path}: inputs must be a non-empty list")

    inputs = []
    for number, item in enumerate(data["inputs"], start=1):
        check_fields(item, INPUT_FIELDS, f"Input {number} in {path}")
        inputs.append(
            TaskInput(
                id=item["id"],
                difficulty=item["difficulty"],
                input=item["input"],
                reference=item["reference"],
            )
        )
    return Task(
        name=data["name"],
        description=data["description"],
        instruction=data["instruction"],
        inputs=inputs,
    )
