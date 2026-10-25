"""Task drafter: turn a plain description into a task file the rest of the tool can run.

The user describes what they want in their own words. A helper model (from a family outside
the comparison) reads it and writes a task as JSON: what it understood, open questions, what
a good answer looks like, the instruction every model will get, and a few test inputs.

The answer is checked by the same code that checks hand-written task files. If it is not a
valid task, the model is asked once more, with the problem it made. The prompt that tells the
model how to do this lives in a private file (see TASK_DRAFTER_PROMPT in settings.py).
"""

import re
from datetime import datetime
from pathlib import Path

import yaml

from model_selection_lab.models.model_client import run_model
from model_selection_lab.runs.task_loader import Task, parse_task
from model_selection_lab.settings import (
    DRAFTED_INPUTS,
    DRAFTER_MAX_TOKENS,
    TASK_DRAFTER_MODEL,
    TASK_DRAFTER_PROMPT,
    TASK_DRAFTER_PROMPT_EXAMPLE,
)
from model_selection_lab.text_files import read_text

GENERATED_FOLDER = "tasks/generated"


def build_drafter_prompt(
    template: str, description: str, number_of_inputs: int, context: str = ""
) -> str:
    """Fill the prompt template with the description, the number of inputs, and any context.

    context: real facts given by the user (for example a project guide), so the test inputs
        use them instead of invented ones. Empty if none was given.
    """
    full_description = description.strip()
    if context.strip():
        full_description += (
            "\n\nContext provided by the person (real facts: use them in the test inputs "
            "instead of inventing any):\n" + context.strip()
        )
    return template.replace("{{DESCRIPTION}}", full_description).replace(
        "{{NUMBER_OF_INPUTS}}", str(number_of_inputs)
    )


def strip_code_fences(text: str) -> str:
    """Remove ``` lines that models sometimes put around their answer even when asked not to."""
    lines = [line for line in text.strip().splitlines() if not line.strip().startswith("```")]
    return "\n".join(lines)


def parse_drafted_task(text: str) -> tuple[dict, Task]:
    """Read the model's answer (JSON, which the YAML reader also accepts) and check the task.

    Returns the raw content (which also holds "understood" and "open_questions") and the
    Task. Raises ValueError with a readable message if anything is wrong.
    """
    try:
        data = yaml.safe_load(strip_code_fences(text))
    except yaml.YAMLError as problem:
        raise ValueError(f"The answer is not valid JSON: {problem}") from problem
    task = parse_task(data, "drafted by the model")
    return data, task


def draft_task(
    description: str,
    run=run_model,
    model_key: str = TASK_DRAFTER_MODEL,
    template: str | None = None,
    context: str = "",
) -> tuple[dict, Task, float]:
    """Ask the helper model to turn a description into a task, with one retry if invalid.

    run: the function that makes one call. Tests pass a fake one so they need no network.
    template: the prompt template. By default it is read from the private prompt file.
    context: real facts given by the user, used in the test inputs (see build_drafter_prompt).
    Returns the raw content, the Task, and the total cost of the calls in USD.
    """
    if template is None:
        template = read_text(TASK_DRAFTER_PROMPT, TASK_DRAFTER_PROMPT_EXAMPLE)
    prompt = build_drafter_prompt(template, description, DRAFTED_INPUTS, context)

    cost = 0.0
    for attempt in (1, 2):
        result = run(model_key, prompt, max_tokens=DRAFTER_MAX_TOKENS)
        cost += result.cost_usd
        if result.error is not None:
            raise RuntimeError(f"The helper model failed: {result.error}")
        try:
            data, task = parse_drafted_task(result.text)
            return data, task, cost
        except ValueError as problem:
            if attempt == 2:
                raise ValueError(f"The helper model did not write a valid task: {problem}")
            # Ask once more, saying exactly what was wrong.
            prompt += (
                f"\n\nYour previous answer could not be used: {problem}\n"
                "Return the complete task again, as one valid JSON object only."
            )


def safe_file_name(name: str) -> str:
    """Turn a task name into something safe for a file name: lower case, letters, digits, _."""
    cleaned = re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_")
    return cleaned or "task"


class ReadableDumper(yaml.SafeDumper):
    """Writes texts of several lines as | blocks, like the hand-written task files."""


def write_text_as_block(dumper: yaml.SafeDumper, text: str):
    """Use a | block for texts of several lines, and the normal style for short ones."""
    if "\n" in text:
        # Spaces at the end of a line stop YAML from using a block, so remove them.
        text = "\n".join(line.rstrip() for line in text.splitlines()) + "\n"
        return dumper.represent_scalar("tag:yaml.org,2002:str", text, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", text)


ReadableDumper.add_representer(str, write_text_as_block)


def save_drafted_task(data: dict, folder: str = GENERATED_FOLDER) -> str:
    """Save a drafted task as a YAML file and return its path.

    The task name gets the time added, so a new draft never overwrites an old one, and the
    file is named exactly after the task, so other parts of the tool can find it by name.
    """
    data = dict(data)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data["name"] = f"{safe_file_name(str(data['name']))}_{stamp}"
    path = Path(folder) / f"{data['name']}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.dump(data, Dumper=ReadableDumper, sort_keys=False, allow_unicode=True, width=100)
    path.write_text(text, encoding="utf-8")
    return str(path)


def describe_understanding(data: dict) -> list[str]:
    """Return readable lines: what the helper model understood, and its open questions."""
    lines = ["What it understood:"]
    for part, value in (data.get("understood") or {}).items():
        lines.append(f"  {part}: {value}")
    questions = data.get("open_questions") or []
    lines.append("Open questions:" if questions else "Open questions: none")
    lines += [f"  - {question}" for question in questions]
    return lines
