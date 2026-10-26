"""Check: the whole chain in one function, from a plain description to a recommendation.

    description -> draft the test (helper model) -> cost estimate -> run on the models
                -> summary per model -> recommendation

Every step already exists elsewhere; this file only puts them in order. It has two parts,
so the user sees the drafted test and its cost before any compared model is called:
- prepare_check(): draft the test and estimate the cost (only the free helper is called),
- finish_check(): run the test on the models, then summarise and recommend.
run_check() does both in one call. list_checks() and load_check() bring past checks back.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from model_selection_lab.models.model_client import run_model
from model_selection_lab.models.model_registry import select_models
from model_selection_lab.report.summary import recommend, summarise
from model_selection_lab.runs.batch_runner import estimate_max_cost, run_task
from model_selection_lab.runs.results_storage import load_reusable_answers
from model_selection_lab.runs.task_drafter import GENERATED_FOLDER, draft_task, save_drafted_task
from model_selection_lab.runs.task_loader import Task, load_task
from model_selection_lab.settings import MAX_RUN_COST_USD, REUSE_ANSWERS_FOR_DAYS


@dataclass
class CheckOutcome:
    """Everything a check produced, for the command line or the MCP server to show."""

    drafted: dict  # the helper model's full answer, with "understood" and "open_questions"
    task: Task
    task_path: str  # where the drafted test was saved
    model_keys: list[str]  # the models compared
    max_cost_usd: float  # the most the run could cost, estimated before any call
    reusable: dict  # saved answers that the run will reuse instead of paying again
    results_path: str | None = None  # None if the check stopped at the preview
    summary: pd.DataFrame | None = None  # one row per model
    recommendation: list[str] | None = None  # short sentences


def prepare_check(
    description: str,
    context: str = "",
    include_premium: bool = False,
    run=run_model,
    template: str | None = None,
    tasks_folder: str = GENERATED_FOLDER,
    results_folder: str = "results",
) -> CheckOutcome:
    """Draft a test from the description and estimate what running it would cost.

    Only the helper model is called. run, template, tasks_folder, results_folder: only for
    tests, to use fakes and a temporary folder.
    """
    drafted, _, _ = draft_task(description, run=run, template=template, context=context)
    task_path = save_drafted_task(drafted, folder=tasks_folder)
    task = load_task(task_path)  # read back, so the run uses the saved name and content
    model_keys = select_models(include_premium=include_premium)
    reusable = load_reusable_answers(results_folder, task.name, REUSE_ANSWERS_FOR_DAYS)
    return CheckOutcome(
        drafted=drafted,
        task=task,
        task_path=task_path,
        model_keys=model_keys,
        max_cost_usd=estimate_max_cost(task, model_keys, reusable=reusable),
        reusable=reusable,
    )


def finish_check(
    outcome: CheckOutcome,
    max_cost_usd: float = MAX_RUN_COST_USD,
    run=run_model,
    results_folder: str = "results",
) -> CheckOutcome:
    """Run the prepared test on the models, then add the summary and the recommendation."""
    task, model_keys, reusable = outcome.task, outcome.model_keys, outcome.reusable
    # The drafted task name already holds the date, so the file name adds only the time.
    started = datetime.now().strftime("%H%M%S")
    outcome.results_path = f"{results_folder}/{task.name}_run_{started}.jsonl"
    run_task(
        task, model_keys, outcome.results_path, run=run,
        max_cost_usd=max_cost_usd, reusable=reusable,
    )
    outcome.summary = summarise(pd.read_json(outcome.results_path, lines=True))
    outcome.recommendation = recommend(outcome.summary)
    return outcome


def run_check(
    description: str,
    context: str = "",
    include_premium: bool = False,
    max_cost_usd: float = MAX_RUN_COST_USD,
    preview_only: bool = False,
    run=run_model,
    template: str | None = None,
    tasks_folder: str = GENERATED_FOLDER,
    results_folder: str = "results",
) -> CheckOutcome:
    """The whole chain in one call: prepare, then (unless preview_only) finish."""
    outcome = prepare_check(
        description, context, include_premium, run, template, tasks_folder, results_folder
    )
    if preview_only:
        return outcome
    return finish_check(outcome, max_cost_usd, run, results_folder)


def latest_run(task_name: str, results_folder: str = "results") -> str | None:
    """Return the newest results file of a check, or None if it was never run."""
    runs = sorted(Path(results_folder).glob(f"{task_name}_run_*.jsonl"))
    return str(runs[-1]) if runs else None


def list_checks(tasks_folder: str = GENERATED_FOLDER, results_folder: str = "results") -> list:
    """Return the past checks, newest first: name, file, goal, and latest run (or None)."""
    checks = []
    for path in sorted(Path(tasks_folder).glob("*.yaml"), reverse=True):
        drafted = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        checks.append(
            {
                "name": path.stem,
                "task_path": str(path),
                "goal": (drafted.get("understood") or {}).get("goal", path.stem),
                "results_path": latest_run(path.stem, results_folder),
            }
        )
    return checks


def load_check(task_path: str, results_folder: str = "results") -> CheckOutcome:
    """Rebuild a past check from its saved files, with its latest run if there is one."""
    drafted = yaml.safe_load(Path(task_path).read_text(encoding="utf-8"))
    task = load_task(task_path)
    results_path = latest_run(task.name, results_folder)
    if results_path is None:
        model_keys = select_models()
    else:
        model_keys = list(dict.fromkeys(pd.read_json(results_path, lines=True)["model_key"]))
    reusable = load_reusable_answers(results_folder, task.name, REUSE_ANSWERS_FOR_DAYS)
    outcome = CheckOutcome(
        drafted=drafted,
        task=task,
        task_path=task_path,
        model_keys=model_keys,
        max_cost_usd=estimate_max_cost(task, model_keys, reusable=reusable),
        reusable=reusable,
    )
    if results_path is not None:
        outcome.results_path = results_path
        outcome.summary = summarise(pd.read_json(results_path, lines=True))
        outcome.recommendation = recommend(outcome.summary)
    return outcome

