"""Ask: one real request, answered by every compared model, with the judge's pick of the best.

This is what the web app runs. Unlike check.py, nothing is drafted: the request itself is
sent to the models, as a task with a single input.

    request -> cost estimate (refused above the cap) -> every standard model answers
            -> the judge picks the best -> AskOutcome

prefer() records that the user preferred another answer than the judge's. It is saved as a
pick, so the judge learns from it next time (see judge/judge.py).
"""

import re
from dataclasses import dataclass
from datetime import datetime

from model_selection_lab.check import CheckOutcome
from model_selection_lab.judge.judge import LETTERS, judge_check, load_verdicts, verdicts_path_for
from model_selection_lab.judge.picks import (
    answers_by_input,
    make_pick,
    picks_path_for,
    save_pick,
    shown_order,
)
from model_selection_lab.models.model_client import run_model
from model_selection_lab.models.model_registry import display_name, select_models
from model_selection_lab.runs.batch_runner import estimate_max_cost, run_task
from model_selection_lab.runs.task_loader import Task, TaskInput
from model_selection_lab.settings import MAX_RUN_COST_USD

INPUT_ID = "request"
DEFAULT_PREFERENCE_REASON = "Preferred over the judge's pick"
GOAL = "Answer the user's request as well as possible."


@dataclass
class AskOutcome:
    """Everything one request produced, for the web app to show."""

    request: str
    task_name: str  # also the name of the saved files
    results_path: str
    answers: dict  # model_key -> saved line (text, cost_usd, latency_s, ...), successful only
    best_model: str | None  # the judge's pick; None if it found none good or was busy
    judge_reason: str
    judge_error: str  # empty if the judge answered


def request_task(request: str, name: str) -> Task:
    """A task with a single input: the request is the instruction, there are no extra facts."""
    return Task(
        name=name,
        description=GOAL,
        instruction=request,
        inputs=[TaskInput(id=INPUT_ID, difficulty="", input="", reference=None)],
    )


def name_the_letters(text: str, letter_to_model: dict) -> str:
    """The judge sees answers labelled A, B, C...; the user sees model names. Rewrite
    'Answer B' and '(B)' in the judge's text as the model's name. Other words such as
    'option B' are left alone: they may refer to the request itself."""

    def name(match):
        return display_name(letter_to_model[match.group(2)])

    letters = "".join(letter_to_model)
    text = re.sub(rf"\b([Aa]nswer) ([{letters}])\b", name, text)
    return re.sub(rf"(\()([{letters}])\)", lambda match: f"({name(match)})", text)


def ask(
    request: str,
    include_premium: bool = False,
    run=run_model,
    judge_run=run_model,
    results_folder: str = "results",
    verdicts_folder: str = "data/verdicts",
    max_cost_usd: float = MAX_RUN_COST_USD,
) -> AskOutcome:
    """Answer the request with every standard model (and the premium ones if asked), then
    ask the judge for the best one.

    Raises ValueError if the request is empty or could cost more than the cap.
    run, judge_run, the folders: only for tests, to use fakes and temporary folders.
    """
    if not request.strip():
        raise ValueError("The request is empty.")
    name = datetime.now().strftime("ask_%Y%m%d_%H%M%S_%f")  # unique, even within a second
    task = request_task(request, name)
    model_keys = select_models(include_premium=include_premium)
    max_cost = estimate_max_cost(task, model_keys)
    if max_cost > max_cost_usd:
        raise ValueError(
            f"This request could cost ${max_cost:.2f}, above the cap of ${max_cost_usd:.2f}."
        )

    results_path = f"{results_folder}/{name}.jsonl"
    run_task(task, model_keys, results_path, run=run, max_cost_usd=max_cost_usd)
    answers = answers_by_input(results_path).get(INPUT_ID, {})

    outcome = CheckOutcome(
        drafted={}, task=task, task_path="", model_keys=model_keys,
        max_cost_usd=max_cost, reusable={}, results_path=results_path,
    )
    judge_error = ""
    try:
        judge_check(outcome, run=judge_run, verdicts_folder=verdicts_folder)
    except (RuntimeError, ValueError) as problem:  # for example the free tier is busy
        judge_error = str(problem)
    verdict = load_verdicts(verdicts_path_for(name, verdicts_folder)).get(INPUT_ID)
    # The same shuffled order the judge saw (see judge_check), to turn letters into names.
    letter_to_model = dict(zip(LETTERS, shown_order(INPUT_ID, list(answers))))

    return AskOutcome(
        request=request,
        task_name=name,
        results_path=results_path,
        answers=answers,
        best_model=verdict.chosen_model if verdict else None,
        judge_reason=name_the_letters(verdict.reason, letter_to_model) if verdict else "",
        judge_error=judge_error,
    )


def prefer(outcome: AskOutcome, model_key: str, reason: str = "", picks_folder="data/picks"):
    """Save that the user prefers this answer. Returns the path of the picks file."""
    pick = make_pick(
        INPUT_ID,
        model_key,
        reason.strip() or DEFAULT_PREFERENCE_REASON,
        refused_models=[],
        refusal_reason="",
        shown_order=shown_order(INPUT_ID, list(outcome.answers)),
        results_file=outcome.results_path,
    )
    path = picks_path_for(outcome.task_name, picks_folder)
    save_pick(path, pick)
    return path
