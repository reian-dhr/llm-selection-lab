"""Batch runner: send every input of a task to every chosen model, and save each answer.

This is the loop that ties the other files together:
- task_loader.py gives the instruction and the inputs,
- model_client.py (run_model) makes one call to one model,
- results_storage.py saves each answer as one line in a .jsonl file.

Each answer is saved as soon as it arrives, so a run that is interrupted keeps
everything it had finished.

Three things protect the budget: estimate_max_cost() says what a run could cost before
it starts, run_task() stops when actual spending reaches a cap, and an answer already
saved for the same prompt and model is reused instead of being paid for again.
"""

from model_selection_lab.models.call_result import RunResult
from model_selection_lab.models.model_client import run_model
from model_selection_lab.models.model_registry import MODELS, calculate_cost
from model_selection_lab.runs.results_storage import append_result, hash_prompt, make_record
from model_selection_lab.runs.task_loader import Task
from model_selection_lab.settings import (
    MAX_ANSWER_TOKENS,
    MAX_RUN_COST_USD,
    THINKING_EXTRA_TOKENS,
    TOKENS_PER_WORD_ESTIMATE,
)


def build_prompt(instruction: str, input_text: str) -> str:
    """Put the task instruction and one input together into the text sent to a model.
    With no input (a plain request), the instruction is sent alone."""
    if not input_text.strip():
        return instruction.strip()
    return f"{instruction.strip()}\n\nFACTS\n{input_text.strip()}"


def estimate_max_cost(
    task: Task, model_keys: list[str], models: dict = MODELS, reusable: dict | None = None
) -> float:
    """Return the most a run could cost in USD, before any call is made.

    It is an upper limit, not a forecast: it assumes every answer uses the whole length
    limit (plus the whole thinking allowance for models that think). Real runs usually
    cost much less.
    models: the registry to read prices from. Tests pass a small fake one.
    reusable: saved answers that will be reused, so those calls cost nothing.
    """
    reusable = reusable or {}
    total = 0.0
    for task_input in task.inputs:
        prompt = build_prompt(task.instruction, task_input.input)
        input_tokens = len(prompt.split()) * TOKENS_PER_WORD_ESTIMATE
        for model_key in model_keys:
            if (task_input.id, model_key, hash_prompt(prompt)) in reusable:
                continue
            model = models[model_key]
            output_tokens = MAX_ANSWER_TOKENS
            if model.reasoning:
                output_tokens += THINKING_EXTRA_TOKENS
            total += calculate_cost(
                input_tokens,
                output_tokens,
                model.input_price_per_mtok,
                model.output_price_per_mtok,
            )
    return total


def run_one(run, model_key: str, prompt: str) -> RunResult:
    """Make one call and always return a RunResult, even if the call crashes.

    run_model already turns API failures into a RunResult with `error` set. This also
    catches anything else (for example an unknown model key), so one bad call can
    never stop the whole run.
    """
    try:
        return run(model_key, prompt, max_tokens=MAX_ANSWER_TOKENS)
    except Exception as exc:
        error = f"{type(exc).__name__}: {str(exc)[:200]}"
        return RunResult(
            text="", input_tokens=0, output_tokens=0, latency_s=0.0, cost_usd=0.0, error=error
        )


def run_task(
    task: Task,
    model_keys: list[str],
    output_path: str,
    run=run_model,
    max_cost_usd: float = MAX_RUN_COST_USD,
    reusable: dict | None = None,
) -> None:
    """Run every input of the task on every model and append each result to output_path.

    run: the function that makes one call. Tests pass a fake one so they need no network.
    max_cost_usd: the spending cap. The run stops before the next call once spending
        has reached it. Answers already saved are kept.
    reusable: saved answers from load_reusable_answers(). When one matches the input,
        the model and the exact prompt, it is copied into this run's file and no call
        is made.
    """
    reusable = reusable or {}
    total = len(task.inputs) * len(model_keys)
    done = 0
    failed = 0
    reused = 0
    spent = 0.0
    for task_input in task.inputs:
        prompt = build_prompt(task.instruction, task_input.input)
        prompt_hash = hash_prompt(prompt)
        for model_key in model_keys:
            saved = reusable.get((task_input.id, model_key, prompt_hash))
            if saved is not None:
                append_result(output_path, saved)
                done += 1
                reused += 1
                print(f"[{done}/{total}] {task_input.id} on {model_key}: reused saved answer, $0")
                continue
            if spent >= max_cost_usd:
                print(
                    f"STOPPED: spending reached the cap of ${max_cost_usd:.2f} "
                    f"after {done} of {total} calls. Results so far saved to {output_path}"
                )
                return
            result = run_one(run, model_key, prompt)
            spent += result.cost_usd
            record = make_record(task.name, task_input.id, model_key, result, prompt_hash)
            append_result(output_path, record)

            done += 1
            if result.error is None:
                status = f"ok, {result.latency_s:.1f}s, ${result.cost_usd:.5f}"
                if result.truncated:
                    status += ", CUT OFF at the length limit"
            else:
                failed += 1
                status = f"FAILED, {result.error}"
            print(f"[{done}/{total}] {task_input.id} on {model_key}: {status}")

    print(
        f"Done: {total} answers ({reused} reused, {failed} failed), ${spent:.4f} spent. "
        f"Results saved to {output_path}"
    )
