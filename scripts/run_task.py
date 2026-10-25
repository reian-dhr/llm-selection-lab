"""Run one task on the models in the registry and save the answers.

Run with: uv run python scripts/run_task.py examples/tasks/example_summary.yaml
Add --premium to include the expensive models as well.
Add --preview to see what the run could cost, without making any call.
Add --max-cost 5 to change the spending cap for this run (in US dollars).
Add --fresh to ignore saved answers and call every model again.

This makes real API calls, so it costs money. An answer already saved for the same
prompt and model is reused and costs nothing.
Each run writes a new file in results/, named with the task and the start time.
"""

import argparse
from datetime import datetime

from model_selection_lab.models.model_registry import MODELS, select_models
from model_selection_lab.runs.batch_runner import estimate_max_cost, run_task
from model_selection_lab.runs.results_storage import load_reusable_answers
from model_selection_lab.runs.task_loader import load_task
from model_selection_lab.settings import MAX_RUN_COST_USD, REUSE_ANSWERS_FOR_DAYS


def read_arguments() -> argparse.Namespace:
    """Read the task file and the options from the command line."""
    parser = argparse.ArgumentParser(description="Run one task on the models in the registry.")
    parser.add_argument("task_file", help="a task YAML file, for example examples/tasks/example_summary.yaml")
    parser.add_argument(
        "--premium", action="store_true", help="also run the premium (expensive) models"
    )
    parser.add_argument(
        "--preview", action="store_true", help="show the cost estimate and stop, no calls made"
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        default=MAX_RUN_COST_USD,
        help=f"spending cap for this run in USD (default {MAX_RUN_COST_USD:.2f})",
    )
    parser.add_argument(
        "--fresh", action="store_true", help="ignore saved answers and call every model again"
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = read_arguments()
    task = load_task(arguments.task_file)
    model_keys = select_models(include_premium=arguments.premium)
    skipped = [key for key in MODELS if key not in model_keys]

    started = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"results/{task.name}_{started}.jsonl"

    print(f"Task '{task.name}': {len(task.inputs)} inputs on {len(model_keys)} models")
    print(f"Models: {', '.join(model_keys)}")
    if skipped:
        print(f"Premium models left out (add --premium to include them): {', '.join(skipped)}")

    reusable = {}
    if not arguments.fresh:
        reusable = load_reusable_answers("results", task.name, REUSE_ANSWERS_FOR_DAYS)

    max_cost = estimate_max_cost(task, model_keys, reusable=reusable)
    print(f"Saved answers available for reuse: {len(reusable)}.")
    print(f"This run will cost at most about ${max_cost:.2f} (usually much less).")
    print(f"Spending cap: ${arguments.max_cost:.2f}.")
    if max_cost > arguments.max_cost:
        print("The cap is below the maximum, so the run may stop before the end.")
    if arguments.preview:
        print("Preview only: no call was made.")
    else:
        run_task(
            task, model_keys, output_path, max_cost_usd=arguments.max_cost, reusable=reusable
        )
