"""Estimate what one real, larger job would cost on each model.

Run with:
  uv run python scripts/estimate_job_cost.py results/<file>.jsonl --context-pages 30 --answer-pages 2 --turns 5

Makes no API calls. It uses the results file to measure how each model counts tokens,
then prices a job of the size you describe. Figures are at list prices and are estimates:
the times_cheapest column is the most reliable one.
"""

import argparse

import pandas as pd

from model_selection_lab.report.job_cost import estimate_job_cost, measure_token_rates
from model_selection_lab.runs.task_loader import find_task_file, load_task


def read_arguments() -> argparse.Namespace:
    """Read the results file and the size of the job from the command line."""
    parser = argparse.ArgumentParser(description="Estimate the cost of one real job per model.")
    parser.add_argument("results_file", help="a results .jsonl file from a test run")
    parser.add_argument(
        "--context-pages", type=float, required=True, help="pages of material given to the model"
    )
    parser.add_argument(
        "--answer-pages", type=float, required=True, help="pages the model writes at each turn"
    )
    parser.add_argument("--turns", type=int, default=1, help="number of exchanges (default 1)")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = read_arguments()
    results = pd.read_json(arguments.results_file, lines=True)
    task = load_task(find_task_file(results["task_name"].iloc[0]))

    token_rates = measure_token_rates(results, task)
    estimate = estimate_job_cost(
        token_rates, arguments.context_pages, arguments.answer_pages, arguments.turns
    )

    print(
        f"One job: {arguments.context_pages:g} pages of context, "
        f"{arguments.answer_pages:g} pages of answer per turn, {arguments.turns} turn(s)."
    )
    print(f"Token rates measured on task '{task.name}'. Costs in USD at list prices.\n")
    print(estimate.round({"job_cost_usd": 4, "low_usd": 4, "high_usd": 4, "times_cheapest": 1}))
