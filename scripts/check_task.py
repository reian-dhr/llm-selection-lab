"""Check which model does a task best, from a plain description, in one command.

Run with:
  uv run python scripts/check_task.py "Turn my meeting notes into a short follow-up email..."
Options:
  --context my_facts.txt   real facts the test should use (for example docs/how_it_works.md)
  --preview                stop after the cost estimate, without calling the compared models
  --premium                also compare the premium (expensive) models
  --max-cost 5             spending cap for this run, in US dollars

It drafts a test with the free helper model, shows what it understood, runs the test on the
models (this costs money, within the cap), then prints a summary and a recommendation.
"""

import argparse
from pathlib import Path

from model_selection_lab.check import finish_check, prepare_check
from model_selection_lab.runs.task_drafter import describe_understanding
from model_selection_lab.settings import MAX_RUN_COST_USD


def read_arguments() -> argparse.Namespace:
    """Read the description and the options from the command line."""
    parser = argparse.ArgumentParser(description="Check which model does a task best.")
    parser.add_argument("description", help="the task, described in your own words")
    parser.add_argument("--context", help="a text file of real facts to use in the test")
    parser.add_argument("--preview", action="store_true", help="stop after the cost estimate")
    parser.add_argument("--premium", action="store_true", help="also compare the premium models")
    parser.add_argument(
        "--max-cost", type=float, default=MAX_RUN_COST_USD, help="spending cap in USD"
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = read_arguments()
    context = Path(arguments.context).read_text(encoding="utf-8") if arguments.context else ""

    print("1. Drafting the test with the helper model...\n")
    outcome = prepare_check(
        arguments.description, context=context, include_premium=arguments.premium
    )
    print("\n".join(describe_understanding(outcome.drafted)))
    print(f"\nTest saved to {outcome.task_path} ({len(outcome.task.inputs)} inputs).")
    print(f"Models compared: {', '.join(outcome.model_keys)}")
    print(
        f"This run costs at most about ${outcome.max_cost_usd:.2f} "
        f"(cap ${arguments.max_cost:.2f})."
    )

    if arguments.preview:
        print("\nPreview only: no model was called. To run this exact test:")
        print(f"  uv run python scripts/run_task.py {outcome.task_path}")
    else:
        print("\n2. Running the test on the models...\n")
        outcome = finish_check(outcome, max_cost_usd=arguments.max_cost)
        print("\n3. Summary per model:\n")
        decimals = {
            "median_latency_s": 2, "avg_input_tokens": 0, "avg_output_tokens": 0, "avg_cost_usd": 6
        }
        print(outcome.summary.round(decimals).to_string(na_rep="-"))
        print("\n4. Recommendation:")
        for line in outcome.recommendation:
            print(f"  - {line}")
        print("  - Quality is not judged in a check yet: read the answers in")
        print(f"    {outcome.results_path}")
