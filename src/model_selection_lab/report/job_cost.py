"""Job cost estimate: what one real, larger job would cost on each model.

The test run uses short prompts. The real job, done later in the chosen model, has far
more context, so it costs far more. This file estimates that cost from two things:
- what the test run measured: how many tokens each model counts per word,
- what the user says about the real job: pages of context, pages of answer, and turns.

It is an estimate at list prices, good for comparing models, not an invoice. It leaves
out reasoning ("thinking") tokens, discounts, and the higher prices some models charge
for very long contexts. The most reliable column is times_cheapest: how many times more
a model costs than the cheapest one.
"""

import pandas as pd

from model_selection_lab.models.model_registry import MODELS, calculate_cost
from model_selection_lab.runs.batch_runner import build_prompt
from model_selection_lab.runs.task_loader import Task
from model_selection_lab.settings import COST_UNCERTAINTY, WORDS_PER_PAGE


def measure_token_rates(results: pd.DataFrame, task: Task) -> pd.DataFrame:
    """Return one row per model: tokens counted per word of prompt, and per word of answer.

    Each provider cuts text into tokens its own way, so the same text is a different
    number of tokens on each model. This measures it from the successful test calls.
    """
    prompt_words = {
        task_input.id: len(build_prompt(task.instruction, task_input.input).split())
        for task_input in task.inputs
    }
    successful = results[results["error"].isna()].copy()
    successful["prompt_words"] = successful["input_id"].map(prompt_words)
    successful["answer_words"] = successful["text"].str.split().str.len()

    columns = ["input_tokens", "prompt_words", "output_tokens", "answer_words"]
    totals = successful.groupby("model_key", sort=False)[columns].sum()
    return pd.DataFrame(
        {
            "input_tokens_per_word": totals["input_tokens"] / totals["prompt_words"],
            "output_tokens_per_word": totals["output_tokens"] / totals["answer_words"],
        }
    )


def estimate_job_cost(
    token_rates: pd.DataFrame,
    context_pages: float,
    answer_pages: float,
    turns: int = 1,
    models: dict = MODELS,
) -> pd.DataFrame:
    """Return one row per model: the estimated cost of one real job.

    token_rates: the table from measure_token_rates().
    context_pages: pages of material given to the model (documents, data, instructions).
    answer_pages: pages the model writes back at each turn.
    turns: number of exchanges. In a conversation everything said so far is sent again
        at every turn, so the cost grows faster than the number of turns.
    models: the registry to read prices from. Tests pass a small fake one.
    """
    if context_pages < 0 or answer_pages < 0 or turns < 1:
        raise ValueError("Pages cannot be negative and turns must be at least 1")

    context_words = context_pages * WORDS_PER_PAGE
    answer_words = answer_pages * WORDS_PER_PAGE
    # Each turn sends the context again, plus every answer written before it.
    earlier_answers = turns * (turns - 1) / 2  # 0 + 1 + 2 + ... + (turns - 1)
    input_words = turns * context_words + earlier_answers * answer_words
    output_words = turns * answer_words

    rows = {}
    for model_key, rate in token_rates.iterrows():
        cost = calculate_cost(
            input_tokens=input_words * rate["input_tokens_per_word"],
            output_tokens=output_words * rate["output_tokens_per_word"],
            input_price_per_m=models[model_key].input_price_per_mtok,
            output_price_per_m=models[model_key].output_price_per_mtok,
        )
        rows[model_key] = {
            "job_cost_usd": cost,
            "low_usd": cost * (1 - COST_UNCERTAINTY),
            "high_usd": cost * (1 + COST_UNCERTAINTY),
        }

    estimate = pd.DataFrame.from_dict(rows, orient="index")
    estimate["times_cheapest"] = estimate["job_cost_usd"] / estimate["job_cost_usd"].min()
    return estimate
