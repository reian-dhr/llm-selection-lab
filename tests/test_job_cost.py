"""Tests for report/job_cost.py: token rates and the cost of one real job."""

from types import SimpleNamespace

import pandas as pd
import pytest

from model_selection_lab.report.job_cost import estimate_job_cost, measure_token_rates
from model_selection_lab.runs.batch_runner import build_prompt
from model_selection_lab.runs.task_loader import Task, TaskInput

TASK = Task(
    name="tiny_task",
    description="A good answer is short.",
    instruction="Summarise the facts in one sentence.",
    inputs=[TaskInput(id="easy_01", difficulty="easy", input="The cat sat.", reference=None)],
)
PROMPT_WORDS = len(build_prompt(TASK.instruction, "The cat sat.").split())

# Fake prices: model-b costs ten times more than model-a, for input and for output.
FAKE_MODELS = {
    "model-a": SimpleNamespace(input_price_per_mtok=1.0, output_price_per_mtok=2.0),
    "model-b": SimpleNamespace(input_price_per_mtok=10.0, output_price_per_mtok=20.0),
}
# Both models count 2 tokens per word, for prompts and for answers.
RATES = pd.DataFrame(
    {"input_tokens_per_word": [2.0, 2.0], "output_tokens_per_word": [2.0, 2.0]},
    index=["model-a", "model-b"],
)


def test_measure_token_rates_divides_tokens_by_words():
    results = pd.DataFrame(
        [
            {
                "model_key": "model-a",
                "input_id": "easy_01",
                "text": "one two three four",  # 4 words
                "input_tokens": PROMPT_WORDS * 2,
                "output_tokens": 6,
                "error": None,
            },
            # A failed call must be ignored.
            {
                "model_key": "model-a",
                "input_id": "easy_01",
                "text": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "error": "timeout",
            },
        ]
    )

    rates = measure_token_rates(results, TASK)

    assert rates.loc["model-a", "input_tokens_per_word"] == pytest.approx(2.0)
    assert rates.loc["model-a", "output_tokens_per_word"] == pytest.approx(1.5)


def test_estimate_job_cost_for_one_turn():
    estimate = estimate_job_cost(RATES, context_pages=10, answer_pages=1, models=FAKE_MODELS)

    # 10 pages = 5,000 words = 10,000 input tokens. 1 page = 500 words = 1,000 output tokens.
    # model-a: 10,000 x $1 per million + 1,000 x $2 per million = 0.010 + 0.002.
    assert estimate.loc["model-a", "job_cost_usd"] == pytest.approx(0.012)
    assert estimate.loc["model-a", "low_usd"] == pytest.approx(0.012 * 0.7)
    assert estimate.loc["model-a", "high_usd"] == pytest.approx(0.012 * 1.3)
    assert estimate.loc["model-a", "times_cheapest"] == pytest.approx(1.0)
    assert estimate.loc["model-b", "times_cheapest"] == pytest.approx(10.0)


def test_estimate_job_cost_resends_everything_each_turn():
    estimate = estimate_job_cost(
        RATES, context_pages=10, answer_pages=1, turns=3, models=FAKE_MODELS
    )

    # Input: the context 3 times (15,000 words) plus earlier answers, 0 + 1 + 2 = 3 pages
    # (1,500 words): 16,500 words = 33,000 tokens. Output: 3 pages = 1,500 words = 3,000 tokens.
    # model-a: 33,000 x $1 per million + 3,000 x $2 per million = 0.033 + 0.006.
    assert estimate.loc["model-a", "job_cost_usd"] == pytest.approx(0.039)


def test_estimate_job_cost_refuses_zero_turns():
    with pytest.raises(ValueError, match="turns must be at least 1"):
        estimate_job_cost(RATES, context_pages=10, answer_pages=1, turns=0, models=FAKE_MODELS)
