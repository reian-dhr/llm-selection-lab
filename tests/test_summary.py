"""Tests for report/summary.py: the per-model summary and the recommendation."""

import pandas as pd

from model_selection_lab.report.summary import format_report, recommend, summarise


def row(model, cost, latency, error=None, truncated=False):
    return {
        "model_key": model, "cost_usd": cost, "latency_s": latency, "input_tokens": 10,
        "output_tokens": 5, "error": error, "truncated": truncated,
    }


def test_recommend_names_cheapest_and_fastest_reliable_models():
    results = pd.DataFrame([
        row("cheap", 0.001, 3.0), row("cheap", 0.001, 3.0),
        row("fast", 0.010, 1.0), row("fast", 0.010, 1.0),
        row("flaky", 0.0001, 0.5, truncated=True), row("flaky", 0.0001, 0.5),
    ])

    lines = recommend(summarise(results))

    assert lines[0].startswith("Cheapest: cheap")
    assert "fast, costs 10 times more" in lines[0]
    assert lines[1].startswith("Fastest: fast")
    assert "Left out because of errors or cut-off answers: flaky." in lines


def test_format_report_shows_the_table_the_recommendation_and_where_answers_are():
    summary = summarise(pd.DataFrame([row("cheap", 0.001234567, 3.14159)]))

    text = format_report(summary, ["Cheapest: cheap."], "results/x.jsonl")

    assert "cheap" in text and "0.001235" in text and "3.14" in text  # rounded table
    assert "- Cheapest: cheap." in text
    assert "Quality is not judged in a check yet: read the answers in results/x.jsonl" in text
