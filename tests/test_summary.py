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
    assert "Left out because of errors, cut-off answers or broken rules: flaky." in lines


def test_format_report_shows_the_table_the_recommendation_and_where_answers_are():
    summary = summarise(pd.DataFrame([row("cheap", 0.001234567, 3.14159)]))

    text = format_report(summary, ["Cheapest: cheap."], "results/x.jsonl")

    assert "cheap" in text and "0.001235" in text and "3.14" in text  # rounded table
    assert "- Cheapest: cheap." in text
    assert "Quality was not judged: read the answers in results/x.jsonl" in text


def test_summary_counts_answers_that_broke_a_rule_and_recommend_leaves_them_out():
    results = pd.DataFrame([
        row("clean", 0.002, 2.0), row("clean", 0.002, 2.0),
        row("cheap_but_wrong", 0.001, 1.0), row("cheap_but_wrong", 0.001, 1.0),
    ])
    results["rule_failures"] = ["", "", "missing '17%'", ""]

    summary = summarise(results)
    lines = recommend(summary)

    assert summary.loc["cheap_but_wrong", "rules_broken"] == 1
    assert summary.loc["clean", "rules_broken"] == 0
    assert lines[0].startswith("Cheapest: clean")
    assert "Left out because of errors, cut-off answers or broken rules: cheap_but_wrong." in lines


def test_recommend_adds_best_quality_and_the_trade_off_when_the_judge_ran():
    results = pd.DataFrame([row("cheap", 0.001, 1.0)] * 4 + [row("good", 0.010, 2.0)] * 4)
    summary = summarise(results)
    summary["judge_wins"] = [1, 3]  # cheap won 1 input, good won 3

    lines = recommend(summary, judged_inputs=4)

    assert "Best quality: good (picked by the judge on 3 of 4 inputs)." in lines
    assert any(
        line.startswith("Trade-off: good costs 10.0 times more than cheap") and "1 of 4" in line
        for line in lines
    )


def test_recommend_has_no_quality_lines_without_the_judge():
    summary = summarise(pd.DataFrame([row("cheap", 0.001, 1.0), row("good", 0.010, 2.0)]))

    assert not any("quality" in line.lower() for line in recommend(summary))


def test_the_report_says_when_quality_was_not_judged():
    summary = summarise(pd.DataFrame([row("cheap", 0.001, 1.0)]))

    assert "Quality was not judged" in format_report(summary, [], "results/x.jsonl")
    summary["judge_wins"] = [1]
    assert "Quality was not judged" not in format_report(summary, [], "results/x.jsonl")


def test_the_report_gives_the_judge_s_reason_when_it_could_not_answer():
    summary = summarise(pd.DataFrame([row("cheap", 0.001, 1.0)]))

    text = format_report(summary, [], "results/x.jsonl", judge_error="RateLimitError: busy")

    assert "the judge could not answer: RateLimitError: busy" in text
