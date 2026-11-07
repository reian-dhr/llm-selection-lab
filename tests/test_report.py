"""Tests for report/report.py and report/critic.py: a report built from claims, and a critic
that recomputes every claim from the raw results file. Fake models, no network."""

import json

from test_check import JUDGE_VERDICT, TEMPLATE, fake_run

from model_selection_lab.check import run_check
from model_selection_lab.report.critic import critic_lines, verify_report
from model_selection_lab.report.report import build_report


def judged_run(model_key, prompt, max_tokens):
    result = fake_run(model_key, prompt, max_tokens)
    if "Answer A:" in prompt:
        result.text = JUDGE_VERDICT
    return result


def make_outcome(tmp_path):
    return run_check(
        "Summarise texts.", run=judged_run, template=TEMPLATE, judge_template="{{ANSWERS}}",
        tasks_folder=tmp_path / "tasks", results_folder=tmp_path / "results",
        verdicts_folder=tmp_path / "verdicts",
    )


def verify(outcome, markdown, claims, tmp_path):
    return verify_report(
        markdown, claims, outcome.results_path, outcome.task, str(tmp_path / "verdicts")
    )


def test_the_report_has_a_recommendation_a_row_per_model_and_failure_cases(tmp_path):
    outcome = make_outcome(tmp_path)

    markdown, claims = build_report(outcome)

    assert "## Recommendation" in markdown and "## Per model" in markdown
    assert "## Failure cases" in markdown
    for model in outcome.model_keys:
        assert f"| {model} |" in markdown
    assert {claim.metric for claim in claims} >= {"calls", "avg_cost_usd", "judge_wins"}
    assert len(claims) == len(outcome.model_keys) * 7


def test_the_critic_finds_nothing_wrong_in_an_untouched_report(tmp_path):
    outcome = make_outcome(tmp_path)
    markdown, claims = build_report(outcome)

    findings = verify(outcome, markdown, claims, tmp_path)

    assert all(finding.ok for finding in findings)
    assert critic_lines(findings)[0].startswith("Critic: all 28 claims match")


def test_the_critic_catches_a_number_edited_in_the_report(tmp_path):
    outcome = make_outcome(tmp_path)
    markdown, claims = build_report(outcome)
    cost = next(c for c in claims if c.metric == "avg_cost_usd").shown
    tampered = markdown.replace(cost, "$0.000001", 1)

    findings = verify(outcome, tampered, claims, tmp_path)

    assert [f for f in findings if not f.ok][0].claim.metric == "avg_cost_usd"
    assert "do not match" in critic_lines(findings)[0]


def test_the_critic_catches_results_that_changed_after_the_report(tmp_path):
    outcome = make_outcome(tmp_path)
    markdown, claims = build_report(outcome)
    lines = [json.loads(line) for line in open(outcome.results_path, encoding="utf-8")]
    lines[0]["cost_usd"] = 5.0  # a different cost than the report states
    with open(outcome.results_path, "w", encoding="utf-8") as file:
        file.write("".join(json.dumps(line) + "\n" for line in lines))

    findings = verify(outcome, markdown, claims, tmp_path)

    wrong = [finding for finding in findings if not finding.ok]
    assert wrong and wrong[0].claim.model == lines[0]["model_key"]
