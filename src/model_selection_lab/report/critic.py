"""The critic: checks every claim of a report against the raw results.

It does not read the summary the report was built from. It goes back to the saved
results file, counts and averages again with plain Python, and compares with what the
report says. A claim is wrong if the recomputed value differs from the one shown, or if the
report text does not contain the shown value in that model's row (so an edit by hand is
caught as well as a bug in the summary).
"""

import json
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from model_selection_lab.judge.code_checks import merge_checks, run_code_checks
from model_selection_lab.judge.judge import load_verdicts, verdicts_path_for
from model_selection_lab.report.report import Claim, show


@dataclass
class Finding:
    """The critic's verdict on one claim."""

    claim: Claim
    actual: str  # what the raw results give, written the way the report writes it
    ok: bool


def recompute(model_rows: list[dict], metric: str, task, wins: Counter, model: str):
    """One measure for one model, from its raw result lines."""
    good = [row for row in model_rows if row["error"] is None]
    checks = {item.id: merge_checks(task.checks, item.checks) for item in task.inputs}
    if metric == "calls":
        return len(model_rows)
    if metric == "errors":
        return len(model_rows) - len(good)
    if metric == "cut_off":
        return sum(1 for row in model_rows if row.get("truncated") is True)
    if metric == "rules_broken":
        return sum(1 for row in good if run_code_checks(row["text"], checks.get(row["input_id"], task.checks)))
    if metric == "judge_wins":
        return wins[model]
    if not good:
        return None
    if metric == "avg_cost_usd":
        return statistics.mean(row["cost_usd"] for row in good)
    return statistics.median(row["latency_s"] for row in good)  # median_latency_s


def verify_report(
    markdown: str, claims: list[Claim], results_path: str, task, verdicts_folder: str
) -> list[Finding]:
    """Check every claim against the results file (and the judge's verdicts for this run)."""
    lines = [
        json.loads(line)
        for line in Path(results_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    verdicts = load_verdicts(verdicts_path_for(Path(results_path).stem, verdicts_folder))
    wins = Counter(verdict.chosen_model for verdict in verdicts.values())
    table_rows = {
        line.split("|")[1].strip(): line for line in markdown.splitlines() if line.startswith("| ")
    }
    findings = []
    for claim in claims:
        model_rows = [row for row in lines if row["model_key"] == claim.model]
        actual = show(claim.metric, recompute(model_rows, claim.metric, task, wins, claim.model))
        in_report = f" {claim.shown} " in table_rows.get(claim.model, "").replace("|", "  ") + " "
        findings.append(Finding(claim, actual, ok=actual == claim.shown and in_report))
    return findings


def critic_lines(findings: list[Finding]) -> list[str]:
    """The critic's result as short sentences: all fine, or each claim that is not."""
    wrong = [finding for finding in findings if not finding.ok]
    if not wrong:
        return [f"Critic: all {len(findings)} claims match the raw results."]
    lines = [f"Critic: {len(wrong)} of {len(findings)} claims do not match the raw results."]
    for finding in wrong:
        claim = finding.claim
        lines.append(
            f"- {claim.model}, {claim.metric}: the report says {claim.shown}, "
            f"the results give {finding.actual}."
        )
    return lines
