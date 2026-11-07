"""The report of a check, built from claims so that another piece of code can verify it.

Every number in the per-model table is a Claim: which model, which measure, and the value
exactly as written. report/critic.py recomputes each claim from the raw results file and
checks that the report really states it. The text of the recommendation is not a claim:
it is built from the same summary, and the table is where its numbers can be checked.
"""

import math
from dataclasses import dataclass

import pandas as pd

from model_selection_lab.report.summary import with_rule_failures
from model_selection_lab.runs.results_storage import read_results

# (measure, column title) in the order of the table. judge_wins only exists if the judge ran.
COLUMNS = [
    ("calls", "calls"),
    ("errors", "errors"),
    ("cut_off", "cut off"),
    ("rules_broken", "rules broken"),
    ("judge_wins", "judge wins"),
    ("avg_cost_usd", "avg cost"),
    ("median_latency_s", "median latency"),
]


@dataclass
class Claim:
    """One number of the report."""

    model: str
    metric: str  # one of the measures in COLUMNS
    shown: str  # the value exactly as written in the report
    value: float


def show(metric: str, value) -> str:
    """Write a value the way the report does; '-' when there is none (a model whose calls
    all failed has no average)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    if metric == "avg_cost_usd":
        return f"${value:.6f}"
    if metric == "median_latency_s":
        return f"{value:.2f} s"
    return str(int(value))


def failure_cases(results: pd.DataFrame) -> list[str]:
    """One line per answer that failed, was cut off, or broke a code check."""
    lines = []
    for _, row in results.iterrows():
        problems = []
        if isinstance(row["error"], str):
            problems.append(f"the call failed ({row['error'][:80]})")
        if row.get("truncated") is True:
            problems.append("cut off at the length limit")
        if row.get("rule_failures"):
            problems.append(f"broke a code check: {row['rule_failures']}")
        if problems:
            lines.append(f"- {row['model_key']} on {row['input_id']}: {'; '.join(problems)}")
    return lines or ["- None."]


def build_report(outcome) -> tuple[str, list[Claim]]:
    """The report as Markdown, and the claims it makes (for the critic)."""
    summary = outcome.summary
    columns = [(metric, title) for metric, title in COLUMNS if metric in summary.columns]
    claims = []
    rows = []
    for model, values in summary.iterrows():
        cells = []
        for metric, _ in columns:
            claim = Claim(model, metric, show(metric, values[metric]), values[metric])
            claims.append(claim)
            cells.append(claim.shown)
        rows.append("| " + " | ".join([model] + cells) + " |")

    results = with_rule_failures(read_results(outcome.results_path), outcome.task)
    header = "| model | " + " | ".join(title for _, title in columns) + " |"
    divider = "|" + "---|" * (len(columns) + 1)
    lines = [
        f"# Check: {outcome.task.name}",
        "",
        "## Recommendation",
        *[f"- {line}" for line in outcome.recommendation],
        "",
        "## Per model",
        header,
        divider,
        *rows,
        "",
        "## Failure cases",
        *failure_cases(results),
        "",
    ]
    return "\n".join(lines), claims
