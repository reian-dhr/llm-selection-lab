"""Summary and recommendation: what the results say, per model, in a few lines.

- with_rule_failures(): adds a column that names the task's code checks each answer broke.
- summarise(): one row per model with calls, errors, cut-off answers, broken rules, latency,
  tokens, cost.
- recommend(): short plain sentences on cost, speed and, when the judge ran, quality.
"""

import pandas as pd

from model_selection_lab.judge.code_checks import merge_checks, run_code_checks
from model_selection_lab.runs.task_loader import Task


def with_rule_failures(results: pd.DataFrame, task: Task) -> pd.DataFrame:
    """Add a rule_failures column: what each answer breaks in the task's code checks
    (see judge/code_checks.py), empty if nothing. Calls that failed have no answer to check."""
    checks = {item.id: merge_checks(task.checks, item.checks) for item in task.inputs}
    failures = [
        "" if isinstance(error, str)  # a failed call has no answer to check
        else "; ".join(run_code_checks(text, checks.get(input_id, task.checks)))
        for text, input_id, error in zip(results["text"], results["input_id"], results["error"])
    ]
    return results.assign(rule_failures=failures)


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Turn a table of results (one row per call) into one row per model."""
    failed = results["error"].notna()
    # Results saved before the cut-off flag existed have no such column: count them as 0.
    cut_off = results["truncated"] if "truncated" in results.columns else False
    # Without code checks there is no such column: no rule is broken.
    broken = results["rule_failures"].ne("") if "rule_failures" in results.columns else False
    by_model = results.assign(failed=failed, cut_off=cut_off, broken=broken).groupby(
        "model_key", sort=False
    )
    counts = by_model.agg(
        calls=("failed", "size"),
        errors=("failed", "sum"),
        cut_off=("cut_off", "sum"),
        rules_broken=("broken", "sum"),
    )

    # Averages use the successful calls only.
    averages = (
        results[~failed]
        .groupby("model_key", sort=False)
        .agg(
            median_latency_s=("latency_s", "median"),
            avg_input_tokens=("input_tokens", "mean"),
            avg_output_tokens=("output_tokens", "mean"),
            avg_cost_usd=("cost_usd", "mean"),
        )
    )
    # A model whose calls all failed has counts but no averages: its row shows "-".
    return counts.join(averages)



def recommend(summary: pd.DataFrame, judged_inputs: int | None = None) -> list[str]:
    """Return short sentences on which model is cheapest and fastest among the reliable ones.

    A model counts as reliable here if none of its calls failed, none of its answers was
    cut off and none broke a code check. When the judge has run (a judge_wins column), the
    best quality and the trade-off against the cheapest are added.
    """
    clean = (summary["errors"] == 0) & (summary["cut_off"] == 0) & (summary["rules_broken"] == 0)
    reliable = summary[clean].dropna()
    left_out = [model for model in summary.index if model not in reliable.index]

    lines = []
    if reliable.empty:
        lines.append("No model answered every input without errors or cut-offs.")
    else:
        cheapest = reliable["avg_cost_usd"].idxmin()
        fastest = reliable["median_latency_s"].idxmin()
        dearest = reliable["avg_cost_usd"].idxmax()
        ratio = reliable.loc[dearest, "avg_cost_usd"] / reliable.loc[cheapest, "avg_cost_usd"]
        lines.append(
            f"Cheapest: {cheapest} (${reliable.loc[cheapest, 'avg_cost_usd']:.5f} per answer). "
            f"The dearest, {dearest}, costs {ratio:.0f} times more."
        )
        speed = reliable.loc[fastest, "median_latency_s"]
        lines.append(f"Fastest: {fastest} ({speed:.1f} s per answer).")
        if "judge_wins" in reliable.columns and reliable["judge_wins"].max() > 0:
            best = reliable["judge_wins"].idxmax()
            lines.append(
                f"Best quality: {best} (picked by the judge on "
                f"{int(reliable.loc[best, 'judge_wins'])} of {judged_inputs} inputs)."
            )
            if best != cheapest:
                times = reliable.loc[best, "avg_cost_usd"] / reliable.loc[cheapest, "avg_cost_usd"]
                lines.append(
                    f"Trade-off: {best} costs {times:.1f} times more than {cheapest} per answer; "
                    f"the judge picked {cheapest} on {int(reliable.loc[cheapest, 'judge_wins'])} "
                    f"of {judged_inputs} inputs."
                )
    if left_out:
        lines.append(
            "Left out because of errors, cut-off answers or broken rules: "
            f"{', '.join(left_out)}."
        )
    return lines


DECIMALS = {"median_latency_s": 2, "avg_input_tokens": 0, "avg_output_tokens": 0, "avg_cost_usd": 6}


def format_report(
    summary: pd.DataFrame, recommendation: list[str], results_path: str, judge_error: str = ""
) -> str:
    """The summary table, the recommendation and a pointer to the answers, as plain text.
    The command line and the MCP server both show a check this way."""
    lines = [summary.round(DECIMALS).to_string(na_rep="-"), "", "Recommendation:"]
    lines += [f"- {line}" for line in recommendation]
    if "judge_wins" not in summary.columns:
        reason = f" (the judge could not answer: {judge_error})" if judge_error else ""
        lines.append(f"- Quality was not judged{reason}: read the answers in {results_path}")
    return "\n".join(lines)
