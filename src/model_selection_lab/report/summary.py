"""Summary and recommendation: what the results say, per model, in a few lines.

- summarise(): one row per model with calls, errors, cut-off answers, latency, tokens, cost.
- recommend(): short plain sentences on cost and speed. Quality is not part of it: in the
  web app it comes from the judge and the user's picks.
"""

import pandas as pd


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Turn a table of results (one row per call) into one row per model."""
    failed = results["error"].notna()
    # Results saved before the cut-off flag existed have no such column: count them as 0.
    cut_off = results["truncated"] if "truncated" in results.columns else False
    by_model = results.assign(failed=failed, cut_off=cut_off).groupby("model_key", sort=False)
    counts = by_model.agg(
        calls=("failed", "size"), errors=("failed", "sum"), cut_off=("cut_off", "sum")
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



def recommend(summary: pd.DataFrame) -> list[str]:
    """Return short sentences on which model is cheapest and fastest among the reliable ones.

    A model counts as reliable here if none of its calls failed and none of its answers was
    cut off. Quality is not part of this yet.
    """
    reliable = summary[(summary["errors"] == 0) & (summary["cut_off"] == 0)].dropna()
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
    if left_out:
        lines.append(f"Left out because of errors or cut-off answers: {', '.join(left_out)}.")
    return lines
