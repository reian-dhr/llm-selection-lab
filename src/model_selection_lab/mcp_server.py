"""MCP server: lets an AI assistant (Claude Code, Claude Desktop...) run a check on request.

MCP (Model Context Protocol) is a standard way to offer tools to an assistant. The assistant
reads each tool's name, description and inputs, and decides when to call it.

The server offers three tools, all thin wrappers over check.py:
- preview_check: draft a test from a description and estimate its cost. Free (helper only).
- run_check: run a previewed test on the models. This is the only tool that spends money.
- list_checks: the past checks, newest first.

Preview and run are separate on purpose: the assistant has to show the drafted test and its
maximum cost before it can spend anything, and run_check refuses anything above the cap.
Premium models are never used from here; use the web app or the command line for those.

Start it from the project folder (so data/, tasks/ and results/ are found):
  uv run python -m model_selection_lab.mcp_server
"""

from mcp.server import MCPServer

from model_selection_lab import check
from model_selection_lab.runs.task_drafter import describe_understanding
from model_selection_lab.settings import MAX_RUN_COST_USD

server = MCPServer("model-selection-lab")


def describe_preview(outcome) -> str:
    """The drafted test and its cost, as text for the assistant to show the user."""
    lines = describe_understanding(outcome.drafted)
    lines += [
        "",
        f"Test saved to: {outcome.task_path} ({len(outcome.task.inputs)} inputs)",
        f"Models: {', '.join(outcome.model_keys)}",
        f"Maximum cost: about ${outcome.max_cost_usd:.2f} (cap ${MAX_RUN_COST_USD:.2f})",
        "Nothing was spent. Show this to the user and ask before calling run_check.",
    ]
    return "\n".join(lines)


def describe_run(outcome) -> str:
    """The summary table and the recommendation, as text."""
    decimals = {
        "median_latency_s": 2, "avg_input_tokens": 0, "avg_output_tokens": 0, "avg_cost_usd": 6
    }
    lines = [
        f"Results saved to: {outcome.results_path}",
        "",
        outcome.summary.round(decimals).to_string(na_rep="-"),
        "",
        "Recommendation:",
    ]
    lines += [f"- {line}" for line in outcome.recommendation]
    lines.append("- Quality: pick the best answer blind in the web app (app/main.py).")
    return "\n".join(lines)


@server.tool()
def preview_check(description: str) -> str:
    """Draft a small test for a task and estimate what running it on the models would cost.

    Free: only a helper model is called. Returns the test file path to pass to run_check.
    """
    return describe_preview(check.prepare_check(description))


@server.tool()
def run_check(task_path: str) -> str:
    """Run a previewed test on the standard models and return cost, speed and a
    recommendation. Costs money: only call it after the user agreed to the preview."""
    outcome = check.load_check(task_path)
    if outcome.max_cost_usd > MAX_RUN_COST_USD:
        return (
            f"Refused: this run could cost ${outcome.max_cost_usd:.2f}, "
            f"above the cap of ${MAX_RUN_COST_USD:.2f}."
        )
    return describe_run(check.finish_check(outcome))


@server.tool()
def list_checks() -> list[dict]:
    """The past checks, newest first: name, goal, test file and latest results file."""
    return check.list_checks()


if __name__ == "__main__":
    server.run()  # stdio: the assistant starts this process and talks to it through pipes
