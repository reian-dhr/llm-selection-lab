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

from pathlib import Path

from mcp.server import MCPServer

from model_selection_lab import check
from model_selection_lab.report.summary import format_report
from model_selection_lab.runs.task_drafter import describe_understanding
from model_selection_lab.runs.task_loader import TASK_FOLDERS
from model_selection_lab.settings import MAX_RUN_COST_USD

server = MCPServer("llm-selection-lab")


def is_task_file(task_path: str) -> bool:
    """True if the path is a YAML file inside one of the task folders (".." cannot escape)."""
    path = Path(task_path).resolve()
    return path.suffix == ".yaml" and any(
        path.is_relative_to(Path(folder).resolve()) for folder in TASK_FOLDERS
    )


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
    report = format_report(
        outcome.summary, outcome.recommendation, outcome.results_path, outcome.judge_error
    )
    critic = "\n".join(outcome.critic)
    return (
        f"Results saved to: {outcome.results_path}\nReport saved to: {outcome.report_path}\n\n"
        f"{report}\n\n{critic}"
    )


@server.tool()
def preview_check(description: str) -> str:
    """Draft a small test for a task and estimate what running it on the models would cost.

    Free: only a helper model is called. Returns the test file path to pass to run_check.
    """
    return describe_preview(check.prepare_check(description))


@server.tool()
def run_check(task_path: str) -> str:
    """Run a previewed test on the standard models and return cost, speed, quality (a judge
    rates the answers) and a recommendation. Costs money: only call it after the user agreed to the preview."""
    if not is_task_file(task_path):
        return f"Refused: {task_path} is not a test file in {' or '.join(TASK_FOLDERS)}."
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
