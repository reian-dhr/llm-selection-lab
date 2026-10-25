"""Tests for mcp_server.py: the tools, with fake models and temporary folders. No network."""

from functools import partial

from model_selection_lab import check, mcp_server
from test_check import TEMPLATE, fake_run


def use_fakes(monkeypatch, tmp_path):
    """Make the tools use fake models and temporary folders instead of real ones."""
    folders = {"tasks_folder": tmp_path / "tasks", "results_folder": tmp_path / "results"}
    monkeypatch.setattr(
        check, "prepare_check",
        partial(check.prepare_check, run=fake_run, template=TEMPLATE, **folders),
    )
    monkeypatch.setattr(
        check, "finish_check",
        partial(check.finish_check, run=fake_run, results_folder=folders["results_folder"]),
    )
    monkeypatch.setattr(
        check, "load_check",
        partial(check.load_check, results_folder=folders["results_folder"]),
    )


def test_preview_shows_the_cost_and_asks_before_running(monkeypatch, tmp_path):
    use_fakes(monkeypatch, tmp_path)

    text = mcp_server.preview_check("Summarise texts.")

    assert "Maximum cost" in text
    assert "ask before calling run_check" in text
    assert "short_summary_" in text  # the test file path, to pass to run_check


def test_run_returns_the_summary_and_a_recommendation(monkeypatch, tmp_path):
    use_fakes(monkeypatch, tmp_path)
    task_path = check.prepare_check("Summarise texts.").task_path

    text = mcp_server.run_check(task_path)

    assert "Results saved to" in text
    assert "Recommendation:" in text


def test_run_refuses_above_the_cap(monkeypatch, tmp_path):
    use_fakes(monkeypatch, tmp_path)
    outcome = check.prepare_check("Summarise texts.")
    outcome.max_cost_usd = mcp_server.MAX_RUN_COST_USD + 1
    monkeypatch.setattr(check, "load_check", lambda task_path: outcome)
    monkeypatch.setattr(check, "finish_check", lambda *args: 1 / 0)  # must never be reached

    text = mcp_server.run_check(outcome.task_path)

    assert text.startswith("Refused")
