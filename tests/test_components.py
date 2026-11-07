"""Tests for app/components.py: the helpers that make text safe to show and format costs.

The web app folder is on the test path (see pyproject.toml), so `components` imports directly.
"""

from components import as_html_text, as_markdown, format_cost


def test_html_in_model_text_is_shown_as_text_never_run():
    shown = as_html_text("<script>alert(1)</script> costs $5 & more")

    assert "<script>" not in shown
    assert "&lt;script&gt;" in shown
    assert "$" not in shown  # the dollar is written as an HTML code, so it is not maths
    assert "&amp;" in shown


def test_dollar_signs_are_escaped_in_text_but_not_inside_code():
    assert as_markdown("$12 and $30") == "\\$12 and \\$30"
    assert as_markdown("pay $5 `x = $y` then $6") == "pay \\$5 `x = $y` then \\$6"
    assert as_markdown("```\nprice = $5\n```") == "```\nprice = $5\n```"


def test_costs_are_written_in_dollars_or_cents():
    assert format_cost(1.234) == "$1.23"
    assert format_cost(0.0042) == "0.42¢"
    assert format_cost(0.00006) == "0.006¢"
