"""Tests for judge/code_checks.py: rules a machine can check before any judge reads an answer."""

import pytest

from model_selection_lab.judge.code_checks import merge_checks, run_code_checks, validate_checks


def test_an_answer_that_follows_every_rule_has_no_failures():
    checks = {"max_words": 5, "must_include": ["17%"], "must_not_include": ["guarantee"]}

    assert run_code_checks("Revenue grew 17% this year.", checks) == []


def test_each_broken_rule_is_named():
    checks = {"max_words": 3, "must_include": ["17%", "2026"], "must_not_include": ["Guarantee"]}

    failures = run_code_checks("We guarantee revenue grew 17% this year", checks)

    assert "more than 3 words (7)" in failures
    assert "missing '2026'" in failures
    assert "contains 'Guarantee'" in failures  # compared without caring about capitals
    assert not any("17%" in failure for failure in failures)


def test_valid_json_accepts_json_and_refuses_prose():
    assert run_code_checks('{"a": 1}', {"valid_json": True}) == []
    assert run_code_checks("Sure! Here it is.", {"valid_json": True}) == ["not valid JSON"]


def test_no_checks_means_no_failures():
    assert run_code_checks("anything", {}) == []


def test_a_misspelt_check_is_refused_when_the_task_is_loaded():
    with pytest.raises(ValueError, match="max_word"):
        validate_checks({"max_word": 5}, "Task x")


def test_check_values_must_have_the_right_shape():
    with pytest.raises(ValueError, match="must_include"):
        validate_checks({"must_include": "17%"}, "Task x")  # a list is needed
    with pytest.raises(ValueError, match="max_words"):
        validate_checks({"max_words": "five"}, "Task x")


def test_an_input_adds_to_the_task_rules_and_can_change_a_limit():
    task_checks = {"max_words": 80, "must_include": ["a"]}
    input_checks = {"max_words": 40, "must_include": ["b"], "valid_json": True}

    merged = merge_checks(task_checks, input_checks)

    assert merged == {"max_words": 40, "must_include": ["a", "b"], "valid_json": True}
