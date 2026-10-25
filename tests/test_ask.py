"""Tests for ask.py: one request, fake models and a fake judge, temporary folders. No network."""

import pytest

from model_selection_lab.ask import (
    DEFAULT_PREFERENCE_REASON,
    INPUT_ID,
    ask,
    name_the_letters,
    prefer,
)
from model_selection_lab.judge.picks import load_picks, shown_order
from model_selection_lab.models.call_result import RunResult
from model_selection_lab.models.model_registry import select_models

JUDGE_ANSWER = '{"critiques": {}, "chosen": "A", "reason": "Clearest.", "refused": []}'


def fake_run(model_key, prompt, max_tokens):
    return RunResult(
        text=f"answer from {model_key}", input_tokens=10, output_tokens=5,
        latency_s=0.1, cost_usd=0.001,
    )


def fake_judge(model_key, prompt, max_tokens):
    return RunResult(
        text=JUDGE_ANSWER, input_tokens=10, output_tokens=5, latency_s=0.1, cost_usd=0.0
    )


def busy_judge(model_key, prompt, max_tokens):
    return RunResult(
        text="", input_tokens=0, output_tokens=0, latency_s=0.1, cost_usd=0.0,
        error="429 Rate limit",
    )


def folders(tmp_path):
    return {"results_folder": tmp_path / "results", "verdicts_folder": tmp_path / "verdicts"}


def test_every_standard_model_answers_and_the_judge_picks_one(tmp_path):
    outcome = ask("Write a haiku about rain.", run=fake_run, judge_run=fake_judge,
                  **folders(tmp_path))

    assert set(outcome.answers) == set(select_models())
    assert outcome.best_model == shown_order(INPUT_ID, list(outcome.answers))[0]  # answer A
    assert outcome.judge_reason == "Clearest."
    assert outcome.judge_error == ""


def test_the_request_is_sent_as_is(tmp_path):
    prompts = []

    def recording_run(model_key, prompt, max_tokens):
        prompts.append(prompt)
        return fake_run(model_key, prompt, max_tokens)

    ask("Write a haiku about rain.", run=recording_run, judge_run=fake_judge,
        **folders(tmp_path))

    assert set(prompts) == {"Write a haiku about rain."}


def test_a_busy_judge_still_returns_the_answers(tmp_path):
    outcome = ask("Write a haiku.", run=fake_run, judge_run=busy_judge, **folders(tmp_path))

    assert outcome.answers
    assert outcome.best_model is None
    assert "429" in outcome.judge_error


def test_refuses_above_the_cap_without_calling_any_model(tmp_path):
    def no_call(*args, **kwargs):
        raise AssertionError("no model should be called")

    with pytest.raises(ValueError, match="cap"):
        ask("Write a haiku.", run=no_call, judge_run=no_call, max_cost_usd=0.0,
            **folders(tmp_path))


def test_refuses_an_empty_request():
    with pytest.raises(ValueError, match="empty"):
        ask("   ")


def test_prefer_saves_a_pick_with_a_default_reason(tmp_path):
    outcome = ask("Write a haiku.", run=fake_run, judge_run=fake_judge, **folders(tmp_path))
    other = next(model for model in outcome.answers if model != outcome.best_model)

    path = prefer(outcome, other, picks_folder=tmp_path / "picks")

    pick = load_picks(path)[INPUT_ID]
    assert pick.chosen_model == other
    assert pick.reason == DEFAULT_PREFERENCE_REASON
    assert pick.results_file == outcome.results_path


def test_the_judges_letters_become_model_names():
    letter_to_model = {"A": "haiku-5-5", "B": "mistral-small-4"}

    text = name_the_letters("Answer A is clearer than answer B (B) and Apple.", letter_to_model)

    assert text == (
        "Claude Haiku 5.5 is clearer than Mistral Small 4 (Mistral Small 4) and Apple."
    )


def test_options_in_the_request_are_not_renamed():
    letter_to_model = {"A": "haiku-5-5", "B": "mistral-small-4"}

    text = name_the_letters("Option B is cheaper than option A.", letter_to_model)

    assert text == "Option B is cheaper than option A."


def test_premium_models_answer_only_when_asked(tmp_path):
    standard = ask("Write a haiku.", run=fake_run, judge_run=fake_judge, **folders(tmp_path))
    premium = ask("Write a haiku.", include_premium=True, run=fake_run, judge_run=fake_judge,
                  **folders(tmp_path))

    assert set(standard.answers) == set(select_models())
    assert set(premium.answers) == set(select_models(include_premium=True))
    assert set(premium.answers) > set(standard.answers)
