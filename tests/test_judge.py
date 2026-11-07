"""Tests for judge/judge.py: the judge's prompt, its verdicts, and agreement. No network."""

import json
import re
from types import SimpleNamespace

import pytest

from model_selection_lab.judge.judge import (
    agreement,
    swap_test,
    wilson_interval,
    build_judge_prompt,
    judge_check,
    load_rules,
    load_verdicts,
    parse_verdict,
    recent_examples,
    verdicts_path_for,
)
from model_selection_lab.judge.picks import make_pick, save_pick, shown_order
from model_selection_lab.models.call_result import RunResult
from model_selection_lab.runs.task_loader import Task, TaskInput

TEMPLATE = "T={{TASK}} R={{RULES}} E={{EXAMPLES}} I={{INPUT}} A={{ANSWERS}}"
MODELS = ["model-a", "model-b", "model-c"]


def write_run(path, task_name="check_1"):
    """A tiny results file: one input, three models."""
    lines = [
        {"task_name": task_name, "input_id": "easy_01", "model_key": m, "error": None,
         "text": f"answer from {m}"}
        for m in MODELS
    ]
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")


def test_rules_apply_by_the_word_in_their_file_name(tmp_path):
    (tmp_path / "slide_rules.md").write_text("Titles state the takeaway.", encoding="utf-8")
    (tmp_path / "general_rules.md").write_text("Never invent numbers.", encoding="utf-8")

    assert "Titles" in load_rules("Draft one slide", folder=tmp_path)
    assert "Titles" not in load_rules("Write an email", folder=tmp_path)
    assert "Never invent" in load_rules("Write an email", folder=tmp_path)


def test_prompt_holds_every_part_with_answers_lettered():
    prompt = build_judge_prompt(TEMPLATE, "task", "facts", ["first", "second"], "rules", "ex")

    assert "T=task R=rules E=ex I=facts" in prompt
    assert "Answer A:\nfirst" in prompt and "Answer B:\nsecond" in prompt


def test_parse_verdict_turns_letters_into_models():
    text = '{"critiques": {"A": "thin", "B": "clear"}, "chosen": "B", "reason": "clearest",' \
           ' "refused": ["A"], "refusal_reason": "too thin"}'

    verdict = parse_verdict(text, {"A": "model-a", "B": "model-b"})

    assert verdict["chosen_model"] == "model-b"
    assert verdict["refused_models"] == ["model-a"]
    assert verdict["critiques"] == {"model-a": "thin", "model-b": "clear"}


def test_parse_verdict_refuses_an_unknown_letter():
    with pytest.raises(ValueError, match="unknown answer"):
        parse_verdict('{"chosen": "Z"}', {"A": "model-a"})


def test_recent_examples_show_reasons_and_answer_excerpts(tmp_path):
    results = tmp_path / "check_1_run.jsonl"
    write_run(results)
    pick = make_pick("easy_01", "model-b", "Clear title.", ["model-c"], "Invents data.",
                     MODELS, results_file=str(results))
    save_pick(tmp_path / "picks" / "check_1.jsonl", pick)

    text = recent_examples(picks_folder=tmp_path / "picks")

    assert "Answer they chose (excerpt): answer from model-b" in text
    assert "Their reason: Clear title." in text
    assert "Why they refused it: Invents data." in text
    assert recent_examples(picks_folder=tmp_path / "picks", exclude_check="check_1") == (
        "No past choices yet."
    )


def test_judge_check_saves_a_verdict_per_input_and_skips_judged_ones(tmp_path):
    results = tmp_path / "check_1_run.jsonl"
    write_run(results)
    task = Task(name="check_1", description="Good answers are short.", instruction="Answer.",
                inputs=[TaskInput(id="easy_01", difficulty="easy", input="facts", reference=None)])
    outcome = SimpleNamespace(task=task, results_path=str(results))
    order = shown_order("easy_01", MODELS)
    calls = []

    def fake_run(model_key, prompt, max_tokens):
        calls.append(prompt)
        return RunResult(text='{"chosen": "A", "reason": "best", "refused": []}',
                         input_tokens=1, output_tokens=1, latency_s=0.1, cost_usd=0.0)

    folder = tmp_path / "verdicts"
    assert judge_check(outcome, run=fake_run, verdicts_folder=folder, template=TEMPLATE) == 1
    assert judge_check(outcome, run=fake_run, verdicts_folder=folder, template=TEMPLATE) == 0
    assert len(calls) == 1
    verdict = load_verdicts(verdicts_path_for("check_1_run", folder))["easy_01"]  # named after the run
    assert verdict.chosen_model == order[0]  # "A" is the first model of the shown order


def test_agreement_counts_inputs_where_judge_and_person_chose_the_same(tmp_path):
    picks, verdicts = tmp_path / "picks", tmp_path / "verdicts"
    save_pick(picks / "check_1.jsonl", make_pick("easy_01", "model-a", "ok", [], "", MODELS))
    save_pick(picks / "check_1.jsonl", make_pick("easy_02", "model-b", "ok", [], "", MODELS))
    verdicts.mkdir()
    lines = [
        {"input_id": i, "chosen_model": m, "reason": "", "refused_models": [], "refusal_reason": "",
         "critiques": {}, "judge_model": "j", "timestamp": "t", "results_file": "r"}
        for i, m in (("easy_01", "model-a"), ("easy_02", "model-c"))
    ]
    (verdicts / "check_1.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))

    result = agreement(picks, verdicts)
    assert (result["compared"], result["agreed"]) == (2, 1)


def test_judge_check_refuses_more_answers_than_it_has_letters(tmp_path):
    many = [f"model-{number}" for number in range(9)]
    results = tmp_path / "check_1_run.jsonl"
    lines = [{"task_name": "check_1", "input_id": "easy_01", "model_key": m, "error": None,
              "text": "x"} for m in many]
    results.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    task = Task(name="check_1", description="d", instruction="i",
                inputs=[TaskInput(id="easy_01", difficulty="easy", input="f", reference=None)])
    outcome = SimpleNamespace(task=task, results_path=str(results))

    with pytest.raises(ValueError, match="at most 8"):
        judge_check(outcome, run=lambda *args, **kwargs: 1 / 0, template=TEMPLATE,
                    verdicts_folder=tmp_path / "verdicts")


def test_load_verdicts_skips_an_unreadable_line_and_says_so(tmp_path, capsys):
    good = {"input_id": "easy_01", "chosen_model": "model-a", "reason": "", "refused_models": [],
            "refusal_reason": "", "critiques": {}, "judge_model": "j", "timestamp": "t",
            "results_file": "r"}
    path = tmp_path / "check_1.jsonl"
    path.write_text(json.dumps(good) + "\n{cut off\n" + json.dumps({"surprise": 1}) + "\n")

    verdicts = load_verdicts(path)

    assert list(verdicts) == ["easy_01"]
    assert capsys.readouterr().out.count("Skipped an unreadable line") == 2


def test_the_judge_is_told_which_answers_broke_a_code_check(tmp_path):
    results = tmp_path / "check_1_run.jsonl"
    write_run(results)
    task = Task(name="check_1", description="d", instruction="i", checks={"must_include": ["zzz"]},
                inputs=[TaskInput(id="easy_01", input="f")])
    outcome = SimpleNamespace(task=task, results_path=str(results))
    prompts = []

    def fake_run(model_key, prompt, max_tokens):
        prompts.append(prompt)
        return RunResult(text='{"chosen": "A", "reason": "r", "refused": []}',
                         input_tokens=1, output_tokens=1, latency_s=0.1, cost_usd=0.0)

    judge_check(outcome, run=fake_run, verdicts_folder=tmp_path / "v", template=TEMPLATE)

    assert prompts[0].count("[Automatic check failed: missing 'zzz']") == 3  # all three answers


def verdict_json(letter):
    return json.dumps({"chosen": letter, "reason": "r", "refused": []})


def content_judge(model_key, prompt, max_tokens):
    """A judge that follows the content: it always prefers the answer from model-b."""
    letter = re.search(r"Answer ([A-H]):\nanswer from model-b", prompt).group(1)
    return RunResult(text=verdict_json(letter), input_tokens=1, output_tokens=1,
                     latency_s=0.1, cost_usd=0.0)


def first_position_judge(model_key, prompt, max_tokens):
    """A judge with position bias: it always picks the answer shown first."""
    return RunResult(text=verdict_json("A"), input_tokens=1, output_tokens=1,
                     latency_s=0.1, cost_usd=0.0)


def judged_outcome(tmp_path, judge):
    results = tmp_path / "check_1_run.jsonl"
    write_run(results)
    task = Task(name="check_1", description="d", instruction="i",
                inputs=[TaskInput(id="easy_01", input="facts")])
    outcome = SimpleNamespace(task=task, results_path=str(results))
    judge_check(outcome, run=judge, verdicts_folder=tmp_path / "v", template=TEMPLATE)
    return outcome


def test_a_judge_that_follows_the_content_gives_the_same_pick_when_answers_are_reversed(tmp_path):
    outcome = judged_outcome(tmp_path, content_judge)

    result = swap_test(outcome, run=content_judge, verdicts_folder=tmp_path / "v", template=TEMPLATE)

    assert result == {"tested": 1, "consistent": 1, "always_first": 0}


def test_a_judge_that_always_picks_the_first_answer_is_caught_by_the_swap(tmp_path):
    outcome = judged_outcome(tmp_path, first_position_judge)

    result = swap_test(
        outcome, run=first_position_judge, verdicts_folder=tmp_path / "v", template=TEMPLATE
    )

    assert result == {"tested": 1, "consistent": 0, "always_first": 1}


def test_wilson_interval_widens_with_few_cases_and_handles_none():
    assert wilson_interval(0, 0) == (0.0, 1.0)
    low, high = wilson_interval(5, 10)
    assert 0.23 < low < 0.24 and 0.76 < high < 0.77
    small_low, small_high = wilson_interval(1, 2)
    assert (small_high - small_low) > (high - low)


def test_agreement_also_gives_the_chance_level_and_an_interval(tmp_path):
    picks, verdicts = tmp_path / "picks", tmp_path / "verdicts"
    save_pick(picks / "check_1.jsonl", make_pick("easy_01", "model-a", "ok", [], "", MODELS))
    save_pick(picks / "check_1.jsonl", make_pick("easy_02", "model-b", "ok", [], "", MODELS))
    verdicts.mkdir()
    lines = [
        {"input_id": i, "chosen_model": m, "reason": "", "refused_models": [], "refusal_reason": "",
         "critiques": {}, "judge_model": "j", "timestamp": "t", "results_file": "r"}
        for i, m in (("easy_01", "model-a"), ("easy_02", "model-c"))
    ]
    (verdicts / "check_1.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))

    result = agreement(picks, verdicts)

    assert (result["compared"], result["agreed"]) == (2, 1)
    assert result["chance"] == pytest.approx(2 / 3)  # a random judge agrees 1 in 3, twice
    assert result["low"] < 0.5 < result["high"]
