"""Tests for judge/judge.py: the judge's prompt, its verdicts, and agreement. No network."""

import json
from types import SimpleNamespace

import pytest

from model_selection_lab.judge.judge import (
    agreement,
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
    verdict = load_verdicts(verdicts_path_for("check_1", folder))["easy_01"]
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

    assert agreement(picks, verdicts) == {"compared": 2, "agreed": 1}
