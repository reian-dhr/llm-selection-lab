"""Tests for judge/picks.py: saving and reloading the expert's picks."""

import pytest

import json

from model_selection_lab.judge.picks import (
    answers_by_input,
    load_picks,
    make_pick,
    picks_path_for,
    save_pick,
    shown_order,
)

ORDER = ["model-b", "model-a", "model-c"]


def test_make_pick_needs_a_reason():
    with pytest.raises(ValueError, match="needs a reason"):
        make_pick("easy_01", "model-a", "  ", [], "", ORDER)


def test_make_pick_refuses_choosing_a_refused_answer():
    with pytest.raises(ValueError, match="cannot also be refused"):
        make_pick("easy_01", "model-a", "Clearest.", ["model-a"], "Wrong number.", ORDER)


def test_make_pick_needs_a_reason_for_refusals():
    with pytest.raises(ValueError, match="Refused answers need a reason"):
        make_pick("easy_01", "model-a", "Clearest.", ["model-c"], "", ORDER)


def test_picks_path_is_named_after_the_check():
    path = picks_path_for("mcp_slide_20261016_101500")

    assert path == "data/picks/mcp_slide_20261016_101500.jsonl"


def test_answers_by_input_keeps_successful_answers_only(tmp_path):
    path = tmp_path / "run.jsonl"
    lines = [
        {"input_id": "easy_01", "model_key": "model-a", "error": None, "text": "A"},
        {"input_id": "easy_01", "model_key": "model-b", "error": "timeout", "text": ""},
    ]
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")

    assert list(answers_by_input(path)["easy_01"]) == ["model-a"]


def test_shown_order_is_shuffled_the_same_way_every_time():
    first = shown_order("easy_01", ["model-c", "model-a", "model-b"])

    assert sorted(first) == ["model-a", "model-b", "model-c"]
    assert shown_order("easy_01", ["model-b", "model-a", "model-c"]) == first


def test_save_then_load_picks_keeps_the_latest_per_input(tmp_path):
    path = tmp_path / "picks" / "run_picks.jsonl"  # the folder does not exist yet
    assert load_picks(path) == {}

    save_pick(path, make_pick("easy_01", "model-a", "Clearest title.", [], "", ORDER))
    refused = make_pick(
        "hard_01", None, "All miss the conflict.", ["model-c"], "Invents a total.", ORDER
    )
    save_pick(path, refused)
    save_pick(path, make_pick("easy_01", "model-b", "On second read, B is tighter.", [], "", ORDER))

    picks = load_picks(path)
    assert len(picks) == 2
    assert picks["easy_01"].chosen_model == "model-b"
    assert picks["hard_01"].chosen_model is None
    assert picks["hard_01"].refused_models == ["model-c"]
    assert picks["hard_01"].shown_order == ORDER
