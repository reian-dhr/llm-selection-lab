"""Picks: the expert's choice among the answers of all models to the same input.

For each input, the expert sees every model's answer side by side, without knowing which
model wrote which, and records:
- the answer they would actually use, and why,
- any answer they would refuse outright, and why.

These picks are what the judge will be built from: the chosen answers show what "good"
looks like, the refused ones show what must never pass, and the reasons say why.

Picks for one check (one drafted task) are kept in one .jsonl file, one line per pick.
Picking again for the same input adds a new line, and the latest one counts.

This file also prepares what the app shows: the answers of each input, and the shuffled order
in which they appear (labelled A, B, C...), so the model behind each answer stays hidden.
"""

import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from model_selection_lab.settings import PICK_SHUFFLE_SEED


@dataclass
class Pick:
    """The expert's verdict on all the answers to one input."""

    input_id: str
    chosen_model: str | None  # the model whose answer they would use, None if none would do
    reason: str  # why that one (or why none), in their own words
    refused_models: list[str]  # answers they would refuse outright
    refusal_reason: str  # why, empty if nothing was refused
    shown_order: list[str]  # which model was shown as A, B, C... (to check position effects)
    timestamp: str  # when the pick was made, UTC
    results_file: str = ""  # the run whose answers were compared


def make_pick(
    input_id: str,
    chosen_model: str | None,
    reason: str,
    refused_models: list[str],
    refusal_reason: str,
    shown_order: list[str],
    results_file: str = "",
) -> Pick:
    """Build a Pick with the current time, refusing incomplete or contradictory ones."""
    if not reason.strip():
        raise ValueError("A pick needs a reason")
    if chosen_model is not None and chosen_model in refused_models:
        raise ValueError("The chosen answer cannot also be refused")
    if refused_models and not refusal_reason.strip():
        raise ValueError("Refused answers need a reason")
    return Pick(
        input_id=input_id,
        chosen_model=chosen_model,
        reason=reason.strip(),
        refused_models=list(refused_models),
        refusal_reason=refusal_reason.strip(),
        shown_order=list(shown_order),
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        results_file=results_file,
    )


def picks_path_for(task_name: str, folder: str = "data/picks") -> str:
    """Return where the picks for one check (one task) are kept."""
    return f"{folder}/{task_name}.jsonl"


def save_pick(path: str, pick: Pick) -> None:
    """Add one pick as one JSON line at the end of the file at path."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(asdict(pick), ensure_ascii=False) + "\n")


def load_picks(path: str) -> dict:
    """Read the picks file as a dictionary: input_id -> latest Pick.

    Returns an empty dictionary if nothing has been picked yet.
    """
    picks = {}
    if not Path(path).exists():
        return picks
    with open(path, encoding="utf-8") as file:
        for line in file:
            if line.strip():
                pick = Pick(**json.loads(line))
                picks[pick.input_id] = pick  # later lines replace earlier ones
    return picks


def answers_by_input(results_path: str) -> dict:
    """Read the successful answers of a run: input_id -> {model_key: saved line}."""
    with open(results_path, encoding="utf-8") as file:
        records = [json.loads(line) for line in file if line.strip()]
    by_input = {}
    for record in records:
        if record["error"] is None:
            by_input.setdefault(record["input_id"], {})[record["model_key"]] = record
    return by_input


def shown_order(input_id: str, model_keys: list[str]) -> list[str]:
    """Shuffle the models for one input, the same way every time (answer A, B, C...)."""
    order = sorted(model_keys)
    random.Random(f"{PICK_SHUFFLE_SEED}-{input_id}").shuffle(order)
    return order

