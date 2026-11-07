"""Saving results: one model answer becomes one line in a .jsonl file.

A .jsonl file is a text file where every line is one small JSON object. We use it
because adding a result is just adding a line at the end: nothing already saved is
rewritten, so a run that stops halfway keeps everything it had finished.

This file has five things:
- ResultRecord: what one saved line contains.
- fingerprint_call(): a short fingerprint of a call (prompt, model, settings), to recognise
  the same call later.
- make_record(): builds a ResultRecord from a RunResult, and stamps the time.
- append_result(): writes one ResultRecord as one line at the end of the file.
- load_reusable_answers(): finds saved answers that can be reused instead of paid for again.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from model_selection_lab.models.call_result import RunResult
from model_selection_lab.models.model_registry import MODELS, ModelInfo
from model_selection_lab.settings import MAX_ANSWER_TOKENS, THINKING_EXTRA_TOKENS


@dataclass
class ResultRecord:
    """One saved line: which task, which input, which model, and what came back."""

    task_name: str
    input_id: str
    model_key: str  # the key of the model in the MODELS registry
    timestamp: str  # when the record was made, UTC, like "2026-10-13T14:05:09+00:00"
    # The fields below are copied from the RunResult of the call.
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    cost_usd: float
    error: str | None  # None means the call succeeded
    truncated: bool  # True if the answer was cut off at the length limit
    # fingerprint of the exact call (prompt, model, settings), see fingerprint_call()
    prompt_hash: str
    # What the call was priced with, so cost_usd can be checked after prices change.
    # Empty or 0 in lines saved before these fields existed.
    model_id: str = ""
    input_price_per_mtok: float = 0.0
    output_price_per_mtok: float = 0.0
    prompt: str = ""  # the exact text sent to the model (empty in older lines)


def fingerprint_call(prompt: str, model_key: str, models: dict = MODELS) -> str:
    """Return a short fingerprint of a call: the same call always gives the same code.

    It covers the prompt and everything that shapes the answer: the model's ID, whether it
    thinks, and the answer length limit. If any of them changes, an answer saved earlier is
    never reused for the new call. A key missing from the registry (only in tests) is
    fingerprinted by its name alone.
    models: the registry to read the model from. Tests pass a small fake one.
    """
    model = models.get(model_key)
    if model is None:
        settings = model_key
    else:
        limit = MAX_ANSWER_TOKENS + (THINKING_EXTRA_TOKENS if model.reasoning else 0)
        settings = f"{model.model_id}|{model.reasoning}|{limit}"
    return hashlib.sha256(f"{settings}|{prompt}".encode("utf-8")).hexdigest()[:16]


def make_record(
    task_name: str,
    input_id: str,
    model_key: str,
    result: RunResult,
    prompt_hash: str = "",
    model: ModelInfo | None = None,
    prompt: str = "",
) -> ResultRecord:
    """Combine the labels of a call with its RunResult, and add the current time and, if the
    model is given, its ID and prices, and the prompt that was sent."""
    return ResultRecord(
        task_name=task_name,
        input_id=input_id,
        model_key=model_key,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        text=result.text,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_s=result.latency_s,
        cost_usd=result.cost_usd,
        error=result.error,
        truncated=result.truncated,
        prompt_hash=prompt_hash,
        model_id=model.model_id if model else "",
        input_price_per_mtok=model.input_price_per_mtok if model else 0.0,
        output_price_per_mtok=model.output_price_per_mtok if model else 0.0,
        prompt=prompt,
    )


def append_result(path: str, record: ResultRecord) -> None:
    """Add one record as one JSON line at the end of the file at path.

    Creates the file (and its folder) if they do not exist yet.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(asdict(record), ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as file:
        file.write(line + "\n")


def load_reusable_answers(results_folder: str, task_name: str, max_age_days: int) -> dict:
    """Return the saved answers of a task that can be reused instead of calling a model again.

    The result is a dictionary: (input_id, model_key, prompt_hash) -> ResultRecord.
    An answer is reusable only if the call succeeded, the answer was not cut off, it has a
    prompt fingerprint, and it is at most max_age_days old. The newest one wins.
    """
    oldest_allowed = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    reusable = {}
    for path in sorted(Path(results_folder).glob(f"{task_name}_*.jsonl")):
        with open(path, encoding="utf-8") as file:
            lines = [json.loads(line) for line in file if line.strip()]
        for line in lines:
            # Results saved before these fields existed are not reusable.
            if not line.get("prompt_hash") or line.get("truncated") is not False:
                continue
            if line["error"] is not None:
                continue
            if datetime.fromisoformat(line["timestamp"]) < oldest_allowed:
                continue
            key = (line["input_id"], line["model_key"], line["prompt_hash"])
            if key not in reusable or line["timestamp"] > reusable[key].timestamp:
                reusable[key] = ResultRecord(**line)
    return reusable


def read_results(path: str) -> pd.DataFrame:
    """Read a results file as a table, one row per saved answer.

    Types are not guessed: left to itself pandas turns a column of answers such as "42" into
    numbers, and every text rule then fails on a number."""
    return pd.read_json(path, lines=True, dtype=False)
