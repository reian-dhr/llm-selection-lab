"""The judge: guesses which answer you would pick, using your rules and your past picks.

For one test input, the judge sees the task, your rules, a few of your most recent picks
(with your reasons) as examples, and every model's answer labelled A, B, C... in a
shuffled order, without model names. It writes a short critique per answer, then picks one and lists any
it would refuse. It is a helper model from a family outside the comparison.

The judge does not train. It improves because every new pick you make becomes an example
for the next judgement. agreement() measures how often it matched your picks. In the web
app its pick is shown first, so your preference is given after seeing it.

Verdicts for one run are kept in data/verdicts/<results file name>.jsonl, one line per input.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

from model_selection_lab.judge.code_checks import merge_checks, run_code_checks
from model_selection_lab.judge.picks import answers_by_input, load_picks, shown_order
from model_selection_lab.models.model_client import run_model
from model_selection_lab.settings import (
    JUDGE_EXAMPLES,
    JUDGE_MAX_TOKENS,
    JUDGE_MODEL,
    JUDGE_PROMPT,
    JUDGE_PROMPT_EXAMPLE,
    JUDGE_RULES_FOLDER,
    JUDGE_SWAP_MAX_INPUTS,
)
from model_selection_lab.text_files import read_records, read_text

LETTERS = "ABCDEFGH"
EXCERPT_CHARS = 400  # how much of an example answer the judge is shown


@dataclass
class Verdict:
    """The judge's guess for one input."""

    input_id: str
    chosen_model: str | None  # None if the judge found no answer good enough
    reason: str
    refused_models: list[str]
    refusal_reason: str
    critiques: dict  # model_key -> short critique
    judge_model: str
    timestamp: str
    results_file: str


def load_rules(task_text: str, folder: str = JUDGE_RULES_FOLDER) -> str:
    """Return the rules that apply to this task: general_*.md, and <word>_*.md if the word
    (for example "slide") appears in the task."""
    lowered = task_text.lower()
    chosen = []
    for path in sorted(Path(folder).glob("*.md")):
        word = path.stem.split("_")[0]
        if word == "general" or word in lowered:
            chosen.append(path.read_text(encoding="utf-8").strip())
    return "\n\n".join(chosen) or "No written rules yet."


def results_file_of(pick, picks_path: Path) -> str:
    """Where the answers of a pick are. Old picks did not store it: derive it from the name."""
    if pick.results_file:
        return pick.results_file
    return str(Path("results") / (picks_path.stem.removesuffix("_picks") + ".jsonl"))


def recent_examples(
    limit: int = JUDGE_EXAMPLES, picks_folder: str = "data/picks", exclude_check: str = ""
) -> str:
    """Your most recent picks from other checks, written out as examples for the judge."""
    collected = []
    for picks_path in Path(picks_folder).glob("*.jsonl"):
        if picks_path.stem == exclude_check:
            continue
        for pick in load_picks(picks_path).values():
            collected.append((pick.timestamp, pick, results_file_of(pick, picks_path)))
    collected.sort(key=lambda item: item[0], reverse=True)

    examples = []
    for _, pick, results_file in collected:
        if len(examples) == limit:
            break
        if not Path(results_file).exists():
            continue
        answers = answers_by_input(results_file).get(pick.input_id, {})
        example = [f"Example {len(examples) + 1}:"]
        if pick.chosen_model in answers:
            excerpt = answers[pick.chosen_model]["text"][:EXCERPT_CHARS]
            example.append(f"Answer they chose (excerpt): {excerpt}")
        else:
            example.append("They found no answer good enough.")
        example.append(f"Their reason: {pick.reason}")
        for model in pick.refused_models:
            if model in answers:
                excerpt = answers[model]["text"][:EXCERPT_CHARS]
                example.append(f"Answer they refused (excerpt): {excerpt}")
        if pick.refusal_reason:
            example.append(f"Why they refused it: {pick.refusal_reason}")
        examples.append("\n".join(example))
    return "\n\n".join(examples) or "No past choices yet."


def build_judge_prompt(
    template, task_text, input_text, answers_in_order, rules, examples, rule_failures=None
):
    """Fill the judge's template. answers_in_order: the answer texts in A, B, C... order.
    rule_failures: for each answer, the code checks it broke (see code_checks.py), in the
    same order. The judge is told, because a broken rule is a fact it should weigh."""
    rule_failures = rule_failures or [[] for _ in answers_in_order]
    answers = "\n\n".join(
        f"Answer {LETTERS[number]}:\n{text}"
        + (f"\n[Automatic check failed: {'; '.join(broken)}]" if broken else "")
        for number, (text, broken) in enumerate(zip(answers_in_order, rule_failures))
    )
    filled = {
        "{{TASK}}": task_text,
        "{{RULES}}": rules,
        "{{EXAMPLES}}": examples,
        "{{INPUT}}": input_text,
        "{{ANSWERS}}": answers,
    }
    for placeholder, value in filled.items():
        template = template.replace(placeholder, value.strip())
    return template


def parse_verdict(text: str, letter_to_model: dict) -> dict:
    """Read the judge's JSON answer and turn letters back into models. Raises ValueError."""
    cleaned = "\n".join(line for line in text.strip().splitlines() if not line.startswith("```"))
    try:
        data = yaml.safe_load(cleaned)  # the YAML reader also reads JSON
    except yaml.YAMLError as problem:
        raise ValueError(f"The judge's answer is not valid JSON: {problem}") from problem
    if not isinstance(data, dict) or "chosen" not in data:
        raise ValueError("The judge's answer has no 'chosen' field")
    chosen = data["chosen"]
    if chosen is not None and chosen not in letter_to_model:
        raise ValueError(f"The judge chose an unknown answer: {chosen}")
    refused = [letter for letter in data.get("refused") or [] if letter in letter_to_model]
    critiques = data.get("critiques") or {}
    return {
        "chosen_model": letter_to_model.get(chosen),
        "reason": str(data.get("reason", "")),
        "refused_models": [letter_to_model[letter] for letter in refused],
        "refusal_reason": str(data.get("refusal_reason", "")),
        "critiques": {
            letter_to_model[letter]: str(text)
            for letter, text in critiques.items()
            if letter in letter_to_model
        },
    }


def verdicts_path_for(task_name: str, folder: str = "data/verdicts") -> str:
    """Where the judge's verdicts for one check are kept."""
    return f"{folder}/{task_name}.jsonl"


def load_verdicts(path: str) -> dict:
    """Read the verdicts of a check: input_id -> latest Verdict. Empty if none yet."""
    verdicts = {}
    if Path(path).exists():
        for verdict in read_records(path, lambda line: Verdict(**json.loads(line))):
            verdicts[verdict.input_id] = verdict
    return verdicts


def judge_input(
    task, task_input, answers: dict, order: list, template, task_text, rules, examples, run
) -> dict:
    """Ask the judge about one input, with the answers shown in the given order.

    answers: model_key -> saved line, for this input. Returns the parsed verdict (see
    parse_verdict). Raises RuntimeError if the judge could not answer, ValueError if it
    answered in an unusable way or there are more answers than letters.
    """
    if len(order) > len(LETTERS):
        raise ValueError(f"The judge can compare at most {len(LETTERS)} answers, got {len(order)}.")
    texts = [answers[model]["text"] for model in order]
    checks = merge_checks(task.checks, task_input.checks)
    broken = [run_code_checks(text, checks) for text in texts]
    prompt = build_judge_prompt(template, task_text, task_input.input, texts, rules, examples, broken)
    result = run(JUDGE_MODEL, prompt, max_tokens=JUDGE_MAX_TOKENS)
    if result.error is not None:
        raise RuntimeError(f"The judge could not answer: {result.error}")
    return parse_verdict(result.text, dict(zip(LETTERS, order)))


def task_text_of(task) -> str:
    """What the judge is told about the task: what good looks like, and the instruction."""
    return f"{task.description}\n\nInstruction given to the models:\n{task.instruction}"


def judge_check(outcome, run=run_model, verdicts_folder: str = "data/verdicts", template=None):
    """Judge every input of a run that has no verdict yet. Returns the number judged.

    run, verdicts_folder, template: only for tests, to use fakes and a temporary folder.
    Stops at the first failure (for example a rate limit): what was judged is kept.
    """
    if template is None:
        template = read_text(JUDGE_PROMPT, JUDGE_PROMPT_EXAMPLE)
    # Named after the run (the results file), so a new run is never judged from old verdicts.
    path = verdicts_path_for(Path(outcome.results_path).stem, verdicts_folder)
    done = load_verdicts(path)
    task_text = task_text_of(outcome.task)
    rules = load_rules(task_text)
    examples = recent_examples(exclude_check=outcome.task.name)
    answers = answers_by_input(outcome.results_path)
    judged = 0
    for task_input in outcome.task.inputs:
        if task_input.id in done or task_input.id not in answers:
            continue
        order = shown_order(task_input.id, list(answers[task_input.id]))
        verdict = Verdict(
            input_id=task_input.id,
            judge_model=JUDGE_MODEL,
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            results_file=outcome.results_path,
            **judge_input(
                outcome.task, task_input, answers[task_input.id], order,
                template, task_text, rules, examples, run,
            ),
        )
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as file:
            file.write(json.dumps(asdict(verdict), ensure_ascii=False) + "\n")
        judged += 1
    return judged


def swap_test(
    outcome, run=run_model, verdicts_folder: str = "data/verdicts", template=None,
    limit: int = JUDGE_SWAP_MAX_INPUTS,
) -> dict:
    """Ask the judge again about inputs it already judged, with the answers in reverse order.

    A judge that follows the content picks the same model both times. One with position
    bias flips with the order. Returns how many inputs were tested, how many gave the same
    pick, and how many times it chose the answer shown first in both orders.
    Costs one judge call per input tested (at most `limit`).
    """
    if template is None:
        template = read_text(JUDGE_PROMPT, JUDGE_PROMPT_EXAMPLE)
    verdicts = load_verdicts(verdicts_path_for(Path(outcome.results_path).stem, verdicts_folder))
    task_text = task_text_of(outcome.task)
    rules = load_rules(task_text)
    examples = recent_examples(exclude_check=outcome.task.name)
    answers = answers_by_input(outcome.results_path)
    tested = consistent = always_first = 0
    for task_input in outcome.task.inputs:
        verdict = verdicts.get(task_input.id)
        if tested == limit or verdict is None or verdict.chosen_model is None:
            continue
        order = shown_order(task_input.id, list(answers[task_input.id]))
        reversed_order = order[::-1]
        swapped = judge_input(
            outcome.task, task_input, answers[task_input.id], reversed_order,
            template, task_text, rules, examples, run,
        )["chosen_model"]
        tested += 1
        consistent += swapped == verdict.chosen_model
        always_first += verdict.chosen_model == order[0] and swapped == reversed_order[0]
    return {"tested": tested, "consistent": consistent, "always_first": always_first}


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """The range the true rate probably lies in (95% by default), given `successes` out of
    `total`. With few cases it is wide, which is the point: 1 of 2 says very little."""
    if total == 0:
        return 0.0, 1.0
    rate = successes / total
    centre = (rate + z * z / (2 * total)) / (1 + z * z / total)
    spread = z * ((rate * (1 - rate) / total + z * z / (4 * total * total)) ** 0.5) / (
        1 + z * z / total
    )
    return centre - spread, centre + spread


def agreement(picks_folder: str = "data/picks", verdicts_folder: str = "data/verdicts") -> dict:
    """Across all checks: on how many inputs the judge chose the same answer as you.

    Also gives `chance`, how many a judge choosing at random would have matched on average,
    and `low` and `high`, the range the true agreement rate probably lies in.
    """
    compared = agreed = 0
    chance = 0.0
    for verdicts_path in Path(verdicts_folder).glob("*.jsonl"):
        picks = load_picks(Path(picks_folder) / verdicts_path.name)
        for input_id, verdict in load_verdicts(verdicts_path).items():
            if input_id in picks:
                compared += 1
                agreed += verdict.chosen_model == picks[input_id].chosen_model
                chance += 1 / max(len(picks[input_id].shown_order), 1)
    low, high = wilson_interval(agreed, compared)
    return {"compared": compared, "agreed": agreed, "chance": chance, "low": low, "high": high}
