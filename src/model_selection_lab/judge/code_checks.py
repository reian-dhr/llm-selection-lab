"""Code checks: rules a machine can verify, applied before any judge reads an answer.

A task (or one of its inputs) can list rules in its YAML file:

    checks:
      max_words: 80              # at most this many words
      must_include: ["17%"]      # each text must appear (capitals ignored)
      must_not_include: ["guarantee"]
      valid_json: true           # the whole answer must be JSON

An answer that breaks a rule is a failure whatever a judge thinks of it: a number that
is missing or an answer that is too long is a fact, not an opinion. Rules on an input are
added to the rules of the task; a limit on an input replaces the task's limit.
"""

import json

LIST_CHECKS = ("must_include", "must_not_include")
NUMBER_CHECKS = ("max_words",)
FLAG_CHECKS = ("valid_json",)


def validate_checks(checks, where: str) -> None:
    """Raise a clear ValueError if a check has an unknown name or the wrong kind of value.
    where: what the checks belong to, for the error message (for example "Input 2 in x.yaml")."""
    if not isinstance(checks, dict):
        raise ValueError(f"{where}: checks must be a set of named rules")
    known = LIST_CHECKS + NUMBER_CHECKS + FLAG_CHECKS
    for name, value in checks.items():
        if name not in known:
            raise ValueError(f"{where}: unknown check '{name}' (known: {', '.join(known)})")
        if name in LIST_CHECKS and not (
            isinstance(value, list) and all(isinstance(item, str) for item in value)
        ):
            raise ValueError(f"{where}: '{name}' must be a list of texts")
        if name in NUMBER_CHECKS and (not isinstance(value, int) or isinstance(value, bool)):
            raise ValueError(f"{where}: '{name}' must be a whole number")
        if name in FLAG_CHECKS and not isinstance(value, bool):
            raise ValueError(f"{where}: '{name}' must be true or false")


def merge_checks(task_checks: dict, input_checks: dict) -> dict:
    """The rules for one input: the task's rules, plus the input's own."""
    merged = dict(task_checks)
    for name, value in input_checks.items():
        if name in LIST_CHECKS:
            merged[name] = list(task_checks.get(name, [])) + value
        else:
            merged[name] = value
    return merged


def run_code_checks(text: str, checks: dict) -> list[str]:
    """Return what is wrong with the answer, one short sentence per broken rule.
    An empty list means every rule is followed (or there are none)."""
    failures = []
    words = len(text.split())
    if "max_words" in checks and words > checks["max_words"]:
        failures.append(f"more than {checks['max_words']} words ({words})")
    lowered = text.lower()
    for wanted in checks.get("must_include", []):
        if wanted.lower() not in lowered:
            failures.append(f"missing '{wanted}'")
    for banned in checks.get("must_not_include", []):
        if banned.lower() in lowered:
            failures.append(f"contains '{banned}'")
    if checks.get("valid_json"):
        try:
            json.loads(text)
        except ValueError:
            failures.append("not valid JSON")
    return failures
