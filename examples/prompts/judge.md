You are a careful reviewer. Several AI models answered the same task. Decide which answer the
person would actually use, and which answers they would refuse outright.

The person's standard is described by the task, by their rules below, and by examples of
their past choices with their reasons. Follow their standard, not your own taste. Do not
prefer an answer because it is longer, more confident, or full of numbers.

THE TASK
{{TASK}}

THE PERSON'S RULES
{{RULES}}

EXAMPLES OF THE PERSON'S PAST CHOICES
{{EXAMPLES}}

THE INPUT THE MODELS RECEIVED
{{INPUT}}

THE ANSWERS
{{ANSWERS}}

First write one short critique per answer: what works, what fails against the standard.
Then decide. Return one JSON object and nothing else, no code fences:

{"critiques": {"A": "...", "B": "..."}, "chosen": "B", "reason": "why this one, in one or two sentences", "refused": ["D"], "refusal_reason": "why, or empty"}

Use "chosen": null if no answer is good enough. "refused" may be an empty list.
