"""Model Selection Lab: compare LLMs on your own task (quality, cost, speed).

One sub-folder per stage of the pipeline (see docs/how_it_works.md for the full guide):
- runs/: write or load a task, run it on the chosen models, saving each answer.
- models/: send one prompt to one model (registry, client, retries, result).
- judge/: decide whether each saved answer is good enough (the expert's picks so far).
- report/: turn results into figures for a decision (summary, recommendation, job cost).
- check.py: the whole chain in one call, from a plain description to a recommendation.
- settings.py: every default value the tool assumes, in one place.
"""
