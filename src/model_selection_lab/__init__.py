"""LLM Selection Lab: compare LLMs on your own task (quality, cost, speed).

One sub-folder per stage of the pipeline (see docs/how_it_works.md for the full guide):
- runs/: write or load a task, run it on the chosen models, saving each answer.
- models/: send one prompt to one model (registry, client, retries, result).
- judge/: the user's picks, and the judge that learns from them to pick the best answer.
- report/: turn results into figures for a decision (summary, recommendation, job cost).
- ask.py: one request on every model, then the judge's pick (used by the web app).
- check.py: the whole chain in one call, from a plain description to a recommendation.
- mcp_server.py: the check as MCP tools, for an AI assistant.
- settings.py: every default value the tool assumes, in one place.
"""
