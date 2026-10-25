"""Calling a model: everything needed to send one prompt to one model.

- model_registry.py: one entry per model (ID, prices, tier), which models a run uses,
  and the cost formula.
- model_client.py: run_model(), the one function that sends a prompt to any model.
- retry.py: retry logic shared by all providers.
- call_result.py: RunResult, what one call returns (including whether it was cut off).
"""
