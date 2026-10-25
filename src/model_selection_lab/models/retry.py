"""Retry logic, shared by every provider.

Calls to an LLM API sometimes fail for a temporary reason (too many requests,
a busy server, a dropped connection). Trying again a moment later usually works.
Other failures are permanent (wrong API key, bad request) and retrying is pointless.

This file has two things:
- is_retryable(): says whether a failure is temporary or permanent.
- call_with_retries(): runs one model call, retries temporary failures, and always
  returns a RunResult (with the error filled in if every try failed) instead of crashing.

Provider-specific code lives in model_client.py. It only has to say how to make ONE attempt.
"""

import time

import anthropic
import httpx2
import openai
from mistralai.client.errors import MistralError

from model_selection_lab.models.call_result import RunResult
from model_selection_lab.models.model_registry import calculate_cost
from model_selection_lab.settings import MAX_RETRY_WAIT_S, MAX_TRIES, RETRY_WAITS_S

# Each provider's library has its own error classes. Some have the same names
# (both anthropic and openai have RateLimitError) but they are different classes,
# so we list them all here. Adding a provider means adding its two kinds of error.
#
# "Connection" errors: the request never got an answer (network down, timeout).
# Mistral's library does not wrap these, so they come through as raw httpx2 errors.
CONNECTION_ERRORS = (
    anthropic.APIConnectionError,
    openai.APIConnectionError,
    httpx2.TransportError,
)
# "Status" errors: the server answered with an error code (400, 401, 429, 500...).
STATUS_ERRORS = (anthropic.APIStatusError, openai.APIStatusError, MistralError)


def is_retryable(exception: Exception) -> bool:
    """True if trying the same call again might work (a temporary problem)."""
    # Connection errors are temporary. Timeouts count as connection errors.
    if isinstance(exception, CONNECTION_ERRORS):
        return True
    # 429 means "too many requests, slow down". 5xx means "the server has a problem".
    # Both usually clear up by themselves. (529 means "overloaded", also a 5xx.)
    if isinstance(exception, STATUS_ERRORS):
        return exception.status_code == 429 or exception.status_code >= 500
    # 400 (bad request), 401 (bad key), 404 (not found) and anything we do not
    # recognise: retrying would fail the same way, so do not.
    return False


def wait_before_retry(exception: Exception, attempt: int) -> float:
    """Seconds to wait before the next try.

    If the provider said how long to wait (the "retry-after" header, sent with rate limits),
    follow it, up to MAX_RETRY_WAIT_S. Otherwise use the usual short waits (1 s, then 2 s).
    """
    # Anthropic and OpenAI errors carry the reply as .response, Mistral's as .raw_response.
    response = getattr(exception, "response", None) or getattr(exception, "raw_response", None)
    headers = getattr(response, "headers", None) or {}
    try:
        asked = float(headers.get("retry-after", ""))
    except ValueError:
        asked = None
    if asked is not None and asked > 0:
        return min(asked, MAX_RETRY_WAIT_S)
    return RETRY_WAITS_S[attempt]


def call_with_retries(send_once, model, retryable, sleep=time.sleep) -> RunResult:
    """Run send_once() with retries and return a RunResult, the same for every provider.

    send_once: a function that makes ONE attempt and returns
        (text, input_tokens, output_tokens, truncated), where truncated says whether
        the answer was cut off. It raises an exception if the call fails.
    model: the ModelInfo from the registry (its prices are used to compute the cost).
    retryable: a function that takes an exception and says if a retry is worthwhile.
    sleep: how to wait between tries. Tests pass a fake one so they run instantly.

    On success: a RunResult with the text, tokens, latency, cost and the truncated flag.
    On failure (after the last try, or at once if the error is permanent): a RunResult
    with empty text, zero tokens, zero cost, and `error` set to a short message.
    """
    for attempt in range(MAX_TRIES):
        # The timer restarts on every try, so latency covers the last attempt only
        # and never includes the waiting between tries.
        start = time.perf_counter()
        try:
            text, input_tokens, output_tokens, truncated = send_once()
            break  # success: leave the loop
        except Exception as exc:
            latency = time.perf_counter() - start
            is_last_try = attempt == MAX_TRIES - 1
            if is_last_try or not retryable(exc):
                # Keep only the error type and a short message. Never headers or keys.
                error = f"{type(exc).__name__}: {str(exc)[:200]}"
                return RunResult(
                    text="",
                    input_tokens=0,
                    output_tokens=0,
                    latency_s=latency,
                    cost_usd=0.0,
                    error=error,
                )
            sleep(wait_before_retry(exc, attempt))
    latency = time.perf_counter() - start  # latency of the successful attempt

    cost = calculate_cost(
        input_tokens,
        output_tokens,
        model.input_price_per_mtok,
        model.output_price_per_mtok,
    )
    return RunResult(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_s=latency,
        cost_usd=cost,
        truncated=truncated,
    )
