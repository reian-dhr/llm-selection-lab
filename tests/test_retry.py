"""Tests for retry.py: which errors are temporary, and how the retry loop behaves."""

import anthropic
import httpx2
import openai
import pytest
from fakes import REQUEST, fake_mistral_error, fake_status_error

from model_selection_lab.models.model_registry import MODELS
from model_selection_lab.models.retry import call_with_retries, is_retryable

# ---------- is_retryable: temporary errors (True) vs permanent errors (False) ----------


@pytest.mark.parametrize(
    "exception",
    [
        fake_status_error(anthropic.RateLimitError, 429),
        fake_status_error(anthropic.InternalServerError, 500),
        fake_status_error(anthropic.APIStatusError, 529),  # "overloaded"
        anthropic.APITimeoutError(request=REQUEST),
        anthropic.APIConnectionError(request=REQUEST),
        fake_status_error(openai.RateLimitError, 429),
        fake_status_error(openai.InternalServerError, 500),
        openai.APITimeoutError(request=REQUEST),
        fake_mistral_error(429),
        fake_mistral_error(503),
        httpx2.ConnectError("down"),
        httpx2.ReadTimeout("slow"),
    ],
)
def test_temporary_errors_are_retryable(exception):
    assert is_retryable(exception) is True


@pytest.mark.parametrize(
    "exception",
    [
        fake_status_error(anthropic.BadRequestError, 400),
        fake_status_error(anthropic.AuthenticationError, 401),
        fake_status_error(anthropic.NotFoundError, 404),
        fake_status_error(openai.AuthenticationError, 401),
        fake_mistral_error(401),
        ValueError("something unknown"),
    ],
)
def test_permanent_or_unknown_errors_are_not_retryable(exception):
    assert is_retryable(exception) is False


# ---------- call_with_retries: the loop ----------


def run_with_retries(outcomes, retryable=lambda exc: True):
    """Run call_with_retries on a scripted list of outcomes.

    Each outcome is either an exception (the attempt fails) or a tuple
    (text, input_tokens, output_tokens, truncated) (the attempt succeeds).
    Returns the result, the outcomes that were actually played, and the waits.
    """
    played = []
    waits = []

    def send_once():
        outcome = outcomes[len(played)]
        played.append(outcome)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    result = call_with_retries(
        send_once, MODELS["haiku-5-5"], retryable, sleep=waits.append
    )
    return result, played, waits


def test_succeeds_after_two_failures_and_computes_the_cost():
    boom = ConnectionError("down")

    result, played, waits = run_with_retries([boom, boom, ("hello", 1_000_000, 1_000_000, False)])

    assert result.error is None
    assert result.text == "hello"
    assert result.cost_usd == pytest.approx(0.10 + 0.50)  # haiku-5-5 prices per million
    assert len(played) == 3
    assert waits == [1, 2]


def test_gives_up_after_three_tries_and_returns_an_error_result():
    boom = ConnectionError("down")

    result, played, waits = run_with_retries([boom, boom, boom])

    assert "ConnectionError" in result.error
    assert result.text == ""
    assert result.input_tokens == 0
    assert result.output_tokens == 0
    assert result.cost_usd == 0.0
    assert result.latency_s >= 0
    assert len(played) == 3
    assert waits == [1, 2]  # no waiting after the last failure


def test_does_not_retry_when_the_error_is_permanent():
    result, played, waits = run_with_retries(
        [ValueError("bad")], retryable=lambda exc: False
    )

    assert "ValueError" in result.error
    assert len(played) == 1
    assert waits == []
