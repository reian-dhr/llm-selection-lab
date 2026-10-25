"""Tests for model_client.py: each provider function and the run_model dispatcher.

All tests use fake clients (see fakes.py). Nothing here touches the network.
"""

import anthropic
import openai
import pytest
from fakes import (
    ScriptedClient,
    fake_anthropic_response,
    fake_groq_response,
    fake_mistral_error,
    fake_mistral_response,
    fake_openai_response,
    fake_status_error,
)

from model_selection_lab.models import model_client as client_module
from model_selection_lab.models.model_client import (
    call_anthropic,
    call_groq,
    call_mistral,
    call_openai,
    run_model,
)
from model_selection_lab.settings import (
    MAX_ANSWER_TOKENS,
    MAX_RETRY_WAIT_S,
    THINKING_EXTRA_TOKENS,
)

# Token counts of one million make the expected cost easy to check by hand:
# cost = input price per million + output price per million.
ONE_MILLION = 1_000_000


# ---------- Anthropic ----------


def test_call_anthropic_returns_text_tokens_and_cost():
    client = ScriptedClient([fake_anthropic_response(ONE_MILLION, ONE_MILLION)])

    result = call_anthropic("haiku-5-5", "hello", system="be brief", client=client)

    assert result.error is None
    assert result.text == "hello"
    assert result.input_tokens == ONE_MILLION
    assert result.output_tokens == ONE_MILLION
    assert result.cost_usd == pytest.approx(0.10 + 0.50)  # haiku-5-5 prices per million
    sent = client.last_kwargs
    assert sent["model"] == "claude-haiku-5-5"
    assert sent["system"] == "be brief"
    assert sent["max_tokens"] == MAX_ANSWER_TOKENS
    assert sent["messages"] == [{"role": "user", "content": "hello"}]


@pytest.mark.parametrize(
    "model_key, expected_thinking_type",
    [("haiku-5-5", "disabled"), ("sonnet-5-5", "between_tools")],
)
def test_call_anthropic_turns_thinking_off_the_way_each_model_wants(
    model_key, expected_thinking_type
):
    client = ScriptedClient([fake_anthropic_response()])

    call_anthropic(model_key, "hi", client=client)

    assert client.last_kwargs["thinking"] == {"type": expected_thinking_type}


def test_call_anthropic_retries_a_rate_limit_then_succeeds():
    rate_limit = fake_status_error(anthropic.RateLimitError, 429)
    client = ScriptedClient([rate_limit, fake_anthropic_response()])
    waits = []

    result = call_anthropic("haiku-5-5", "hi", client=client, sleep=waits.append)

    assert result.error is None
    assert client.calls == 2
    assert waits == [1]


def test_call_anthropic_does_not_retry_a_bad_key():
    bad_key = fake_status_error(anthropic.AuthenticationError, 401)
    client = ScriptedClient([bad_key])

    result = call_anthropic("haiku-5-5", "hi", client=client, sleep=lambda s: None)

    assert "AuthenticationError" in result.error
    assert result.text == ""
    assert client.calls == 1


# ---------- OpenAI ----------


def test_call_openai_returns_text_tokens_and_cost():
    client = ScriptedClient([fake_openai_response(ONE_MILLION, ONE_MILLION)])

    result = call_openai("gpt-6-luna", "hello", system="be brief", client=client)

    assert result.error is None
    assert result.text == "hi from openai"
    assert result.input_tokens == ONE_MILLION
    assert result.output_tokens == ONE_MILLION
    assert result.cost_usd == pytest.approx(0.10 + 0.50)  # gpt-6-luna prices per million
    sent = client.last_kwargs
    assert sent["model"] == "gpt-6-luna"
    assert sent["input"] == "hello"
    assert sent["instructions"] == "be brief"
    assert sent["max_output_tokens"] == MAX_ANSWER_TOKENS
    assert sent["reasoning"] == {"effort": "none"}  # reasoning must be switched off


def test_call_openai_retries_a_rate_limit_then_succeeds():
    rate_limit = fake_status_error(openai.RateLimitError, 429)
    client = ScriptedClient([rate_limit, fake_openai_response()])
    waits = []

    result = call_openai("gpt-6-luna", "hi", client=client, sleep=waits.append)

    assert result.error is None
    assert client.calls == 2
    assert waits == [1]


def test_call_openai_does_not_retry_a_bad_key():
    bad_key = fake_status_error(openai.AuthenticationError, 401)
    client = ScriptedClient([bad_key])

    result = call_openai("gpt-6-luna", "hi", client=client, sleep=lambda s: None)

    assert "AuthenticationError" in result.error
    assert client.calls == 1


# ---------- Mistral ----------


def test_call_mistral_returns_text_tokens_and_cost():
    client = ScriptedClient([fake_mistral_response(ONE_MILLION, ONE_MILLION)])

    result = call_mistral("mistral-small-4", "hello", system="be brief", client=client)

    assert result.error is None
    assert result.text == "hi from mistral"
    assert result.input_tokens == ONE_MILLION
    assert result.output_tokens == ONE_MILLION
    assert result.cost_usd == pytest.approx(0.15 + 0.60)  # mistral-small-4 prices per million
    sent = client.last_kwargs
    assert sent["model"] == "mistral-small-2603"
    assert sent["messages"][0] == {"role": "system", "content": "be brief"}
    assert sent["messages"][1] == {"role": "user", "content": "hello"}
    assert sent["max_tokens"] == MAX_ANSWER_TOKENS
    assert sent["reasoning_effort"] == "none"


def test_call_mistral_retries_a_rate_limit_then_succeeds():
    client = ScriptedClient([fake_mistral_error(429), fake_mistral_response()])
    waits = []

    result = call_mistral("mistral-small-4", "hi", client=client, sleep=waits.append)

    assert result.error is None
    assert client.calls == 2
    assert waits == [1]


# ---------- run_model: picks the right function for the model's provider ----------


# --- Cut-off answers: each provider signals them its own way ---


def test_call_anthropic_flags_a_cut_off_answer():
    client = ScriptedClient([fake_anthropic_response(stop_reason="max_tokens")])

    result = call_anthropic("haiku-5-5", "hello", client=client)

    assert result.error is None  # a cut-off answer is still an answer, not a failure
    assert result.truncated is True


def test_call_openai_flags_a_cut_off_answer():
    client = ScriptedClient([fake_openai_response(incomplete_reason="max_output_tokens")])

    result = call_openai("gpt-6-luna", "hello", client=client)

    assert result.truncated is True


def test_call_mistral_flags_a_cut_off_answer():
    client = ScriptedClient([fake_mistral_response(finish_reason="length")])

    result = call_mistral("mistral-small-4", "hello", client=client)

    assert result.truncated is True


def test_complete_answers_are_not_flagged():
    anthropic_result = call_anthropic(
        "haiku-5-5", "hello", client=ScriptedClient([fake_anthropic_response()])
    )
    openai_result = call_openai(
        "gpt-6-luna", "hello", client=ScriptedClient([fake_openai_response()])
    )
    mistral_result = call_mistral(
        "mistral-small-4", "hello", client=ScriptedClient([fake_mistral_response()])
    )

    assert anthropic_result.truncated is False
    assert openai_result.truncated is False
    assert mistral_result.truncated is False


@pytest.mark.parametrize(
    "model_key, provider",
    [("haiku-5-5", "anthropic"), ("gpt-6-luna", "openai"), ("mistral-small-4", "mistral")],
)
def test_run_model_picks_the_function_for_the_models_provider(
    monkeypatch, model_key, provider
):
    calls = []

    def fake_function(model_key, prompt, system, max_tokens):
        calls.append((model_key, prompt, system, max_tokens))
        return "fake result"

    # monkeypatch swaps the dictionary entry for the length of this test only.
    monkeypatch.setitem(client_module.PROVIDER_FUNCTIONS, provider, fake_function)

    result = run_model(model_key, "hello", system="be brief", max_tokens=99)

    assert result == "fake result"
    assert calls == [(model_key, "hello", "be brief", 99)]


def test_run_model_gives_thinking_models_extra_room(monkeypatch):
    calls = []

    def fake_function(model_key, prompt, system, max_tokens):
        calls.append(max_tokens)
        return "fake result"

    monkeypatch.setitem(client_module.PROVIDER_FUNCTIONS, "anthropic", fake_function)

    run_model("haiku-5-5", "hello", max_tokens=100)  # thinking off
    run_model("opus-5-5", "hello", max_tokens=100)  # always thinks

    assert calls == [100, 100 + THINKING_EXTRA_TOKENS]


def test_run_model_rejects_a_provider_without_a_function(monkeypatch):
    monkeypatch.delitem(client_module.PROVIDER_FUNCTIONS, "openai")

    with pytest.raises(ValueError, match="openai"):
        run_model("gpt-6-luna", "hello")


def test_call_groq_returns_text_tokens_and_cost():
    client = ScriptedClient([fake_groq_response(ONE_MILLION, ONE_MILLION)])

    result = call_groq("qwen-3-8-27b", "hello", system="be brief", client=client)

    assert result.error is None
    assert result.text == "hi from groq"
    assert result.cost_usd == 0.0  # used on the free tier
    sent = client.last_kwargs
    assert sent["model"] == "qwen/qwen3.8-27b"
    assert sent["messages"][0] == {"role": "system", "content": "be brief"}
    assert sent["max_tokens"] == MAX_ANSWER_TOKENS


def test_call_groq_flags_a_cut_off_answer():
    client = ScriptedClient([fake_groq_response(finish_reason="length")])

    result = call_groq("qwen-3-8-27b", "hello", client=client)

    assert result.truncated is True


def test_a_rate_limit_waits_as_long_as_the_provider_asks():
    rate_limit = fake_status_error(openai.RateLimitError, 429, headers={"retry-after": "20"})
    client = ScriptedClient([rate_limit, fake_openai_response()])
    waits = []

    result = call_openai("gpt-6-luna", "hi", client=client, sleep=waits.append)

    assert result.error is None
    assert waits == [20.0]


def test_a_very_long_wait_is_capped():
    rate_limit = fake_status_error(openai.RateLimitError, 429, headers={"retry-after": "3600"})
    client = ScriptedClient([rate_limit, fake_openai_response()])
    waits = []

    call_openai("gpt-6-luna", "hi", client=client, sleep=waits.append)

    assert waits == [MAX_RETRY_WAIT_S]
