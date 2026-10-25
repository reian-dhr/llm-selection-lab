"""Fake building blocks for the tests, so that no test ever uses the network.

- Fake errors: real error objects from each provider's library, built by hand.
- ScriptedClient: pretends to be a provider's client and plays back a list of
  outcomes (an error to raise, or a response to return), counting the calls.
- fake_*_response(): what each provider's successful answer looks like, reduced
  to the few fields our code reads.

IMPORTANT: a test that triggers a retry must pass `sleep=sleeps.append` (or any fake),
otherwise the real code really waits 1 and 2 seconds.
"""

from types import SimpleNamespace

import httpx2
from mistralai.client.errors import MistralError

REQUEST = httpx2.Request("POST", "https://example.com")  # fake request, never sent


def fake_status_error(error_class, status_code, headers=None):
    """Build an Anthropic or OpenAI error that carries an HTTP status code (and headers)."""
    response = httpx2.Response(status_code, request=REQUEST, headers=headers or {})
    return error_class("fake", response=response, body=None)


def fake_mistral_error(status_code):
    """Build a Mistral error that carries an HTTP status code."""
    return MistralError("fake", raw_response=httpx2.Response(status_code, request=REQUEST))


class ScriptedClient:
    """A fake client. Each call plays the next outcome from the list.

    It answers to all three providers' call styles:
    client.messages.create (Anthropic), client.responses.create (OpenAI),
    client.chat.complete (Mistral).
    """

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0  # how many times the model was called
        self.last_kwargs = None  # what the last call was sent, to check the request
        self.messages = self
        self.responses = self
        self.chat = self
        self.completions = self  # OpenAI chat style: client.chat.completions.create

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def complete(self, **kwargs):
        return self.create(**kwargs)


def fake_anthropic_response(input_tokens=10, output_tokens=5, stop_reason="end_turn"):
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=[SimpleNamespace(type="text", text="hello")],
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def fake_openai_response(input_tokens=10, output_tokens=5, incomplete_reason=None):
    details = SimpleNamespace(reason=incomplete_reason) if incomplete_reason else None
    return SimpleNamespace(
        incomplete_details=details,
        output_text="hi from openai",
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def fake_mistral_response(input_tokens=10, output_tokens=5, finish_reason="stop"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="hi from mistral"), finish_reason=finish_reason
            )
        ],
        usage=SimpleNamespace(prompt_tokens=input_tokens, completion_tokens=output_tokens),
    )


def fake_groq_response(input_tokens=10, output_tokens=5, finish_reason="stop"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="hi from groq"), finish_reason=finish_reason
            )
        ],
        usage=SimpleNamespace(prompt_tokens=input_tokens, completion_tokens=output_tokens),
    )
