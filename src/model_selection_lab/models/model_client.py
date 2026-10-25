"""Talking to the LLM providers.

One function per provider (call_anthropic, call_openai, call_mistral, call_groq), plus run_model(),
the single entry point the rest of the tool should use.

Every provider function follows the same recipe:
1. Look up the model in the registry (model_registry.py).
2. Build the request the way that provider's library wants it.
3. Define send_once(): ONE attempt, returning (text, input_tokens, output_tokens, truncated).
4. Hand send_once to call_with_retries() (retry.py), which does the retrying, the timing
   and the cost, and returns a RunResult.

Only steps 2 and 3 differ between providers. That is the whole point of this layout.
"""

import os
import time

from anthropic import Anthropic
from dotenv import load_dotenv
from mistralai.client import Mistral
from openai import OpenAI

from model_selection_lab.models.call_result import RunResult
from model_selection_lab.models.model_registry import MODELS
from model_selection_lab.models.retry import call_with_retries, is_retryable
from model_selection_lab.settings import MAX_ANSWER_TOKENS, THINKING_EXTRA_TOKENS

# API keys live in the .env file. Python does not read that file by itself,
# so this copies the keys into the environment where the libraries look for them.
load_dotenv()


def call_anthropic(
    model_key: str,
    prompt: str,
    system: str | None = None,
    max_tokens: int = MAX_ANSWER_TOKENS,
    client: Anthropic | None = None,
    sleep=time.sleep,
) -> RunResult:
    """Send one prompt to an Anthropic (Claude) model and return a RunResult.

    model_key: the model's name in the registry, for example "haiku-5-5".
    system: optional instructions that frame the whole conversation.
    client, sleep: only for tests. They let a test pass a fake client and a fake
        sleep so that no network is used and no time is spent waiting.
    """
    model = MODELS[model_key]
    if client is None:
        # max_retries=0: the library retries 2 times on its own by default, but we
        # have our own retry loop and want it to be the only one.
        # Source: https://platform.claude.com/docs/en/cli-sdks-libraries/sdks/python#retries
        client = Anthropic(max_retries=0)  # reads ANTHROPIC_API_KEY from the environment

    # Only send optional settings when we have them.
    options = {}
    if system is not None:
        options["system"] = system
    if not model.reasoning:
        # Turn thinking off. Models want different words for "off" (see model_registry.py).
        options["thinking"] = {"type": model.thinking_off}

    def send_once():
        response = client.messages.create(
            model=model.model_id,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **options,
        )
        # The answer comes as a list of blocks. Keep the text ones and join them.
        text = "".join(block.text for block in response.content if block.type == "text")
        # Cut off: the answer hit our length limit, or the model ran out of room.
        truncated = response.stop_reason in ("max_tokens", "model_context_window_exceeded")
        return text, response.usage.input_tokens, response.usage.output_tokens, truncated

    return call_with_retries(send_once, model, is_retryable, sleep)


def call_openai(
    model_key: str,
    prompt: str,
    system: str | None = None,
    max_tokens: int = MAX_ANSWER_TOKENS,
    client: OpenAI | None = None,
    sleep=time.sleep,
) -> RunResult:
    """Send one prompt to an OpenAI model and return a RunResult.

    Same inputs and behaviour as call_anthropic. Uses OpenAI's "Responses API",
    which the model page recommends: https://developers.openai.com/api/docs/models/gpt-6-luna
    """
    model = MODELS[model_key]
    if client is None:
        # max_retries=0: same reason as in call_anthropic.
        # Source: https://developers.openai.com/api/reference/python
        client = OpenAI(max_retries=0)  # reads OPENAI_API_KEY from the environment

    # Only send optional settings when we have them.
    options = {}
    if system is not None:
        options["instructions"] = system  # OpenAI's word for the system prompt
    if not model.reasoning:
        # The default effort is "medium", so we must send "none" to turn reasoning off.
        options["reasoning"] = {"effort": "none"}

    def send_once():
        response = client.responses.create(
            model=model.model_id,
            input=prompt,
            max_output_tokens=max_tokens,  # OpenAI counts reasoning tokens in this limit
            **options,
        )
        # Cut off: OpenAI marks the response as incomplete and gives the reason.
        details = response.incomplete_details
        truncated = details is not None and details.reason == "max_output_tokens"
        usage = response.usage
        return response.output_text, usage.input_tokens, usage.output_tokens, truncated

    return call_with_retries(send_once, model, is_retryable, sleep)


def call_mistral(
    model_key: str,
    prompt: str,
    system: str | None = None,
    max_tokens: int = MAX_ANSWER_TOKENS,
    client: Mistral | None = None,
    sleep=time.sleep,
) -> RunResult:
    """Send one prompt to a Mistral model and return a RunResult.

    Same inputs and behaviour as call_anthropic. Mistral's library does not retry
    by itself, so there is nothing to switch off.
    """
    model = MODELS[model_key]
    if client is None:
        client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))

    # Mistral takes the system prompt as the first message of the conversation.
    messages = [{"role": "user", "content": prompt}]
    if system is not None:
        messages.insert(0, {"role": "system", "content": system})

    # Only send optional settings when we have them.
    options = {}
    if not model.reasoning:
        options["reasoning_effort"] = "none"

    def send_once():
        response = client.chat.complete(
            model=model.model_id,
            messages=messages,
            max_tokens=max_tokens,
            **options,
        )
        choice = response.choices[0]
        content = choice.message.content
        if not isinstance(content, str):  # sometimes a list of chunks: keep the text ones
            content = "".join(chunk.text for chunk in content if chunk.type == "text")
        # Cut off: the answer hit our length limit, or the model's own limit.
        truncated = choice.finish_reason in ("length", "model_length")
        usage = response.usage
        return content, usage.prompt_tokens, usage.completion_tokens, truncated

    return call_with_retries(send_once, model, is_retryable, sleep)


def call_groq(
    model_key: str,
    prompt: str,
    system: str | None = None,
    max_tokens: int = MAX_ANSWER_TOKENS,
    client: OpenAI | None = None,
    sleep=time.sleep,
) -> RunResult:
    """Send one prompt to a model hosted by Groq and return a RunResult.

    Same inputs and behaviour as call_anthropic. Groq accepts OpenAI's request format, so
    the openai library is used, pointed at Groq's address with Groq's key.
    Source: https://console.groq.com/docs/openai
    """
    model = MODELS[model_key]
    if client is None:
        # max_retries=0: same reason as in call_anthropic.
        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ.get("GROQ_API_KEY"),
            max_retries=0,
        )

    messages = [{"role": "user", "content": prompt}]
    if system is not None:
        messages.insert(0, {"role": "system", "content": system})

    def send_once():
        response = client.chat.completions.create(
            model=model.model_id,
            messages=messages,
            max_tokens=max_tokens,
        )
        choice = response.choices[0]
        truncated = choice.finish_reason == "length"  # the answer hit the length limit
        usage = response.usage
        return choice.message.content, usage.prompt_tokens, usage.completion_tokens, truncated

    return call_with_retries(send_once, model, is_retryable, sleep)


# Which function talks to which provider. The names on the left must match the
# `provider` field in the registry (model_registry.py). A new provider adds one line here.
PROVIDER_FUNCTIONS = {
    "anthropic": call_anthropic,
    "openai": call_openai,
    "mistral": call_mistral,
    "groq": call_groq,
}


def run_model(
    model_key: str,
    prompt: str,
    system: str | None = None,
    max_tokens: int = MAX_ANSWER_TOKENS,
) -> RunResult:
    """Run one prompt on any model in the registry. The rest of the tool uses only this.

    It looks up which provider the model belongs to and calls the right function.
    Models that think before answering get extra room, because their thinking counts
    toward the length limit.
    """
    model = MODELS[model_key]
    provider = model.provider
    if model.reasoning:
        max_tokens += THINKING_EXTRA_TOKENS
    if provider not in PROVIDER_FUNCTIONS:
        # This is a mistake in our own setup (not an API failure), so fail loudly.
        raise ValueError(f"No function for provider '{provider}' (model '{model_key}')")
    return PROVIDER_FUNCTIONS[provider](model_key, prompt, system, max_tokens)
