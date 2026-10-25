"""The model registry: every model the tool can call, with its facts in one place.

To add a model, add one entry to MODELS below. Copy the model ID and the prices from
the provider's official docs, and fill in docs_url and date_checked so that anyone can
see where the numbers came from and how fresh they are. Prices change often.

Two functions are at the bottom: select_models (which models a run uses) and
calculate_cost (the cost formula, next to the prices it uses).
"""

from dataclasses import dataclass


@dataclass(frozen=True)  # frozen: an entry cannot be changed by accident while running
class ModelInfo:
    """Facts about one model, copied from the provider's official docs."""

    provider: str
    model_id: str
    input_price_per_mtok: float  # USD per million input tokens
    output_price_per_mtok: float  # USD per million output tokens
    docs_url: str  # where the ID and prices were checked
    date_checked: str  # YYYY-MM-DD, prices change often
    # False: we switch the model's "thinking" off, so it answers directly.
    # True: the model thinks before answering, at the provider's default effort. This is
    # the only choice for models that cannot switch thinking off. Thinking is billed as
    # output tokens and counts toward the answer length limit.
    reasoning: bool
    # Anthropic only: the thinking type that turns thinking off. Sonnet 5.5 rejects
    # "disabled", see https://platform.claude.com/docs/en/api/errors
    thinking_off: str = "disabled"
    # "standard" models run by default. "premium" ones are expensive, so they run only
    # when asked for explicitly. "helper" models are never compared: they do work for the
    # tool itself (such as turning a description into a task), and come from a family
    # outside the study so they cannot favour one of the candidates.
    tier: str = "standard"
    # How the model is named in the app. Empty means the registry key is shown.
    display_name: str = ""


# The key (for example "haiku-5-5") is the name we use everywhere else in the tool.
MODELS = {
    "haiku-5-5": ModelInfo(
        display_name="Claude Haiku 5.5",
        provider="anthropic",
        model_id="claude-haiku-5-5",
        input_price_per_mtok=0.10,  # prompts up to 100k tokens
        output_price_per_mtok=0.50,  # prompts up to 100k tokens
        docs_url="https://platform.claude.com/docs/en/about-claude/pricing",
        date_checked="2026-10-12",
        reasoning=False,
    ),
    "sonnet-5-5": ModelInfo(
        display_name="Claude Sonnet 5.5",
        provider="anthropic",
        model_id="claude-sonnet-5-5",
        input_price_per_mtok=2.00,
        output_price_per_mtok=10.00,
        docs_url="https://platform.claude.com/docs/en/about-claude/pricing",
        date_checked="2026-10-12",
        reasoning=False,
        thinking_off="between_tools",
    ),
    "gpt-6-luna": ModelInfo(
        display_name="GPT-6 Luna",
        provider="openai",
        model_id="gpt-6-luna",
        input_price_per_mtok=0.10,  # prompts up to 272K tokens
        output_price_per_mtok=0.50,
        docs_url="https://developers.openai.com/api/docs/pricing",
        date_checked="2026-10-12",
        reasoning=False,
    ),
    "mistral-small-4": ModelInfo(
        display_name="Mistral Small 4",
        provider="mistral",
        # pinned ID on purpose: mistral-small-latest is an alias we deliberately avoid
        model_id="mistral-small-2603",  # confirmed with a real call on 2026-10-12
        input_price_per_mtok=0.15,  # order not labelled on the official page: check on the dashboard
        output_price_per_mtok=0.60,
        docs_url="https://docs.mistral.ai/models/mistral-small-4-0-26-03",
        date_checked="2026-10-12",
        reasoning=False,
    ),
    # --- Premium models: expensive, so they run only when asked for (tier="premium"). ---
    # None of them can switch thinking off, so reasoning=True: they think at the
    # provider's default effort, and that thinking is billed as output tokens.
    "opus-5-5": ModelInfo(
        display_name="Claude Opus 5.5",
        provider="anthropic",
        model_id="claude-opus-5-5",
        input_price_per_mtok=4.00,
        output_price_per_mtok=20.00,
        docs_url="https://platform.claude.com/docs/en/about-claude/pricing",
        date_checked="2026-10-13",
        reasoning=True,  # thinking cannot be disabled on this model
        tier="premium",
    ),
    "fable-5-1": ModelInfo(
        display_name="Claude Fable 5.1",
        provider="anthropic",
        model_id="claude-fable-5-1",
        input_price_per_mtok=10.00,
        output_price_per_mtok=50.00,
        docs_url="https://platform.claude.com/docs/en/about-claude/pricing",
        date_checked="2026-10-13",
        reasoning=True,  # thinking is always on for this model
        tier="premium",
    ),
    "gpt-6.1-sol": ModelInfo(
        display_name="GPT-6.1 Sol",
        provider="openai",
        model_id="gpt-6.1-sol",
        input_price_per_mtok=2.00,  # prompts up to 272K tokens
        output_price_per_mtok=10.00,
        docs_url="https://developers.openai.com/api/docs/models/gpt-6.1-sol",
        date_checked="2026-10-13",
        reasoning=True,  # reasoning effort "none" is not supported, the default is "medium"
        tier="premium",
    ),
    "gpt-6-astra": ModelInfo(
        display_name="GPT-6 Astra",
        provider="openai",
        model_id="gpt-6-astra",
        input_price_per_mtok=10.00,  # prompts up to 272K tokens
        output_price_per_mtok=50.00,
        docs_url="https://developers.openai.com/api/docs/models/gpt-6-astra",
        date_checked="2026-10-13",
        reasoning=True,  # reasoning effort "none" is not listed as supported
        tier="premium",
    ),
    # --- Helper models: never compared, they do work for the tool itself (tier="helper"). ---
    "qwen-3-8-27b": ModelInfo(
        display_name="Qwen 3.8 (helper)",
        provider="groq",
        model_id="qwen/qwen3.8-27b",  # Alibaba's open-weight model, hosted by Groq
        # Used on Groq's free tier, which costs nothing within its rate limits. Groq lists
        # this model as a preview (it may be withdrawn at short notice) and publishes no
        # price for it. Llama 3.3 70B was the first choice, but this account cannot use it.
        input_price_per_mtok=0.0,
        output_price_per_mtok=0.0,
        docs_url="https://console.groq.com/docs/models",
        date_checked="2026-10-14",
        reasoning=False,
        tier="helper",
    ),
}


def select_models(include_premium: bool = False, models: dict = MODELS) -> list[str]:
    """Return the keys of the models to compare: standard ones, plus premium ones if asked.

    Helper models are never returned: they are not part of the comparison.

    models: the registry to choose from. Tests pass a small fake one.
    """
    return [
        key
        for key, model in models.items()
        if model.tier == "standard" or (include_premium and model.tier == "premium")
    ]


def display_name(model_key: str, models: dict = MODELS) -> str:
    """Return the model's name for the app, or its key if the model is unknown."""
    model = models.get(model_key)
    return (model.display_name or model_key) if model else model_key


def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    input_price_per_m: float,
    output_price_per_m: float,
) -> float:
    """Cost in USD of one call, given prices per million tokens.

Example: 2,000 input tokens at $0.10 per million, plus 500 output tokens at
$0.50 per million, costs 0.0002 + 0.00025 = $0.00045.
"""
    input_cost = input_tokens / 1_000_000 * input_price_per_m
    output_cost = output_tokens / 1_000_000 * output_price_per_m
    return input_cost + output_cost
