"""RunResult: the one shape every model call returns, whatever the provider."""

from dataclasses import dataclass


@dataclass
class RunResult:
    """What one model call returns, the same shape for every provider.

If the call failed, `error` holds a short message and the other fields are empty
or zero. Always check `error` before trusting `text`.
"""

    text: str  # the model's answer, empty if the call failed
    input_tokens: int
    output_tokens: int
    latency_s: float  # seconds the (last) attempt took, not counting waiting between tries
    cost_usd: float  # computed from the token counts and the registry prices
    error: str | None = None  # None means the call succeeded
    truncated: bool = False  # True if the answer was cut off at the length limit
