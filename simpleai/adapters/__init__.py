"""Provider adapters for SimpleAI."""

from __future__ import annotations

from typing import Any

from .anthropic_adapter import AnthropicAdapter
from .base import BaseAdapter
from .gemini_adapter import GeminiAdapter
from .grok_adapter import GrokAdapter
from .openai_adapter import OpenAIAdapter
from .perplexity_adapter import PerplexityAdapter

ADAPTER_CLASSES = {
    "openai": OpenAIAdapter,
    "gemini": GeminiAdapter,
    "claude": AnthropicAdapter,
    "grok": GrokAdapter,
    "perplexity": PerplexityAdapter,
}


def get_adapter(provider: str, provider_settings: dict[str, Any]) -> BaseAdapter:
    """Return instantiated adapter for canonical provider key."""

    try:
        adapter_cls = ADAPTER_CLASSES[provider]
    except KeyError as exc:
        raise ValueError(f"Unsupported provider: {provider}") from exc
    return adapter_cls(provider_settings)


__all__ = [
    "AnthropicAdapter",
    "BaseAdapter",
    "GeminiAdapter",
    "get_adapter",
    "GrokAdapter",
    "OpenAIAdapter",
    "PerplexityAdapter",
]
