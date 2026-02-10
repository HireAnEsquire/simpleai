"""Model/provider resolution for SimpleAI."""

from __future__ import annotations

from typing import Any

from .exceptions import ModelResolutionError
from .settings import canonical_provider_name, get_provider_api_key

# Known model IDs from official provider model docs as of 2026-02-06.
MODEL_PROVIDER_MAP: dict[str, str] = {
    # OpenAI
    "gpt-5.2": "openai",
    "gpt-5.2-mini": "openai",
    "gpt-5.2-nano": "openai",
    "gpt-5.2-pro": "openai",
    "gpt-5.2-chat-latest": "openai",
    "gpt-5": "openai",
    "gpt-5-chat-latest": "openai",
    "gpt-5-mini": "openai",
    "gpt-5-nano": "openai",
    "gpt-4.1-nano": "openai",
    "gpt-4.1": "openai",
    "gpt-4.1-mini": "openai",
    "gpt-4o": "openai",
    "gpt-4o-mini": "openai",
    "o4-mini": "openai",
    "o4-mini-deep-research": "openai",
    "o3": "openai",
    "o3-pro": "openai",
    "o3-mini": "openai",
    "o1": "openai",
    "gpt-image-1": "openai",
    "gpt-image-1-mini": "openai",
    "computer-use-preview": "openai",
    "codex-mini-latest": "openai",
    # Gemini
    "gemini-3-pro": "gemini",
    "gemini-3-pro-preview": "gemini",
    "gemini-3-flash-preview": "gemini",
    "gemini-3-flash-lite-preview": "gemini",
    "gemini-2.5-pro": "gemini",
    "gemini-2.5-pro-preview-tts": "gemini",
    "gemini-2.5-flash": "gemini",
    "gemini-2.5-flash-preview-native-audio-dialog": "gemini",
    "gemini-2.5-flash-lite": "gemini",
    "gemini-2.0-flash": "gemini",
    "gemini-2.0-flash-preview-image-generation": "gemini",
    "gemini-2.0-flash-lite": "gemini",
    "gemini-embedding-001": "gemini",
    "text-embedding-005": "gemini",
    "veo-3.1-generate-preview": "gemini",
    "veo-3.0-generate-preview": "gemini",
    # Anthropic Claude
    "claude-opus-4-6": "claude",
    "claude-opus-4-6-20260115": "claude",
    "claude-sonnet-4-5": "claude",
    "claude-opus-4-1-20250805": "claude",
    "claude-opus-4-20250514": "claude",
    "claude-haiku-4-5": "claude",
    "claude-haiku-4-5-20251001": "claude",
    "claude-sonnet-4-5-20250929": "claude",
    "claude-sonnet-4-20250514": "claude",
    "claude-haiku-3-5-20241022": "claude",
    "claude-3-7-sonnet-20250219": "claude",
    # xAI Grok
    "grok-4-1-fast-reasoning": "grok",
    "grok-4-0709": "grok",
    "grok-4": "grok",
    "grok-4-fast": "grok",
    "grok-4-fast-reasoning": "grok",
    "grok-4-fast-reasoning-latest": "grok",
    "grok-4-fast-non-reasoning": "grok",
    "grok-4-fast-non-reasoning-latest": "grok",
    "grok-4-1-fast": "grok",
    "grok-4-1-fast-reasoning": "grok",
    "grok-4-1-fast-reasoning-latest": "grok",
    "grok-4-1-fast-non-reasoning": "grok",
    "grok-4-1-fast-non-reasoning-latest": "grok",
    "grok-3": "grok",
    "grok-3-latest": "grok",
    "grok-3-fast": "grok",
    "grok-3-fast-latest": "grok",
    "grok-3-mini": "grok",
    "grok-3-mini-fast": "grok",
    "grok-3-mini-fast-latest": "grok",
    "grok-code-fast-1": "grok",
    # Perplexity
    "fast-search": "perplexity",
    "pro-search": "perplexity",
    "deep-research": "perplexity",
    "sonar": "perplexity",
    "sonar-pro": "perplexity",
    "sonar-reasoning": "perplexity",
    "sonar-reasoning-pro": "perplexity",
    "sonar-deep-research": "perplexity",
    "r1-1776": "perplexity",
    "openai/o4-mini": "perplexity",
    "openai/gpt-4.1": "perplexity",
    "xai/grok-4-1": "perplexity",
}

_PROVIDER_HINTS: dict[str, str] = {
    "openai": "openai",
    "gpt": "openai",
    "o3": "openai",
    "o4": "openai",
    "gemini": "gemini",
    "claude": "claude",
    "anthropic": "claude",
    "grok": "grok",
    "xai": "grok",
    "perplexity": "perplexity",
    "sonar": "perplexity",
}


def _default_model(settings: dict[str, Any], provider: str) -> str:
    provider_config = settings.get("providers", {}).get(provider, {})
    model = provider_config.get("default_model") if isinstance(provider_config, dict) else None
    if model:
        return str(model)
    raise ModelResolutionError(f"No default model configured for provider '{provider}'.")



def _provider_has_credentials(settings: dict[str, Any], provider: str) -> bool:
    return bool(get_provider_api_key(settings, provider))



def select_default_provider(settings: dict[str, Any]) -> str:
    """Select first configured default provider with credentials."""

    defaults = settings.get("defaults", [])
    if not isinstance(defaults, list):
        defaults = []

    canonical_defaults: list[str] = []
    for item in defaults:
        if not isinstance(item, str):
            continue
        canonical = canonical_provider_name(item) or item.strip().lower()
        if canonical not in canonical_defaults:
            canonical_defaults.append(canonical)

    for provider in canonical_defaults:
        if _provider_has_credentials(settings, provider):
            return provider

    if canonical_defaults:
        raise ModelResolutionError(
            f"No credentials found for any configured default provider: {canonical_defaults}. "
             "Set an API key in settings or environment."
            )

    raise ModelResolutionError("No default providers configured.")



def resolve_provider_and_model(
    settings: dict[str, Any],
    requested_model: str | None,
) -> tuple[str, str]:
    """Resolve canonical provider + model from user input and settings."""

    if requested_model is None:
        provider = select_default_provider(settings)
        return provider, _default_model(settings, provider)

    requested = requested_model.strip()
    requested_lower = requested.lower()

    provider_alias = canonical_provider_name(requested_lower)
    if provider_alias:
        return provider_alias, _default_model(settings, provider_alias)

    mapped_provider = MODEL_PROVIDER_MAP.get(requested_lower)
    if mapped_provider:
        return mapped_provider, requested

    for token, provider in _PROVIDER_HINTS.items():
        if token in requested_lower:
            return provider, requested

    raise ModelResolutionError(
        "Unable to resolve provider for model "
        f"'{requested_model}'. Provide a known provider alias or known model name."
    )
