from __future__ import annotations

from simpleai.model_registry import resolve_provider_and_model


BASE_SETTINGS = {
    "defaults": ["gemini", "openai", "claude", "grok", "perplexity"],
    "providers": {
        "gemini": {"default_model": "gemini-3-pro-preview", "api_key": None},
        "openai": {"default_model": "gpt-5.2", "api_key": "sk-test"},
        "claude": {"default_model": "claude-opus-4-6", "api_key": None},
        "grok": {"default_model": "grok-4-latest", "api_key": None},
        "perplexity": {"default_model": "sonar-deep-research", "api_key": None},
    },
}


def test_resolve_provider_alias_uses_default_model() -> None:
    provider, model = resolve_provider_and_model(BASE_SETTINGS, "chatgpt")
    assert provider == "openai"
    assert model == "gpt-5.2"


def test_resolve_known_model_mapping() -> None:
    provider, model = resolve_provider_and_model(BASE_SETTINGS, "claude-sonnet-4-5-20250929")
    assert provider == "claude"
    assert model == "claude-sonnet-4-5-20250929"


def test_resolve_unknown_model_heuristic() -> None:
    provider, model = resolve_provider_and_model(BASE_SETTINGS, "custom-grok-experimental")
    assert provider == "grok"
    assert model == "custom-grok-experimental"


def test_resolve_latest_gemini_family() -> None:
    provider, model = resolve_provider_and_model(BASE_SETTINGS, "gemini-3-pro-preview")
    assert provider == "gemini"
    assert model == "gemini-3-pro-preview"


def test_default_provider_prefers_credentials() -> None:
    provider, model = resolve_provider_and_model(BASE_SETTINGS, None)
    assert provider == "openai"
    assert model == "gpt-5.2"
