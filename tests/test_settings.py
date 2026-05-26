from __future__ import annotations

import json
from pathlib import Path

from simpleai.settings import (
    gemini_uses_enterprise_auth,
    get_default_reasoning_level,
    get_provider_api_key,
    load_settings,
    provider_has_credentials,
    provider_requires_api_key,
)


def test_load_settings_from_json_file(tmp_path: Path) -> None:
    settings_path = tmp_path / "ai_settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "defaults": ["openai", "gemini"],
                "providers": {
                    "chatgpt": {"default_model": "gpt-5-mini"},
                    "anthropic": {"default_model": "claude-sonnet-4-5-20250929"},
                    "xai": {"default_model": "grok-4-1-fast-reasoning"},
                },
                "logging": {"enabled": True},
            }
        ),
        encoding="utf-8",
    )

    settings = load_settings(settings_file=settings_path)

    assert settings["defaults"][:2] == ["openai", "gemini"]
    assert settings["providers"]["openai"]["default_model"] == "gpt-5-mini"
    assert settings["providers"]["claude"]["default_model"] == "claude-sonnet-4-5-20250929"
    assert settings["providers"]["grok"]["default_model"] == "grok-4-1-fast-reasoning"
    assert settings["logging"]["enabled"] is True


def test_load_settings_defaults_when_no_sources(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SIMPLEAI_SETTINGS_FILE", raising=False)

    settings = load_settings()

    assert settings["defaults"] == ["gemini", "openai", "claude", "grok", "perplexity"]
    assert settings["default_reasoning_level"] is None
    assert settings["providers"]["gemini"]["default_model"] == "gemini-3.5-flash"
    assert settings["providers"]["gemini"]["use_enterprise"] is None
    assert settings["providers"]["openai"]["default_model"] == "gpt-5.5"
    assert settings["providers"]["claude"]["default_model"] == "claude-opus-4-7"
    assert settings["providers"]["grok"]["default_model"] == "grok-4.3"
    assert "providers" in settings
    assert "citation_normalization" in settings
    assert settings["citation_normalization"]["source_alias_by_domain"] == {}


def test_django_settings_take_priority(monkeypatch) -> None:
    monkeypatch.setattr(
        "simpleai.settings._load_from_django",
        lambda: {
            "defaults": ["openai"],
            "providers": {"openai": {"default_model": "gpt-5-mini"}},
        },
    )
    monkeypatch.setattr("simpleai.settings._load_from_json", lambda explicit: None)

    settings = load_settings()

    assert settings["defaults"] == ["openai"]
    assert settings["providers"]["openai"]["default_model"] == "gpt-5-mini"


def test_load_settings_finds_app_root_json(tmp_path: Path, monkeypatch) -> None:
    app_root = tmp_path / "myapp"
    app_root.mkdir()
    (app_root / "pyproject.toml").write_text("[project]\nname='x'\nversion='0.0.1'\n", encoding="utf-8")
    (app_root / "ai_settings.json").write_text(
        json.dumps(
            {
                "providers": {
                    "openai": {"default_model": "gpt-5.2"},
                }
            }
        ),
        encoding="utf-8",
    )

    nested = app_root / "src" / "pkg"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    monkeypatch.delenv("SIMPLEAI_SETTINGS_FILE", raising=False)
    monkeypatch.delenv("SIMPLEAI_APP_ROOT", raising=False)

    settings = load_settings()

    assert settings["providers"]["openai"]["default_model"] == "gpt-5.2"


def test_get_provider_api_key_supports_grok_alias_env_var(monkeypatch) -> None:
    settings = {"providers": {"grok": {"api_key": None}}}
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setenv("GROK_API_KEY", "grok-test-key")

    assert get_provider_api_key(settings, "grok") == "grok-test-key"


def test_gemini_enterprise_auth_does_not_require_api_key(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_USE_ENTERPRISE", raising=False)
    settings = {"providers": {"gemini": {"api_key": None, "use_enterprise": True}}}

    assert gemini_uses_enterprise_auth(settings["providers"]["gemini"]) is True
    assert provider_requires_api_key(settings, "gemini") is False
    assert provider_has_credentials(settings, "gemini") is True


def test_gemini_enterprise_auth_can_be_enabled_from_env(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_USE_ENTERPRISE", "true")
    settings = {"providers": {"gemini": {"api_key": None, "use_enterprise": None}}}

    assert provider_requires_api_key(settings, "gemini") is False
    assert provider_has_credentials(settings, "gemini") is True


def test_gemini_enterprise_auth_supports_google_genai_env(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_USE_ENTERPRISE", raising=False)
    monkeypatch.delenv("GEMINI_USE_VERTEXAI", raising=False)
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    settings = {"providers": {"gemini": {"api_key": None, "use_enterprise": None}}}

    assert provider_requires_api_key(settings, "gemini") is False
    assert provider_has_credentials(settings, "gemini") is True


def test_default_reasoning_level_empty_uses_provider_defaults() -> None:
    settings = load_settings()
    assert get_default_reasoning_level(settings) is None

    settings["default_reasoning_level"] = ""
    assert get_default_reasoning_level(settings) is None

    settings["default_reasoning_level"] = "  "
    assert get_default_reasoning_level(settings) is None


def test_load_settings_default_reasoning_level(tmp_path: Path) -> None:
    settings_path = tmp_path / "ai_settings.json"
    settings_path.write_text(
        json.dumps({"default_reasoning_level": "high"}),
        encoding="utf-8",
    )

    settings = load_settings(settings_file=settings_path)

    assert settings["default_reasoning_level"] == "high"
    assert get_default_reasoning_level(settings) == "high"


def test_load_settings_merges_citation_domain_aliases(tmp_path: Path) -> None:
    settings_path = tmp_path / "ai_settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "citation_normalization": {
                    "source_alias_by_domain": {
                        "www.customnews.example": "Custom News Network"
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    settings = load_settings(settings_file=settings_path)
    assert settings["citation_normalization"]["source_alias_by_domain"] == {
        "www.customnews.example": "Custom News Network"
    }
