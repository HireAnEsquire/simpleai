from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from simpleai.api import run_prompt
from simpleai.exceptions import SettingsError, SimpleAIException
from simpleai.types import AdapterResponse, Citation


class PayloadModel(BaseModel):
    value: int


class DummyAdapter:
    def __init__(self, supports_binary_files: bool = False) -> None:
        self.supports_binary_files = supports_binary_files
        self.last_kwargs: dict[str, Any] = {}

    def run(self, **kwargs: Any) -> AdapterResponse:
        self.last_kwargs = kwargs
        text = '{"value": 7}'
        citations = [Citation(provider="openai", url="https://example.com", title="Example")]
        return AdapterResponse(text=text, citations=citations, raw={"ok": True})


BASE_SETTINGS = {
    "defaults": ["openai"],
    "providers": {
        "openai": {
            "default_model": "gpt-5",
            "api_key": "sk-test",
        }
    },
    "logging": {"enabled": False},
}


def test_run_prompt_returns_model_instance(monkeypatch) -> None:
    adapter = DummyAdapter()

    monkeypatch.setattr("simpleai.api.load_settings", lambda settings_file=None: BASE_SETTINGS)
    monkeypatch.setattr("simpleai.api.resolve_provider_and_model", lambda settings, model: ("openai", "gpt-5"))
    monkeypatch.setattr("simpleai.api.get_adapter", lambda provider, provider_settings: adapter)

    result = run_prompt("hello", model="openai", output_format=PayloadModel)
    assert isinstance(result, PayloadModel)
    assert result.value == 7


def test_run_prompt_returns_tuple_when_citations_enabled(monkeypatch) -> None:
    adapter = DummyAdapter()

    monkeypatch.setattr("simpleai.api.load_settings", lambda settings_file=None: BASE_SETTINGS)
    monkeypatch.setattr("simpleai.api.resolve_provider_and_model", lambda settings, model: ("openai", "gpt-5"))
    monkeypatch.setattr("simpleai.api.get_adapter", lambda provider, provider_settings: adapter)

    result, citations = run_prompt("hello", model="openai", return_citations=True)
    assert isinstance(result, str)
    assert isinstance(citations, list)
    assert citations[0]["url"] == "https://example.com"
    assert adapter.last_kwargs["require_search"] is True
    assert adapter.last_kwargs["return_citations"] is True


def test_run_prompt_infers_return_citations_from_require_search(monkeypatch) -> None:
    adapter = DummyAdapter()

    monkeypatch.setattr("simpleai.api.load_settings", lambda settings_file=None: BASE_SETTINGS)
    monkeypatch.setattr("simpleai.api.resolve_provider_and_model", lambda settings, model: ("openai", "gpt-5"))
    monkeypatch.setattr("simpleai.api.get_adapter", lambda provider, provider_settings: adapter)

    result, citations = run_prompt("hello", model="openai", require_search=True)
    assert isinstance(result, str)
    assert len(citations) == 1
    assert adapter.last_kwargs["require_search"] is True
    assert adapter.last_kwargs["return_citations"] is True


def test_return_citations_true_forces_require_search_even_if_false(monkeypatch) -> None:
    adapter = DummyAdapter()

    monkeypatch.setattr("simpleai.api.load_settings", lambda settings_file=None: BASE_SETTINGS)
    monkeypatch.setattr("simpleai.api.resolve_provider_and_model", lambda settings, model: ("openai", "gpt-5"))
    monkeypatch.setattr("simpleai.api.get_adapter", lambda provider, provider_settings: adapter)

    run_prompt("hello", model="openai", require_search=False, return_citations=True)

    assert adapter.last_kwargs["require_search"] is True
    assert adapter.last_kwargs["return_citations"] is True


def test_run_prompt_accepts_string_bool_for_return_citations(monkeypatch) -> None:
    adapter = DummyAdapter()

    monkeypatch.setattr("simpleai.api.load_settings", lambda settings_file=None: BASE_SETTINGS)
    monkeypatch.setattr("simpleai.api.resolve_provider_and_model", lambda settings, model: ("openai", "gpt-5"))
    monkeypatch.setattr("simpleai.api.get_adapter", lambda provider, provider_settings: adapter)

    result, citations = run_prompt("hello", model="openai", return_citations="True")
    assert isinstance(result, str)
    assert len(citations) == 1
    assert adapter.last_kwargs["require_search"] is True
    assert adapter.last_kwargs["return_citations"] is True


def test_run_prompt_extracts_files_when_binary_not_supported(monkeypatch, tmp_path: Path) -> None:
    adapter = DummyAdapter(supports_binary_files=False)
    note = tmp_path / "note.txt"
    note.write_text("attached content", encoding="utf-8")

    monkeypatch.setattr("simpleai.api.load_settings", lambda settings_file=None: BASE_SETTINGS)
    monkeypatch.setattr("simpleai.api.resolve_provider_and_model", lambda settings, model: ("openai", "gpt-5"))
    monkeypatch.setattr("simpleai.api.get_adapter", lambda provider, provider_settings: adapter)

    run_prompt("base prompt", model="openai", file=note, binary_files=True)

    prompt_payload = adapter.last_kwargs["prompt"]
    assert isinstance(prompt_payload, str)
    assert "attached content" in prompt_payload
    assert adapter.last_kwargs["files"] is None


def test_run_prompt_passes_binary_files_when_supported(monkeypatch, tmp_path: Path) -> None:
    adapter = DummyAdapter(supports_binary_files=True)
    note = tmp_path / "note.txt"
    note.write_text("attached content", encoding="utf-8")

    monkeypatch.setattr("simpleai.api.load_settings", lambda settings_file=None: BASE_SETTINGS)
    monkeypatch.setattr("simpleai.api.resolve_provider_and_model", lambda settings, model: ("openai", "gpt-5"))
    monkeypatch.setattr("simpleai.api.get_adapter", lambda provider, provider_settings: adapter)

    run_prompt("base prompt", model="openai", file=note, binary_files=True)

    files = adapter.last_kwargs["files"]
    assert files is not None
    assert files[0] == note.resolve()


def test_run_prompt_merges_provider_kwargs(monkeypatch) -> None:
    adapter = DummyAdapter()

    monkeypatch.setattr("simpleai.api.load_settings", lambda settings_file=None: BASE_SETTINGS)
    monkeypatch.setattr("simpleai.api.resolve_provider_and_model", lambda settings, model: ("openai", "gpt-5"))
    monkeypatch.setattr("simpleai.api.get_adapter", lambda provider, provider_settings: adapter)

    run_prompt("hello", model="openai", temperature=0.2, adapter_options={"top_p": 0.8})

    options = adapter.last_kwargs["adapter_options"]
    assert options["top_p"] == 0.8
    assert options["temperature"] == 0.2


def test_run_prompt_missing_provider_key_raises_settings_error(monkeypatch) -> None:
    settings = {
        "defaults": ["grok"],
        "providers": {
            "grok": {
                "default_model": "grok-4-1-fast-reasoning",
                "api_key": None,
            }
        },
        "logging": {"enabled": False},
    }

    monkeypatch.setattr("simpleai.api.load_settings", lambda settings_file=None: settings)
    monkeypatch.setattr("simpleai.api.resolve_provider_and_model", lambda settings, model: ("grok", "grok-4-1-fast-reasoning"))
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("GROK_API_KEY", raising=False)

    with pytest.raises(SettingsError) as exc:
        run_prompt("hello", model="grok")

    message = str(exc.value)
    assert "Missing API key for provider 'grok'" in message
    assert "XAI_API_KEY" in message
    assert "GROK_API_KEY" in message


def test_run_prompt_missing_provider_key_is_catchable_as_simpleai_exception(monkeypatch) -> None:
    settings = {
        "defaults": ["grok"],
        "providers": {
            "grok": {
                "default_model": "grok-4-1-fast-reasoning",
                "api_key": None,
            }
        },
        "logging": {"enabled": False},
    }

    monkeypatch.setattr("simpleai.api.load_settings", lambda settings_file=None: settings)
    monkeypatch.setattr("simpleai.api.resolve_provider_and_model", lambda settings, model: ("grok", "grok-4-1-fast-reasoning"))
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("GROK_API_KEY", raising=False)

    with pytest.raises(SimpleAIException):
        run_prompt("hello", model="grok")


def test_run_prompt_wraps_unexpected_errors_in_simpleai_exception(monkeypatch) -> None:
    adapter = DummyAdapter()

    monkeypatch.setattr("simpleai.api.load_settings", lambda settings_file=None: BASE_SETTINGS)
    monkeypatch.setattr("simpleai.api.resolve_provider_and_model", lambda settings, model: ("openai", "gpt-5"))
    monkeypatch.setattr("simpleai.api.get_adapter", lambda provider, provider_settings: adapter)
    monkeypatch.setattr("simpleai.api.coerce_output", lambda text, output_format: (_ for _ in ()).throw(ValueError("bad parse")))

    with pytest.raises(SimpleAIException) as exc:
        run_prompt("hello", model="openai", output_format=PayloadModel)

    assert "run_prompt failed" in str(exc.value)
    assert isinstance(exc.value.original_exception, ValueError)
    assert str(exc.value.original_exception) == "bad parse"


def test_run_prompt_normalizes_citations_and_preserves_originals(monkeypatch) -> None:
    class WeirdCitationAdapter(DummyAdapter):
        def run(self, **kwargs: Any) -> AdapterResponse:
            self.last_kwargs = kwargs
            return AdapterResponse(
                text="ok",
                citations=[
                    Citation(
                        provider="openai",
                        url="https://news.example/story?utm_source=abc&id=1",
                        title="https://news.example/story?utm_source=abc&id=1",
                        source="news.example",
                    )
                ],
                raw={"ok": True},
            )

    adapter = WeirdCitationAdapter()
    monkeypatch.setattr("simpleai.api.load_settings", lambda settings_file=None: BASE_SETTINGS)
    monkeypatch.setattr("simpleai.api.resolve_provider_and_model", lambda settings, model: ("openai", "gpt-5"))
    monkeypatch.setattr("simpleai.api.get_adapter", lambda provider, provider_settings: adapter)

    _, citations = run_prompt("hello", model="openai", return_citations=True, validate_urls=False)
    c = citations[0]

    assert c["original_url"] == "https://news.example/story?utm_source=abc&id=1"
    assert c["url"] == "https://news.example/story?id=1"
    assert c["original_title"] == "https://news.example/story?utm_source=abc&id=1"
    assert c["title"] == "Story"
    assert c["original_source"] == "news.example"
    assert c["source"] == "News Example"
    assert c["source_key"] == "news.example"
    assert c["analytics_source"] == "News Example"
    assert c["analytics_title"] == "Story"
    assert c["analytics_url"] == "https://news.example/story?id=1"
    assert c["normalization_version"] == "v1"


def test_run_prompt_enriches_citations_when_flag_enabled(monkeypatch) -> None:
    class MissingMetadataAdapter(DummyAdapter):
        def run(self, **kwargs: Any) -> AdapterResponse:
            self.last_kwargs = kwargs
            return AdapterResponse(
                text="ok",
                citations=[Citation(provider="gemini", url="https://example.com/path", title=None, source=None)],
                raw={"ok": True},
            )

    adapter = MissingMetadataAdapter()
    monkeypatch.setattr("simpleai.api.load_settings", lambda settings_file=None: BASE_SETTINGS)
    monkeypatch.setattr("simpleai.api.resolve_provider_and_model", lambda settings, model: ("openai", "gpt-5"))
    monkeypatch.setattr("simpleai.api.get_adapter", lambda provider, provider_settings: adapter)
    monkeypatch.setattr(
        "simpleai.citations._fetch_url_metadata",
        lambda url, timeout: ("Example Article", "Example Publication"),
    )

    _, citations = run_prompt(
        "hello",
        model="openai",
        return_citations=True,
        validate_urls=False,
        enrich_citations=True,
    )
    c = citations[0]
    assert c["title"] == "Example Article"
    assert c["source"] == "Example Publication"
    assert c["source_confidence"] == "html_meta"


def test_run_prompt_applies_domain_aliases_from_settings(monkeypatch) -> None:
    class AliasCitationAdapter(DummyAdapter):
        def run(self, **kwargs: Any) -> AdapterResponse:
            self.last_kwargs = kwargs
            return AdapterResponse(
                text="ok",
                citations=[
                    Citation(
                        provider="openai",
                        url="https://www.customnews.example/world/story",
                        title="World Story",
                        source="www.customnews.example",
                    )
                ],
                raw={"ok": True},
            )

    adapter = AliasCitationAdapter()
    settings_with_aliases = {
        **BASE_SETTINGS,
        "citation_normalization": {
            "source_alias_by_domain": {
                "customnews.example": "Custom News Network",
            }
        },
    }
    monkeypatch.setattr("simpleai.api.load_settings", lambda settings_file=None: settings_with_aliases)
    monkeypatch.setattr("simpleai.api.resolve_provider_and_model", lambda settings, model: ("openai", "gpt-5"))
    monkeypatch.setattr("simpleai.api.get_adapter", lambda provider, provider_settings: adapter)

    _, citations = run_prompt("hello", model="openai", return_citations=True, validate_urls=False)
    c = citations[0]

    assert c["source"] == "Custom News Network"
    assert c["analytics_source"] == "Custom News Network"
    assert c["source_confidence"] == "mapped_domain"


def test_run_prompt_applies_domain_aliases_from_parameter(monkeypatch) -> None:
    class AliasCitationAdapter(DummyAdapter):
        def run(self, **kwargs: Any) -> AdapterResponse:
            self.last_kwargs = kwargs
            return AdapterResponse(
                text="ok",
                citations=[
                    Citation(
                        provider="openai",
                        url="https://jobs.example.org/hiring/page",
                        title="Hiring Page",
                        source="jobs.example.org",
                    )
                ],
                raw={"ok": True},
            )

    adapter = AliasCitationAdapter()
    monkeypatch.setattr("simpleai.api.load_settings", lambda settings_file=None: BASE_SETTINGS)
    monkeypatch.setattr("simpleai.api.resolve_provider_and_model", lambda settings, model: ("openai", "gpt-5"))
    monkeypatch.setattr("simpleai.api.get_adapter", lambda provider, provider_settings: adapter)

    _, citations = run_prompt(
        "hello",
        model="openai",
        return_citations=True,
        validate_urls=False,
        source_alias_by_domain={"jobs.example.org": "Example Jobs Board"},
    )
    c = citations[0]

    assert c["source"] == "Example Jobs Board"
    assert c["analytics_source"] == "Example Jobs Board"
    assert c["source_confidence"] == "mapped_domain"


def test_run_prompt_param_aliases_override_settings_aliases(monkeypatch) -> None:
    class AliasCitationAdapter(DummyAdapter):
        def run(self, **kwargs: Any) -> AdapterResponse:
            self.last_kwargs = kwargs
            return AdapterResponse(
                text="ok",
                citations=[
                    Citation(
                        provider="openai",
                        url="https://www.customnews.example/world/story",
                        title="World Story",
                        source="www.customnews.example",
                    )
                ],
                raw={"ok": True},
            )

    adapter = AliasCitationAdapter()
    settings_with_aliases = {
        **BASE_SETTINGS,
        "citation_normalization": {
            "source_alias_by_domain": {
                "customnews.example": "Settings News Name",
            }
        },
    }
    monkeypatch.setattr("simpleai.api.load_settings", lambda settings_file=None: settings_with_aliases)
    monkeypatch.setattr("simpleai.api.resolve_provider_and_model", lambda settings, model: ("openai", "gpt-5"))
    monkeypatch.setattr("simpleai.api.get_adapter", lambda provider, provider_settings: adapter)

    _, citations = run_prompt(
        "hello",
        model="openai",
        return_citations=True,
        validate_urls=False,
        source_alias_by_domain={"customnews.example": "Param News Name"},
    )
    c = citations[0]

    assert c["source"] == "Param News Name"
