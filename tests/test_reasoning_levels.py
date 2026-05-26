"""Tests for reasoning_level mapping across SimpleAI adapters."""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from simpleai.adapters.reasoning import (
    REASONING_LEVELS,
    build_anthropic_reasoning_payload,
    build_gemini_reasoning_config_kwargs,
    build_grok_reasoning_payload,
    build_openai_reasoning_payload,
    build_perplexity_reasoning_payload,
    parse_reasoning_level,
    resolve_reasoning_level,
)
from simpleai.adapters.anthropic_adapter import AnthropicAdapter
from simpleai.adapters.gemini_adapter import GeminiAdapter
from simpleai.adapters.grok_adapter import GrokAdapter
from simpleai.adapters.openai_adapter import OpenAIAdapter
from simpleai.adapters.perplexity_adapter import PerplexityAdapter
from simpleai.types import AdapterResponse


class ReasoningHelpersTest(unittest.TestCase):
    def test_parse_reasoning_level_accepts_aliases(self) -> None:
        self.assertEqual(parse_reasoning_level("xhigh"), "extra_high")
        self.assertEqual(parse_reasoning_level("extra-high"), "extra_high")
        self.assertIsNone(parse_reasoning_level(None))

    def test_parse_reasoning_level_rejects_invalid(self) -> None:
        with self.assertRaises(ValueError):
            parse_reasoning_level("turbo")

    def test_resolve_reasoning_level_fallbacks(self) -> None:
        self.assertEqual(
            resolve_reasoning_level("none", supports_none=False, supports_extra_high=True),
            "low",
        )
        self.assertEqual(
            resolve_reasoning_level("extra_high", supports_none=True, supports_extra_high=False),
            "high",
        )
        self.assertIsNone(resolve_reasoning_level(None, supports_none=True, supports_extra_high=True))

    def test_all_levels_are_documented(self) -> None:
        self.assertEqual(
            REASONING_LEVELS,
            ("none", "low", "medium", "high", "extra_high"),
        )


class OpenAIReasoningPayloadTest(unittest.TestCase):
    def test_none_supported_on_gpt_5_4(self) -> None:
        payload = build_openai_reasoning_payload("none", model="gpt-5.4")
        self.assertEqual(payload, {"reasoning": {"effort": "none"}})

    def test_none_falls_back_to_low_on_o3(self) -> None:
        payload = build_openai_reasoning_payload("none", model="o3-mini")
        self.assertEqual(payload, {"reasoning": {"effort": "low"}})

    def test_extra_high_maps_to_xhigh_on_codex_max(self) -> None:
        payload = build_openai_reasoning_payload("extra_high", model="gpt-5.3-codex-max")
        self.assertEqual(payload, {"reasoning": {"effort": "xhigh"}})

    def test_extra_high_falls_back_to_high_on_o3(self) -> None:
        payload = build_openai_reasoning_payload("extra_high", model="o3")
        self.assertEqual(payload, {"reasoning": {"effort": "high"}})

    def test_omitted_uses_model_defaults(self) -> None:
        self.assertEqual(build_openai_reasoning_payload(None, model="gpt-5.4"), {})
        self.assertEqual(build_openai_reasoning_payload("medium", model="gpt-4o"), {})


class AnthropicReasoningPayloadTest(unittest.TestCase):
    def test_none_omits_thinking_config(self) -> None:
        self.assertEqual(build_anthropic_reasoning_payload("none", model="claude-opus-4-7"), {})

    def test_adaptive_effort_on_opus_4_7(self) -> None:
        payload = build_anthropic_reasoning_payload("high", model="claude-opus-4-7")
        self.assertEqual(
            payload,
            {
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": "high"},
            },
        )

    def test_extra_high_maps_to_xhigh_on_opus_4_7(self) -> None:
        payload = build_anthropic_reasoning_payload("extra_high", model="claude-opus-4-7")
        self.assertEqual(payload["output_config"], {"effort": "xhigh"})

    def test_extra_high_falls_back_on_sonnet_4_6(self) -> None:
        payload = build_anthropic_reasoning_payload("extra_high", model="claude-sonnet-4-6")
        self.assertEqual(payload["output_config"], {"effort": "high"})

    def test_legacy_budget_on_older_sonnet(self) -> None:
        payload = build_anthropic_reasoning_payload("medium", model="claude-sonnet-4-5")
        self.assertEqual(payload["thinking"]["type"], "enabled")
        self.assertEqual(payload["thinking"]["budget_tokens"], 8000)


class GeminiReasoningPayloadTest(unittest.TestCase):
    def test_gemini_3_uses_thinking_level(self) -> None:
        payload = build_gemini_reasoning_config_kwargs("medium", model="gemini-3.1-pro-preview")
        self.assertEqual(payload, {"thinking_config": {"thinking_level": "MEDIUM"}})

    def test_gemini_3_none_defaults_to_low(self) -> None:
        payload = build_gemini_reasoning_config_kwargs("none", model="gemini-3.1-pro-preview")
        self.assertEqual(payload, {"thinking_config": {"thinking_level": "LOW"}})

    def test_gemini_3_extra_high_defaults_to_high(self) -> None:
        payload = build_gemini_reasoning_config_kwargs("extra_high", model="gemini-3-flash-preview")
        self.assertEqual(payload, {"thinking_config": {"thinking_level": "HIGH"}})

    def test_gemini_25_uses_thinking_budget(self) -> None:
        payload = build_gemini_reasoning_config_kwargs("low", model="gemini-2.5-flash")
        self.assertEqual(payload, {"thinking_budget": 1024})

    def test_gemini_25_none_uses_zero_budget(self) -> None:
        payload = build_gemini_reasoning_config_kwargs("none", model="gemini-2.5-flash")
        self.assertEqual(payload, {"thinking_budget": 0})


class GrokReasoningPayloadTest(unittest.TestCase):
    def test_none_on_standard_grok(self) -> None:
        payload = build_grok_reasoning_payload("none", model="grok-4.3")
        self.assertEqual(payload, {"reasoning": {"effort": "none"}})

    def test_none_falls_back_on_multi_agent(self) -> None:
        payload = build_grok_reasoning_payload("none", model="grok-4.20-multi-agent")
        self.assertEqual(payload, {"reasoning": {"effort": "low"}})

    def test_extra_high_on_multi_agent(self) -> None:
        payload = build_grok_reasoning_payload("extra_high", model="grok-4.20-multi-agent")
        self.assertEqual(payload, {"reasoning": {"effort": "xhigh"}})

    def test_extra_high_falls_back_on_standard_grok(self) -> None:
        payload = build_grok_reasoning_payload("extra_high", model="grok-4.3")
        self.assertEqual(payload, {"reasoning": {"effort": "high"}})


class PerplexityReasoningPayloadTest(unittest.TestCase):
    def test_deep_research_preset_gets_reasoning_effort(self) -> None:
        payload = build_perplexity_reasoning_payload(
            "high",
            target={"preset": "deep-research"},
        )
        self.assertEqual(payload, {"reasoning_effort": "high"})

    def test_none_omitted_for_deep_research(self) -> None:
        self.assertEqual(
            build_perplexity_reasoning_payload("none", target={"preset": "deep-research"}),
            {},
        )

    def test_fast_search_uses_model_defaults(self) -> None:
        self.assertEqual(
            build_perplexity_reasoning_payload("high", target={"preset": "fast-search"}),
            {},
        )

    def test_extra_high_maps_to_high(self) -> None:
        payload = build_perplexity_reasoning_payload(
            "extra_high",
            target={"preset": "deep-research"},
        )
        self.assertEqual(payload, {"reasoning_effort": "high"})


def _mock_response() -> MagicMock:
    response = MagicMock()
    response.model_dump.return_value = {
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}],
        "content": [{"type": "text", "text": "ok"}],
        "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
    }
    response.output_text = "ok"
    response.text = "ok"
    response.content = "ok"
    return response


class AdapterRunReasoningIntegrationTest(unittest.TestCase):
    @patch("openai.OpenAI")
    def test_openai_adapter_passes_reasoning_effort(self, openai_cls: MagicMock) -> None:
        client = MagicMock()
        client.responses.create.return_value = _mock_response()
        openai_cls.return_value = client

        adapter = OpenAIAdapter({"api_key": "test"})
        adapter.run(
            prompt="hello",
            model="gpt-5.3-codex-max",
            require_search=False,
            return_citations=False,
            files=None,
            output_format=None,
            adapter_options=None,
            reasoning_level="extra_high",
        )

        payload = client.responses.create.call_args.kwargs
        self.assertEqual(payload["reasoning"], {"effort": "xhigh"})

    @patch("anthropic.Anthropic")
    def test_anthropic_adapter_passes_adaptive_effort(self, anthropic_cls: MagicMock) -> None:
        client = MagicMock()
        client.messages.create.return_value = _mock_response()
        anthropic_cls.return_value = client

        adapter = AnthropicAdapter({"api_key": "test"})
        adapter.run(
            prompt="hello",
            model="claude-opus-4-7",
            require_search=False,
            return_citations=False,
            files=None,
            output_format=None,
            adapter_options=None,
            reasoning_level="medium",
        )

        payload = client.messages.create.call_args.kwargs
        self.assertEqual(payload["thinking"], {"type": "adaptive"})
        self.assertEqual(payload["output_config"], {"effort": "medium"})

    @patch("xai_sdk.Client")
    def test_grok_adapter_passes_reasoning_effort(self, client_cls: MagicMock) -> None:
        response = _mock_response()
        chat = MagicMock()
        chat.sample.return_value = response
        client = MagicMock()
        client.chat.create.return_value = chat
        client_cls.return_value = client

        adapter = GrokAdapter({"api_key": "test"})
        adapter.run(
            prompt="hello",
            model="grok-4.3",
            require_search=False,
            return_citations=False,
            files=None,
            output_format=None,
            adapter_options=None,
            reasoning_level="none",
        )

        payload = client.chat.create.call_args.kwargs
        self.assertEqual(payload["reasoning"], {"effort": "none"})

    @patch("perplexity.Perplexity")
    def test_perplexity_adapter_passes_reasoning_effort(self, perplexity_cls: MagicMock) -> None:
        client = MagicMock()
        client.responses.create.return_value = _mock_response()
        perplexity_cls.return_value = client

        adapter = PerplexityAdapter({"api_key": "test"})
        adapter.run(
            prompt="hello",
            model="deep-research",
            require_search=False,
            return_citations=False,
            files=None,
            output_format=None,
            adapter_options=None,
            reasoning_level="low",
        )

        payload = client.responses.create.call_args.kwargs
        self.assertEqual(payload["reasoning_effort"], "low")
        self.assertEqual(payload["preset"], "deep-research")


if __name__ == "__main__":
    unittest.main()
