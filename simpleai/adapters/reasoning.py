"""Reasoning level normalization and provider-specific payload mapping."""

from typing import Any, Literal, TypeAlias

ReasoningLevel: TypeAlias = Literal["none", "low", "medium", "high", "extra_high"]

REASONING_LEVELS: tuple[ReasoningLevel, ...] = (
    "none",
    "low",
    "medium",
    "high",
    "extra_high",
)

_REASONING_ALIASES: dict[str, ReasoningLevel] = {
    "xhigh": "extra_high",
    "extra-high": "extra_high",
    "extrahigh": "extra_high",
}


def parse_reasoning_level(value: str | ReasoningLevel | None) -> ReasoningLevel | None:
    """Validate and normalize a reasoning level string."""

    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    normalized = str(value).strip().lower().replace(" ", "_")
    normalized = _REASONING_ALIASES.get(normalized, normalized)
    if normalized not in REASONING_LEVELS:
        allowed = ", ".join(REASONING_LEVELS)
        raise ValueError(f"Invalid reasoning_level {value!r}. Must be one of: {allowed}")
    return normalized  # type: ignore[return-value]


def resolve_reasoning_level(
    reasoning_level: ReasoningLevel | None,
    *,
    supports_none: bool,
    supports_extra_high: bool,
) -> ReasoningLevel | None:
    """Map requested level to the nearest level supported by the provider/model."""

    if reasoning_level is None:
        return None

    resolved: ReasoningLevel = reasoning_level
    if resolved == "none" and not supports_none:
        resolved = "low"
    if resolved == "extra_high" and not supports_extra_high:
        resolved = "high"
    return resolved


def openai_model_supports_reasoning(model: str) -> bool:
    lowered = model.lower()
    if lowered.startswith(("gpt-4", "gpt-3", "gpt-image", "dall-e", "tts-", "whisper")):
        return False
    if lowered.startswith(("o1", "o2", "o3", "o4", "gpt-5", "codex")):
        return True
    return "reasoning" in lowered


def openai_model_supports_none(model: str) -> bool:
    lowered = model.lower()
    return any(
        token in lowered
        for token in (
            "gpt-5.1",
            "gpt-5.2",
            "gpt-5.3",
            "gpt-5.4",
            "gpt-5.5",
            "gpt-5-1",
            "gpt-5-2",
            "gpt-5-3",
            "gpt-5-4",
            "gpt-5-5",
        )
    )


def openai_model_supports_extra_high(model: str) -> bool:
    lowered = model.lower()
    return "codex-max" in lowered or "codex_max" in lowered


def build_openai_reasoning_payload(
    reasoning_level: ReasoningLevel | None,
    *,
    model: str,
) -> dict[str, Any]:
    if reasoning_level is None or not openai_model_supports_reasoning(model):
        return {}

    resolved = resolve_reasoning_level(
        reasoning_level,
        supports_none=openai_model_supports_none(model),
        supports_extra_high=openai_model_supports_extra_high(model),
    )
    if resolved is None:
        return {}

    effort_map: dict[ReasoningLevel, str] = {
        "none": "none",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "extra_high": "xhigh",
    }
    return {"reasoning": {"effort": effort_map[resolved]}}


def anthropic_model_supports_adaptive(model: str) -> bool:
    lowered = model.lower()
    return any(
        token in lowered
        for token in (
            "opus-4-6",
            "opus-4-7",
            "sonnet-4-6",
            "mythos",
        )
    )


def anthropic_model_supports_extra_high(model: str) -> bool:
    return "opus-4-7" in model.lower()


def anthropic_model_supports_thinking(model: str) -> bool:
    lowered = model.lower()
    if any(token in lowered for token in ("haiku-3", "haiku-3-5", "instant")):
        return False
    return any(token in lowered for token in ("opus", "sonnet", "mythos"))


def build_anthropic_reasoning_payload(
    reasoning_level: ReasoningLevel | None,
    *,
    model: str,
) -> dict[str, Any]:
    if reasoning_level is None or not anthropic_model_supports_thinking(model):
        return {}

    if reasoning_level == "none":
        return {}

    resolved = resolve_reasoning_level(
        reasoning_level,
        supports_none=False,
        supports_extra_high=anthropic_model_supports_extra_high(model),
    )
    if resolved is None:
        return {}

    if anthropic_model_supports_adaptive(model):
        effort_map: dict[ReasoningLevel, str] = {
            "low": "low",
            "medium": "medium",
            "high": "high",
            "extra_high": "xhigh",
        }
        return {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": effort_map[resolved]},
        }

    budget_map: dict[ReasoningLevel, int] = {
        "low": 2048,
        "medium": 8000,
        "high": 16000,
        "extra_high": 32000,
    }
    return {
        "thinking": {
            "type": "enabled",
            "budget_tokens": budget_map[resolved],
        }
    }


def gemini_model_supports_thinking_level(model: str) -> bool:
    return "gemini-3" in model.lower()


def gemini_model_supports_thinking(model: str) -> bool:
    lowered = model.lower()
    if lowered.startswith("gemini-embedding") or lowered.startswith("text-embedding"):
        return False
    return lowered.startswith("gemini-")


def gemini_model_supports_minimal_level(model: str) -> bool:
    return "flash" in model.lower()


def build_gemini_reasoning_config_kwargs(
    reasoning_level: ReasoningLevel | None,
    *,
    model: str,
) -> dict[str, Any]:
    """Return GenerateContentConfig kwargs for reasoning (thinking_config)."""

    if reasoning_level is None or not gemini_model_supports_thinking(model):
        return {}

    if reasoning_level == "none":
        if gemini_model_supports_thinking_level(model):
            resolved: ReasoningLevel = "low"
        else:
            return {"thinking_config": {"thinking_budget": 0}}
    else:
        resolved = resolve_reasoning_level(
            reasoning_level,
            supports_none=False,
            supports_extra_high=False,
        )
        if resolved is None:
            return {}

    if gemini_model_supports_thinking_level(model):
        if resolved == "none":
            level = "MINIMAL" if gemini_model_supports_minimal_level(model) else "LOW"
        else:
            level_map: dict[ReasoningLevel, str] = {
                "low": "LOW",
                "medium": "MEDIUM",
                "high": "HIGH",
                "extra_high": "HIGH",
            }
            level = level_map[resolved]
        return {"thinking_config": {"thinking_level": level}}

    budget_map: dict[ReasoningLevel, int] = {
        "none": 0,
        "low": 1024,
        "medium": 4096,
        "high": 8192,
        "extra_high": 16384,
    }
    return {"thinking_config": {"thinking_budget": budget_map[resolved]}}


def grok_model_supports_reasoning(model: str) -> bool:
    lowered = model.lower()
    return "grok" in lowered and any(token in lowered for token in ("reasoning", "4", "beta"))


def grok_model_supports_none(model: str) -> bool:
    lowered = model.lower()
    if "multi-agent" in lowered:
        return False
    return True


def grok_model_supports_extra_high(model: str) -> bool:
    return False


def build_grok_reasoning_payload(
    reasoning_level: ReasoningLevel | None,
    *,
    model: str,
) -> dict[str, Any]:
    if reasoning_level is None or not grok_model_supports_reasoning(model):
        return {}

    resolved = resolve_reasoning_level(
        reasoning_level,
        supports_none=grok_model_supports_none(model),
        supports_extra_high=grok_model_supports_extra_high(model),
    )
    if resolved is None:
        return {}

    effort_map: dict[ReasoningLevel, str] = {
        "none": "none",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "extra_high": "high",
    }
    return {"reasoning_effort": effort_map[resolved]}


def perplexity_target_supports_reasoning(target: dict[str, str]) -> bool:
    preset = (target.get("preset") or "").lower()
    model = (target.get("model") or "").lower()
    if "deep-research" in preset or "deep-research" in model:
        return True
    if "reasoning" in preset or "reasoning" in model:
        return True
    return False


def build_perplexity_reasoning_payload(
    reasoning_level: ReasoningLevel | None,
    *,
    target: dict[str, str],
) -> dict[str, Any]:
    if reasoning_level is None or not perplexity_target_supports_reasoning(target):
        return {}

    if reasoning_level == "none":
        return {}

    resolved = resolve_reasoning_level(
        reasoning_level,
        supports_none=False,
        supports_extra_high=False,
    )
    if resolved is None:
        return {}

    effort_map: dict[ReasoningLevel, str] = {
        "low": "low",
        "medium": "medium",
        "high": "high",
        "extra_high": "high",
    }
    return {"reasoning_effort": effort_map[resolved]}


def merge_reasoning_payload(payload: dict[str, Any], reasoning_payload: dict[str, Any]) -> None:
    """Deep-merge reasoning settings into an adapter payload without clobbering nested dicts."""

    for key, value in reasoning_payload.items():
        existing = payload.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged = dict(existing)
            merged.update(value)
            payload[key] = merged
        else:
            payload[key] = value
