"""Utility helpers for SimpleAI."""

from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import BaseModel, TypeAdapter

from .types import PromptInput

T = TypeVar("T")


def normalize_prompt(prompt: PromptInput) -> str:
    """Normalize supported prompt shapes into a single string."""

    if isinstance(prompt, str):
        return prompt

    if not prompt:
        return ""

    turns = [str(item) for item in prompt]
    return "\n\n".join(f"Turn {idx + 1}: {item}" for idx, item in enumerate(turns))



def pydantic_schema(output_format: type[BaseModel] | None) -> dict[str, Any] | None:
    if output_format is None:
        return None
    return output_format.model_json_schema()



def _extract_candidate_json(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        raise ValueError("No content to parse.")

    if (stripped.startswith("{") and stripped.endswith("}")) or (
        stripped.startswith("[") and stripped.endswith("]")
    ):
        return stripped

    decoder = json.JSONDecoder()
    for start in range(len(stripped)):
        char = stripped[start]
        if char not in "[{":
            continue
        try:
            _, end = decoder.raw_decode(stripped[start:])
        except json.JSONDecodeError:
            continue
        return stripped[start : start + end]

    raise ValueError("Could not find JSON object/array in model output.")



def coerce_output(
    text: str,
    output_format: type[BaseModel] | None,
) -> Any:
    """Return plain text or validated Pydantic model instance."""

    if output_format is None:
        return text

    json_payload = _extract_candidate_json(text)
    adapter = TypeAdapter(output_format)
    return adapter.validate_json(json_payload)
