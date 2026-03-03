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



def _extract_candidate_json_blocks(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []

    blocks = []

    if (stripped.startswith("{") and stripped.endswith("}")) or (
        stripped.startswith("[") and stripped.endswith("]")
    ):
        try:
            json.loads(stripped)
            blocks.append(stripped)
        except json.JSONDecodeError:
            pass  # Fall through to raw_decode path

    decoder = json.JSONDecoder()
    start = 0
    length = len(stripped)
    while start < length:
        char = stripped[start]
        if char not in "[{":
            start += 1
            continue
        try:
            _, end = decoder.raw_decode(stripped[start:])
            blocks.append(stripped[start : start + end])
            start += end
        except json.JSONDecodeError:
            start += 1

    return blocks



def coerce_output(
    text: str,
    output_format: type[BaseModel] | None,
) -> Any:
    """Return plain text or validated Pydantic model instance."""

    if output_format is None:
        return text

    blocks = _extract_candidate_json_blocks(text)
    if not blocks:
        raise ValueError("Could not find JSON object/array in model output.")

    adapter = TypeAdapter(output_format)
    
    last_error = None
    for payload in blocks:
        try:
            return adapter.validate_json(payload)
        except Exception as e:
            last_error = e
            
    raise ValueError(f"No extracted JSON block passed validation. Last error: {last_error}")
