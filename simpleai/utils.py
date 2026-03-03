"""Utility helpers for SimpleAI."""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

from pydantic import BaseModel, TypeAdapter

from .types import Citation, PromptInput

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

    # Sort blocks by length descending. The intended payload is typically the largest JSON block.
    blocks.sort(key=len, reverse=True)

    adapter = TypeAdapter(output_format)
    
    first_error = None
    for payload in blocks:
        try:
            return adapter.validate_json(payload)
        except Exception as e:
            if first_error is None:
                first_error = e
            
    raise ValueError(f"No extracted JSON block passed validation. Last error: {first_error}")


def _check_url_alive(url: str) -> bool:
    """Return True unless the URL explicitly returns 404 or 5xx."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"},
        method="HEAD",
    )
    try:
        # Timeout to avoid hanging on dead servers
        with urllib.request.urlopen(req, timeout=5) as response:
            return True
    except urllib.error.HTTPError as e:
        if e.code == 404 or e.code >= 500:
            return False
        # 403, 401, etc. are considered alive
        return True
    except urllib.error.URLError:
        # DNS errors or connection refused might be transient or permanent.
        # We err on the side of "not a confirmed 404" to avoid stripping valid internal or temperamental links.
        return True
    except Exception:
        return True


def validate_citations(citations: list[Citation]) -> None:
    """Check citation URLs concurrently and set `is_alive` boolean."""
    def _validate(citation: Citation) -> None:
        if not citation.url:
            citation.is_alive = None
            return
        if not citation.url.startswith("http"):
            citation.is_alive = True
            return
        citation.is_alive = _check_url_alive(citation.url)

    with ThreadPoolExecutor(max_workers=10) as executor:
        list(executor.map(_validate, citations))

