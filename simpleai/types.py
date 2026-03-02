"""Shared types for SimpleAI."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

PromptInput: TypeAlias = str | list[str]


@dataclass(slots=True)
class Citation:
    """Normalized citation shape returned to callers."""

    provider: str
    url: str | None = None
    title: str | None = None
    source: str | None = None
    snippet: str | None = None
    citation_id: str | None = None
    start_index: int | None = None
    end_index: int | None = None
    raw: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AdapterResponse:
    """Internal response from provider adapters."""

    text: str
    citations: list[Citation] = field(default_factory=list)
    raw: dict[str, Any] | None = None


@dataclass(slots=True)
class ExtractedFile:
    """Text extracted from a file path."""

    path: Path
    text: str


@dataclass(slots=True)
class PromptRunContext:
    """Runtime details captured for logging."""

    provider: str
    model: str
    started_at: float
