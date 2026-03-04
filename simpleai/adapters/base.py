"""Base adapter class for provider integrations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel

from simpleai.types import AdapterResponse, PromptInput


class BaseAdapter(ABC):
    """Provider adapter interface."""

    provider_name: str = "unknown"
    supports_binary_files: bool = False

    def __init__(self, provider_settings: dict[str, Any]) -> None:
        self.provider_settings = provider_settings

    @abstractmethod
    def run(
        self,
        *,
        prompt: PromptInput,
        model: str,
        require_search: bool,
        return_citations: bool,
        files: Sequence[Path] | None,
        output_format: type[BaseModel] | None,
        adapter_options: dict[str, Any] | None,
    ) -> AdapterResponse:
        """Execute the prompt on the provider and return normalized output."""
