"""Google Gemini adapter using google-genai SDK."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel

from simpleai.adapters.base import BaseAdapter
from simpleai.exceptions import ProviderError
from simpleai.types import AdapterResponse, Citation, PromptInput


class GeminiAdapter(BaseAdapter):
    provider_name = "gemini"
    supports_binary_files = True

    def __init__(self, provider_settings: dict[str, Any]) -> None:
        super().__init__(provider_settings)

        try:
            from google import genai
            from google.genai import types
        except Exception as exc:  # pragma: no cover - dependency missing path
            raise ProviderError("google-genai package is required for GeminiAdapter.") from exc

        api_key = provider_settings.get("api_key") or os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.types = types

    def _build_contents(self, prompt: PromptInput, files: Sequence[Path] | None) -> Any:
        contents: list[Any] = []

        if files:
            for path in files:
                uploaded = self.client.files.upload(file=str(path))
                contents.append(uploaded)

        if isinstance(prompt, str):
            contents.append(prompt)
        else:
            contents.extend(str(item) for item in prompt)

        if len(contents) == 1:
            return contents[0]
        return contents

    def _extract_citations(self, response_dict: dict[str, Any]) -> list[Citation]:
        citations: list[Citation] = []

        for candidate in response_dict.get("candidates", []):
            grounding = candidate.get("grounding_metadata") or {}
            for chunk in grounding.get("grounding_chunks") or []:
                web = chunk.get("web") or {}
                if not web:
                    continue
                citations.append(
                    Citation(
                        provider=self.provider_name,
                        url=web.get("uri"),
                        title=web.get("title"),
                        source=web.get("domain"),
                        raw=chunk,
                    )
                )

        return citations

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
        try:
            config_kwargs: dict[str, Any] = {
                "max_output_tokens": int(self.provider_settings.get("max_output_tokens", 8192)),
            }

            if require_search:
                config_kwargs["tools"] = [
                    self.types.Tool(google_search=self.types.GoogleSearch())
                ]

            if output_format is not None:
                config_kwargs["response_mime_type"] = "application/json"
                config_kwargs["response_schema"] = output_format.model_json_schema()

            if adapter_options:
                config_kwargs.update(adapter_options)

            config = self.types.GenerateContentConfig(**config_kwargs)
            response = self.client.models.generate_content(
                model=model,
                contents=self._build_contents(prompt, files),
                config=config,
            )

            response_dict = response.model_dump(mode="json") if hasattr(response, "model_dump") else {}
            text = getattr(response, "text", "") or ""
            if not text and response_dict:
                chunks: list[str] = []
                for candidate in response_dict.get("candidates", []):
                    content = candidate.get("content") or {}
                    for part in content.get("parts") or []:
                        if "text" in part:
                            chunks.append(part["text"])
                text = "\n".join(chunks)

            citations = self._extract_citations(response_dict) if return_citations else []
            return AdapterResponse(text=text, citations=citations, raw=response_dict)

        except Exception as exc:  # pragma: no cover - network/provider behavior
            raise ProviderError(f"Gemini adapter failed: {exc}") from exc
