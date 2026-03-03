"""Google Gemini adapter using google-genai SDK."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from simpleai.adapters.base import BaseAdapter
from simpleai.exceptions import ProviderError
from simpleai.types import AdapterResponse, Citation, PromptInput

logger = logging.getLogger(__name__)


def _is_retryable_gemini_error(exc: BaseException) -> bool:
    """Check if the exception from Gemini is retryable (e.g., 503 or 429)."""
    exc_str = str(exc)
    return "503" in exc_str or "429" in exc_str or "UNAVAILABLE" in exc_str or "Too Many Requests" in exc_str


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

        use_vertexai = provider_settings.get("use_vertexai")
        if use_vertexai is None:
            use_vertexai = str(os.getenv("GEMINI_USE_VERTEXAI", "")).lower() in ("true", "1", "yes")

        if use_vertexai:
            project = provider_settings.get("vertexai_project") or os.getenv("GEMINI_VERTEXAI_PROJECT")
            location = provider_settings.get("vertexai_location") or os.getenv("GEMINI_VERTEXAI_LOCATION")
            self.client = genai.Client(vertexai=True, project=project, location=location)
            self._project = project
        else:
            api_key = provider_settings.get("api_key") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            self.client = genai.Client(api_key=api_key)

        self.types = types
        self._genai = genai
        self._use_vertexai = use_vertexai

    def _build_contents(self, prompt: PromptInput, files: Sequence[Path] | None, client: Any = None) -> Any:
        client = client or self.client
        contents: list[Any] = []

        if files:
            @retry(
                retry=retry_if_exception(_is_retryable_gemini_error),
                wait=wait_exponential(multiplier=1, min=2, max=120),
                stop=stop_after_attempt(9),
                before_sleep=before_sleep_log(logger, logging.WARNING),
                reraise=True,
            )
            def _upload(p: Path) -> Any:
                return client.files.upload(file=str(p))

            for path in files:
                uploaded = _upload(path)
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
        seen: set[tuple[Any, ...]] = set()

        def append_citation(
            *,
            url: str | None,
            title: str | None,
            source: str | None,
            snippet: str | None,
            start_index: int | None = None,
            end_index: int | None = None,
            raw: dict[str, Any],
        ) -> None:
            key = (url, title, source, snippet, start_index, end_index)
            if key in seen:
                return
            seen.add(key)
            citations.append(
                Citation(
                    provider=self.provider_name,
                    url=url,
                    title=title,
                    source=source,
                    snippet=snippet,
                    start_index=start_index,
                    end_index=end_index,
                    raw=raw,
                )
            )

        for candidate in response_dict.get("candidates", []):
            # Citation metadata (inline offsets + URI/title).
            citation_meta = candidate.get("citation_metadata") or candidate.get("citationMetadata") or {}
            for item in citation_meta.get("citations") or []:
                append_citation(
                    url=item.get("uri"),
                    title=item.get("title"),
                    source=item.get("uri"),
                    snippet=None,
                    start_index=item.get("start_index") or item.get("startIndex"),
                    end_index=item.get("end_index") or item.get("endIndex"),
                    raw=item,
                )

            # Grounding metadata from Google Search tool.
            grounding = candidate.get("grounding_metadata") or candidate.get("groundingMetadata") or {}
            chunks = grounding.get("grounding_chunks") or grounding.get("groundingChunks") or []
            for chunk in chunks:
                web = chunk.get("web") or {}
                if web:
                    append_citation(
                        url=web.get("uri") or web.get("url"),
                        title=web.get("title"),
                        source=web.get("domain") or web.get("uri") or web.get("url"),
                        snippet=None,
                        raw=chunk,
                    )

                retrieved = chunk.get("retrieved_context") or chunk.get("retrievedContext") or {}
                if retrieved:
                    append_citation(
                        url=retrieved.get("uri"),
                        title=retrieved.get("title") or retrieved.get("document_name") or retrieved.get("documentName"),
                        source=retrieved.get("document_name") or retrieved.get("documentName") or retrieved.get("uri"),
                        snippet=retrieved.get("text"),
                        raw=chunk,
                    )

                maps = chunk.get("maps") or {}
                if maps:
                    append_citation(
                        url=maps.get("uri"),
                        title=maps.get("title"),
                        source="google_maps",
                        snippet=maps.get("text"),
                        raw=chunk,
                    )

            # Query metadata can still be useful provenance even when chunks are absent.
            for query in grounding.get("web_search_queries") or grounding.get("webSearchQueries") or []:
                append_citation(
                    url=None,
                    title=None,
                    source="google_search_query",
                    snippet=str(query),
                    raw={"query": query},
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
            client = self.client
            if getattr(self, "_use_vertexai", False) and model.startswith("gemini-3.1"):
                client = self._genai.Client(vertexai=True, project=self._project, location="global")

            default_max_tokens = 65536 if "gemini-3.1" in model else 8192
            config_kwargs: dict[str, Any] = {
                "max_output_tokens": int(self.provider_settings.get("max_output_tokens", default_max_tokens)),
            }

            if require_search:
                config_kwargs["tools"] = [
                    self.types.Tool(google_search=self.types.GoogleSearch())
                ]
                config_kwargs.setdefault(
                    "system_instruction",
                    "Use Google Search to ground your answer and provide citations to sources. Ensure that all cited URLs are publicly accessible. Do not cite links that result in a 404 or 5xx error.",
                )

            if output_format is not None:
                config_kwargs["response_mime_type"] = "application/json"
                config_kwargs["response_schema"] = output_format.model_json_schema()

            if adapter_options:
                config_kwargs.update(adapter_options)

            config = self.types.GenerateContentConfig(**config_kwargs)
            contents = self._build_contents(prompt, files, client=client)

            @retry(
                retry=retry_if_exception(_is_retryable_gemini_error),
                wait=wait_exponential(multiplier=1, min=2, max=120),
                stop=stop_after_attempt(7),
                before_sleep=before_sleep_log(logger, logging.WARNING),
                reraise=True,
            )
            def _generate() -> Any:
                return client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )

            response = _generate()

            response_dict = response.model_dump(mode="json") if hasattr(response, "model_dump") else {}
            text = getattr(response, "text", "") or ""
            if not text and response_dict:
                chunks: list[str] = []
                for candidate in response_dict.get("candidates", []):
                    content = candidate.get("content") or {}
                    for part in content.get("parts") or []:
                        if part.get("text"):
                            chunks.append(part["text"])
                text = "\n".join(chunks)

            if not text.strip():
                raise ProviderError(f"Gemini returned empty response. Raw payload: {response_dict}")

            # Check if generation was cut off
            for candidate in response_dict.get("candidates", []):
                fr = str(candidate.get("finish_reason") or candidate.get("finishReason") or "").upper()
                if fr in ("MAX_TOKENS", "2"):
                    if output_format is not None:
                        max_tokens = config_kwargs.get("max_output_tokens", "unknown")
                        raise ProviderError(f"Gemini hit MAX_TOKENS before finishing the JSON response. Try increasing max_output_tokens (currently {max_tokens} for model {model}).")
                    else:
                        logger.warning("Gemini hit MAX_TOKENS. The response may be incomplete.")

            citations = self._extract_citations(response_dict) if return_citations else []
            return AdapterResponse(text=text, citations=citations, raw=response_dict)

        except Exception as exc:  # pragma: no cover - network/provider behavior
            raise ProviderError(f"Gemini adapter failed: {exc}") from exc
