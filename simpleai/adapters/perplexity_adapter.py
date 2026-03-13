"""Perplexity adapter using the Responses API."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Any, ClassVar, Sequence
from urllib.parse import urlparse

from pydantic import BaseModel

from simpleai.adapters.base import BaseAdapter
from simpleai.exceptions import ProviderError
from simpleai.schema import perplexity_response_schema
from simpleai.types import AdapterResponse, Citation, PromptInput


class PerplexityAdapter(BaseAdapter):
    provider_name = "perplexity"
    supports_binary_files = False
    _GENERIC_SOURCE_LABELS: ClassVar[set[str]] = {"web"}

    _PRESET_ALIASES: ClassVar[dict[str, str]] = {
        "fast-search": "fast-search",
        "pro-search": "pro-search",
        "deep-research": "deep-research",
        # Backward-compatible aliases from older Sonar naming.
        "sonar": "fast-search",
        "sonar-pro": "pro-search",
        "sonar-reasoning": "pro-search",
        "sonar-reasoning-pro": "deep-research",
        "sonar-deep-research": "deep-research",
    }

    def __init__(self, provider_settings: dict[str, Any]) -> None:
        super().__init__(provider_settings)

        try:
            from perplexity import Perplexity
        except Exception as exc:  # pragma: no cover - dependency missing path
            raise ProviderError("perplexityai package is required for PerplexityAdapter.") from exc

        api_key = (
            provider_settings.get("api_key")
            or os.getenv("PERPLEXITY_API_KEY")
            or os.getenv("PPLX_API_KEY")
        )
        self.client = Perplexity(api_key=api_key)

    def _build_input(self, prompt: PromptInput) -> str | list[dict[str, Any]]:
        if isinstance(prompt, str):
            return prompt

        messages: list[dict[str, Any]] = []
        for turn in prompt:
            messages.append({
                "type": "message",
                "role": "user",
                "content": str(turn),
            })
        if not messages:
            return ""
        return messages

    def _resolve_model_target(self, model: str) -> dict[str, str]:
        normalized = model.strip()
        lowered = normalized.lower()

        preset = self._PRESET_ALIASES.get(lowered)
        if preset:
            return {"preset": preset}

        # Responses API model names are provider/model.
        if "/" in normalized:
            return {"model": normalized}

        # Heuristic provider prefixing for common raw model names.
        if lowered.startswith(("gpt-", "o1", "o3", "o4")):
            return {"model": f"openai/{normalized}"}
        if lowered.startswith("claude"):
            return {"model": f"anthropic/{normalized}"}
        if lowered.startswith("gemini"):
            return {"model": f"google/{normalized}"}
        if lowered.startswith("grok"):
            return {"model": f"xai/{normalized}"}
        if lowered.startswith("sonar"):
            return {"model": f"perplexity/{normalized}"}

        return {"model": normalized}

    def _append_json_instruction(
        self,
        prompt: PromptInput,
        schema: dict[str, Any],
    ) -> PromptInput:
        instruction = (
            "Return only valid JSON matching this schema. "
            "Do not include markdown fences or explanatory text.\n"
            f"{json.dumps(schema, ensure_ascii=True)}"
        )
        if isinstance(prompt, str):
            return f"{prompt}\n\n{instruction}"

        prompt_list = [str(item) for item in prompt]
        prompt_list.append(instruction)
        return prompt_list

    def _extract_citations(self, response_dict: dict[str, Any]) -> list[Citation]:
        citations: list[Citation] = []
        seen: set[tuple[Any, ...]] = set()

        search_results: list[dict[str, Any]] = []
        search_results_by_url: dict[str, dict[str, Any]] = {}
        for output_item in response_dict.get("output", []):
            if output_item.get("type") != "search_results":
                continue
            for result in output_item.get("results") or []:
                if not isinstance(result, dict):
                    continue
                search_results.append(result)
                url = result.get("url")
                if url and url not in search_results_by_url:
                    search_results_by_url[url] = result

        def append_citation(
            *,
            url: str | None,
            title: str | None,
            source: str | None,
            snippet: str | None = None,
            start_index: int | None = None,
            end_index: int | None = None,
            raw: dict[str, Any],
        ) -> None:
            if not url and not title and not source:
                return
            source_label = self._normalize_source_label(url=url, title=title, source=source)
            key = (url, title, source_label, snippet, start_index, end_index)
            if key in seen:
                return
            seen.add(key)
            citations.append(
                Citation(
                    provider=self.provider_name,
                    url=url,
                    title=title,
                    source=source_label,
                    snippet=snippet,
                    start_index=start_index,
                    end_index=end_index,
                    raw=raw,
                )
            )

        for output_item in response_dict.get("output", []):
            if output_item.get("type") != "message":
                continue
            for part in output_item.get("content", []):
                for annotation in part.get("annotations") or []:
                    if not isinstance(annotation, dict):
                        continue
                    url = annotation.get("url")
                    matched_result = search_results_by_url.get(url) if url else None
                    title = annotation.get("title") or (matched_result or {}).get("title")
                    raw = annotation
                    if matched_result:
                        raw = {"annotation": annotation, "search_result": matched_result}
                    append_citation(
                        url=url,
                        title=title,
                        source=annotation.get("source") or (matched_result or {}).get("source"),
                        snippet=(matched_result or {}).get("snippet"),
                        start_index=annotation.get("start_index"),
                        end_index=annotation.get("end_index"),
                        raw=raw,
                    )

        return citations

    def _normalize_source_label(
        self,
        *,
        url: str | None,
        title: str | None,
        source: str | None,
    ) -> str | None:
        if title:
            return title

        if source and source.strip().lower() not in self._GENERIC_SOURCE_LABELS:
            return source

        if url:
            hostname = urlparse(url).hostname or ""
            if hostname.startswith("www."):
                hostname = hostname[4:]
            if hostname:
                return hostname

        return source or url

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
        del files  # unsupported in this adapter; caller should pass extracted text instead

        try:
            target = self._resolve_model_target(model)
            payload: dict[str, Any] = {
                "input": self._build_input(prompt),
                **target,
            }

            # Search is implicit in Perplexity presets; avoid redundant tool payload there.
            if require_search and "model" in target:
                payload["tools"] = [{"type": "web_search"}]

            schema: dict[str, Any] | None = None
            if output_format is not None:
                schema = perplexity_response_schema(output_format)
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "schema": schema,
                    },
                }

            if adapter_options:
                payload.update(adapter_options)

            response = None
            try:
                response = self.client.responses.create(**payload)
            except Exception as exc:
                # Retry without response_format for models/presets that reject it.
                # We still ask for JSON so run_prompt can validate with Pydantic.
                message = str(exc).lower()
                is_bad_request = ("400" in message) or ("invalid request" in message) or ("invalid schema" in message)
                if output_format is None or schema is None or not is_bad_request:
                    raise
                retry_payload = deepcopy(payload)
                retry_payload.pop("response_format", None)
                retry_payload["input"] = self._build_input(
                    self._append_json_instruction(prompt, schema)
                )
                response = self.client.responses.create(**retry_payload)

            response_dict = response.model_dump(mode="json") if hasattr(response, "model_dump") else {}
            text = getattr(response, "output_text", "") or ""
            if not text and response_dict:
                chunks: list[str] = []
                for output in response_dict.get("output", []):
                    if output.get("type") != "message":
                        continue
                    for part in output.get("content", []):
                        if part.get("type") == "output_text":
                            chunks.append(part.get("text", ""))
                text = "".join(chunks)

            citations = self._extract_citations(response_dict) if return_citations else []
            return AdapterResponse(text=text, citations=citations, raw=response_dict)

        except Exception as exc:  # pragma: no cover - network/provider behavior
            raise ProviderError(f"Perplexity adapter failed: {exc}") from exc
