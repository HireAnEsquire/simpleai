"""Perplexity adapter using the Responses API."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import re
from typing import Any, ClassVar, Sequence
from urllib.parse import urlparse

from pydantic import BaseModel

from simpleai.adapters.base import BaseAdapter
from simpleai.adapters.reasoning import (
    ReasoningLevel,
    build_perplexity_reasoning_payload,
    merge_reasoning_payload,
)
from simpleai.exceptions import ProviderError
from simpleai.schema import perplexity_response_schema
from simpleai.types import AdapterResponse, Citation, PromptInput


class PerplexityAdapter(BaseAdapter):
    provider_name = "perplexity"
    supports_binary_files = False
    _GENERIC_SOURCE_LABELS: ClassVar[set[str]] = {"web"}
    _CITATION_BLOCK_RE: ClassVar[re.Pattern[str]] = re.compile(r"\[([^\[\]]+)\]")
    _CITATION_TOKEN_RE: ClassVar[re.Pattern[str]] = re.compile(r"(?:(?P<kind>[a-z_]+):)?(?P<index>\d+)$")

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

    def _extract_citations(self, response_dict: dict[str, Any]) -> tuple[list[Citation], dict[tuple[str | None, int], int]]:
        citations: list[Citation] = []
        seen_keys: dict[tuple[Any, ...], int] = {}
        marker_mapping: dict[tuple[str | None, int], int] = {}

        search_results: list[dict[str, Any]] = []
        search_results_by_url: dict[str, dict[str, Any]] = {}
        search_results_by_id: dict[int, dict[str, Any]] = {}
        fetch_results_by_index: dict[int, dict[str, Any]] = {}
        for output_item in response_dict.get("output", []):
            output_type = output_item.get("type")
            if output_type == "search_results":
                for result in output_item.get("results") or []:
                    if not isinstance(result, dict):
                        continue
                    search_results.append(result)
                    url = result.get("url")
                    if url and url not in search_results_by_url:
                        search_results_by_url[url] = result
                    
                    # Some responses use explicit IDs; others rely on 1-based index of the search result.
                    result_id = result.get("id")
                    if not isinstance(result_id, int):
                        result_id = len(search_results)
                    if result_id not in search_results_by_id:
                        search_results_by_id[result_id] = result
            if output_type == "fetch_url_results":
                for index, result in enumerate(output_item.get("contents") or [], start=1):
                    if isinstance(result, dict) and index not in fetch_results_by_index:
                        fetch_results_by_index[index] = result

        def append_citation(
            *,
            url: str | None,
            title: str | None,
            source: str | None,
            snippet: str | None = None,
            start_index: int | None = None,
            end_index: int | None = None,
            raw: dict[str, Any],
        ) -> int | None:
            if not url and not title and not source:
                return None
            source_label = self._normalize_source_label(url=url, title=title, source=source)
            key = (url, title, source_label, snippet, start_index, end_index)
            if key in seen_keys:
                return seen_keys[key]
            
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
            new_idx = len(citations)
            seen_keys[key] = new_idx
            return new_idx

        for output_item in response_dict.get("output", []):
            if output_item.get("type") != "message":
                continue
            for part in output_item.get("content", []):
                part_text = part.get("text") or ""
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
                for marker, result_type, result_index in self._extract_citation_references(part_text):
                    if result_type == "page":
                        fetch_result = fetch_results_by_index.get(result_index)
                        if not fetch_result:
                            continue
                        new_idx = append_citation(
                            url=fetch_result.get("url"),
                            title=fetch_result.get("title"),
                            source=None,
                            snippet=fetch_result.get("snippet"),
                            raw={"citation_marker": marker, "fetch_url_result": fetch_result},
                        )
                        if new_idx is not None:
                            marker_mapping[(result_type, result_index)] = new_idx
                        continue

                    search_result = search_results_by_id.get(result_index)
                    if not search_result:
                        continue
                    new_idx = append_citation(
                        url=search_result.get("url"),
                        title=search_result.get("title"),
                        source=search_result.get("source"),
                        snippet=search_result.get("snippet"),
                        raw={"citation_marker": marker, "search_result": search_result},
                    )
                    if new_idx is not None:
                        marker_mapping[(result_type, result_index)] = new_idx

        return citations, marker_mapping

    def _extract_citation_references(self, text: str) -> list[tuple[str, str | None, int]]:
        references: list[tuple[str, str | None, int]] = []
        if not text:
            return references

        for match in self._CITATION_BLOCK_RE.finditer(text):
            raw_content = match.group(1).strip()
            if not raw_content:
                continue
            parts = [part.strip() for part in raw_content.split(",")]
            for part in parts:
                token_match = self._CITATION_TOKEN_RE.fullmatch(part)
                if not token_match:
                    continue
                references.append(
                    (
                        match.group(0),
                        token_match.group("kind"),
                        int(token_match.group("index")),
                    )
                )
        return references

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
        reasoning_level: ReasoningLevel | None = None,
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

            merge_reasoning_payload(
                payload,
                build_perplexity_reasoning_payload(reasoning_level, target=target),
            )

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

            citations = []
            if return_citations:
                citations, marker_mapping = self._extract_citations(response_dict)
                if marker_mapping and text:
                    def rewrite_citations(match: re.Match) -> str:
                        raw_content = match.group(1).strip()
                        parts = [p.strip() for p in raw_content.split(",")]
                        new_parts = []
                        for p in parts:
                            token_match = self._CITATION_TOKEN_RE.fullmatch(p)
                            if token_match:
                                kind = token_match.group("kind")
                                idx = int(token_match.group("index"))
                                new_idx = marker_mapping.get((kind, idx))
                                if new_idx is not None:
                                    new_parts.append(str(new_idx))
                                else:
                                    new_parts.append(p)
                            else:
                                new_parts.append(p)
                        if not new_parts:
                            return ""
                        return f"[{', '.join(new_parts)}]"

                    text = self._CITATION_BLOCK_RE.sub(rewrite_citations, text)
            
            return AdapterResponse(text=text, citations=citations, raw=response_dict)

        except Exception as exc:  # pragma: no cover - network/provider behavior
            raise ProviderError(f"Perplexity adapter failed: {exc}") from exc
