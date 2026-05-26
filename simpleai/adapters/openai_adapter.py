"""OpenAI adapter using the Responses API."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel

from simpleai.adapters.base import BaseAdapter
from simpleai.adapters.reasoning import ReasoningLevel, build_openai_reasoning_payload, merge_reasoning_payload
from simpleai.exceptions import ProviderError
from simpleai.schema import openai_response_schema
from simpleai.types import AdapterResponse, Citation, PromptInput


def _emit_openai_file_event(message: str) -> None:
    """Mirror OpenAI file upload events to console."""
    print(message, file=sys.stderr, flush=True)


class OpenAIAdapter(BaseAdapter):
    provider_name = "openai"
    supports_binary_files = True

    def __init__(self, provider_settings: dict[str, Any]) -> None:
        super().__init__(provider_settings)

        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency missing path
            raise ProviderError("openai package is required for OpenAIAdapter.") from exc

        api_key = provider_settings.get("api_key") or os.getenv("OPENAI_API_KEY")
        base_url = provider_settings.get("base_url")

        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url

        self.client = OpenAI(**kwargs)

    def _build_input(self, prompt: PromptInput, file_ids: Sequence[str]) -> list[dict[str, Any]]:
        if isinstance(prompt, str):
            messages: list[dict[str, Any]] = [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": str(turn)}],
                }
                for turn in prompt
            ]

        if not messages:
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": ""}],
                }
            ]

        if file_ids:
            content = messages[-1]["content"]
            for file_id in file_ids:
                content.append({"type": "input_file", "file_id": file_id})

        return messages

    def _extract_citations(self, response_dict: dict[str, Any]) -> list[Citation]:
        citations: list[Citation] = []
        seen: set[tuple[Any, ...]] = set()

        def append_citation(
            *,
            url: str | None,
            title: str | None,
            source: str | None,
            start_index: int | None,
            end_index: int | None,
            raw: dict[str, Any],
        ) -> None:
            key = (url, title, source, start_index, end_index)
            if key in seen:
                return
            seen.add(key)
            citations.append(
                Citation(
                    provider=self.provider_name,
                    url=url,
                    title=title,
                    source=source,
                    start_index=start_index,
                    end_index=end_index,
                    raw=raw,
                )
            )

        # 1) Message text annotations (inline citations in generated text).
        for output in response_dict.get("output", []):
            if output.get("type") != "message":
                continue

            for part in output.get("content", []):
                for annotation in part.get("annotations") or []:
                    url_citation = annotation.get("url_citation") or {}
                    url = annotation.get("url") or url_citation.get("url")
                    title = annotation.get("title") or url_citation.get("title")
                    start_index = annotation.get("start_index")
                    if start_index is None:
                        start_index = url_citation.get("start_index")
                    end_index = annotation.get("end_index")
                    if end_index is None:
                        end_index = url_citation.get("end_index")

                    append_citation(
                        url=url,
                        title=title,
                        source=url,
                        start_index=start_index,
                        end_index=end_index,
                        raw=annotation,
                    )

        return citations

    _SEARCH_INSTRUCTION = (
        "You are an expert researcher. You must ALWAYS use the web_search tool to ground your answer, "
        "even if you think you already know the answer. Ensure that all cited URLs are publicly accessible. "
        "Do not cite links that result in a 404 or 5xx error. Cite every factual claim with provider-native "
        "inline URL citations. If a claim cannot be cited, omit it."
    )

    def _collect_citations_followup(
        self,
        *,
        model: str,
        prompt: PromptInput,
        text: str,
        adapter_options: dict[str, Any] | None,
        reasoning_level: ReasoningLevel | None,
        structured_output: bool,
    ) -> tuple[str, list[Citation], dict[str, Any]]:
        preview = text.strip()[:4000]
        if structured_output:
            followup_prompt = (
                "Use web search to verify this structured answer. Return a concise cited explanation "
                "of the facts in the structured answer. Every factual claim must have an inline URL "
                "citation. If you cannot cite a claim, omit it.\n\n"
                f"Structured answer:\n{preview}"
            )
        else:
            original_prompt = prompt if isinstance(prompt, str) else "\n\n".join(str(item) for item in prompt)
            followup_prompt = (
                "Answer the original request again using web search. Every factual claim must have "
                "an inline URL citation. If you cannot cite a claim, omit it.\n\n"
                f"Original request:\n{original_prompt}\n\nPrevious uncited answer:\n{preview}"
            )

        payload: dict[str, Any] = {
            "model": model,
            "input": self._build_input(followup_prompt, []),
            "tools": [{"type": "web_search"}],
            "tool_choice": "required",
            "instructions": self._SEARCH_INSTRUCTION,
            "include": ["web_search_call.action.sources"],
        }
        merge_reasoning_payload(payload, build_openai_reasoning_payload(reasoning_level, model=model))
        if adapter_options:
            passthrough = {
                key: value
                for key, value in adapter_options.items()
                if key not in {"include", "input", "instructions", "text", "tool_choice", "tools"}
            }
            payload.update(passthrough)

        response = self.client.responses.create(**payload)
        response_dict = response.model_dump(mode="json") if hasattr(response, "model_dump") else {}
        followup_text = getattr(response, "output_text", "")
        if not followup_text and response_dict:
            chunks: list[str] = []
            for output in response_dict.get("output", []):
                if output.get("type") != "message":
                    continue
                for part in output.get("content", []):
                    if part.get("type") == "output_text":
                        chunks.append(part.get("text", ""))
            followup_text = "".join(chunks)
        return followup_text, self._extract_citations(response_dict), response_dict

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
        file_ids: list[str] = []
        try:
            if files:
                for path in files:
                    _emit_openai_file_event(
                        f"OpenAI adapter: starting binary file upload path={path!s}"
                    )
                    with path.open("rb") as handle:
                        uploaded = self.client.files.create(file=handle, purpose="user_data")
                    _emit_openai_file_event(
                        f"OpenAI adapter: binary file upload succeeded path={path!s}"
                    )
                    file_ids.append(uploaded.id)

            payload: dict[str, Any] = {
                "model": model,
                "input": self._build_input(prompt, file_ids),
            }

            if require_search:
                payload["tools"] = [{"type": "web_search"}]
                payload["tool_choice"] = "required"
                payload.setdefault("instructions", self._SEARCH_INSTRUCTION)
                if return_citations:
                    payload["include"] = ["web_search_call.action.sources"]

            if output_format is not None:
                payload["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": "simpleai_output",
                        "schema": openai_response_schema(output_format),
                        "strict": True,
                    }
                }

            merge_reasoning_payload(payload, build_openai_reasoning_payload(reasoning_level, model=model))

            if adapter_options:
                payload.update(adapter_options)

            response = self.client.responses.create(**payload)
            response_dict = response.model_dump(mode="json") if hasattr(response, "model_dump") else {}
            text = getattr(response, "output_text", "")
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
            if return_citations and require_search and not citations:
                followup_text, followup_citations, followup_raw = self._collect_citations_followup(
                    model=model,
                    prompt=prompt,
                    text=text,
                    adapter_options=adapter_options,
                    reasoning_level=reasoning_level,
                    structured_output=output_format is not None,
                )
                if followup_citations:
                    citations = followup_citations
                    response_dict = {**response_dict, "citation_followup": followup_raw}
                    if output_format is None and followup_text:
                        text = followup_text

            if return_citations and require_search and not citations:
                raise ProviderError(
                    "OpenAI did not return provider-anchored URL citations after a citation follow-up pass."
                )

            return AdapterResponse(text=text, citations=citations, raw=response_dict)

        except ProviderError:
            raise
        except Exception as exc:  # pragma: no cover - network/provider behavior
            msg = f"OpenAI adapter failed: {exc}"

            # OpenAI's python SDK stores response headers on `exc.response.headers` for API errors.
            headers = getattr(exc, "headers", None)
            response = getattr(exc, "response", None)
            if not headers and response is not None:
                headers = getattr(response, "headers", None)

            if headers:
                # Helpful identifiers
                id_headers = [
                    "x-request-id",
                    "openai-request-id",
                    "cf-ray",
                ]
                id_details = [
                    f"{key}: {headers.get(key)}" for key in id_headers if headers.get(key) is not None
                ]
                if id_details:
                    msg += "\n\nRequest identifiers:\n" + "\n".join(id_details)

                # Rate limiting (docs: https://platform.openai.com/docs/guides/error-codes/api-errors)
                relevant_headers = [
                    "x-ratelimit-limit-requests",
                    "x-ratelimit-limit-tokens",
                    "x-ratelimit-remaining-requests",
                    "x-ratelimit-remaining-tokens",
                    "x-ratelimit-reset-requests",
                    "x-ratelimit-reset-tokens",
                ]
                rate_details = [
                    f"{key}: {headers.get(key)}"
                    for key in relevant_headers
                    if headers.get(key) is not None
                ]
                if rate_details:
                    msg += "\n\nRate limit headers:\n" + "\n".join(rate_details)
                else:
                    # Many 429s with `insufficient_quota` are *billing/quota* issues, not rate limit issues,
                    # and may not include rate limit headers.
                    msg += "\n\nRate limit headers: (not present in provider response)"

            raise ProviderError(msg) from exc
        finally:
            for file_id in file_ids:
                try:
                    self.client.files.delete(file_id)
                except Exception:
                    pass  # Best-effort cleanup
