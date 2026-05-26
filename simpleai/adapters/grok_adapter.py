"""xAI Grok adapter using xai-sdk chat API."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel

from simpleai.adapters.base import BaseAdapter
from simpleai.adapters.reasoning import ReasoningLevel, build_grok_reasoning_payload, merge_reasoning_payload
from simpleai.exceptions import ProviderError
from simpleai.types import AdapterResponse, Citation, PromptInput


class GrokAdapter(BaseAdapter):
    provider_name = "grok"
    supports_binary_files = True

    _SEARCH_INSTRUCTION = (
        "You are an expert researcher. You must ALWAYS use the web_search tool to ground your answer, "
        "even if you think you already know the answer. Ensure that all cited URLs are publicly accessible. "
        "Do not cite links that result in a 404 or 5xx error. Cite every factual claim with provider-native "
        "inline citations. If a claim cannot be cited, omit it."
    )

    def __init__(self, provider_settings: dict[str, Any]) -> None:
        super().__init__(provider_settings)

        try:
            from xai_sdk import Client
            from xai_sdk import chat as chat_helpers
            from xai_sdk import tools as xai_tools
        except Exception as exc:  # pragma: no cover - dependency missing path
            raise ProviderError("xai-sdk package is required for GrokAdapter.") from exc

        api_key = provider_settings.get("api_key") or os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")
        if not api_key or not str(api_key).strip():
            raise ProviderError("XAI_API_KEY or GROK_API_KEY is required for GrokAdapter")

        self.client = Client(api_key=api_key)
        self.chat_helpers = chat_helpers
        self.xai_tools = xai_tools

    def _build_messages(
        self,
        prompt: PromptInput,
        files: Sequence[Path] | None,
        require_search: bool,
    ) -> list[Any]:
        attachments: list[Any] = []
        if files:
            for path in files:
                uploaded = self.client.files.upload(str(path))
                attachments.append(self.chat_helpers.file(uploaded.id))

        messages: list[Any] = []
        if require_search:
            messages.append(
                self.chat_helpers.system(self._SEARCH_INSTRUCTION)
            )

        if isinstance(prompt, str):
            contents = [prompt]
            contents.extend(attachments)
            messages.append(self.chat_helpers.user(*contents))
            return messages

        for idx, turn in enumerate(prompt):
            if idx == len(prompt) - 1:
                contents = [str(turn)]
                contents.extend(attachments)
                messages.append(self.chat_helpers.user(*contents))
            else:
                messages.append(self.chat_helpers.user(str(turn)))

        if not messages:
            base = [""]
            base.extend(attachments)
            messages.append(self.chat_helpers.user(*base))

        return messages

    def _citation_key(self, item: Citation) -> tuple[Any, ...]:
        return (
            item.url,
            item.title,
            item.source,
            item.citation_id,
            item.start_index,
            item.end_index,
        )

    def _append_citation(
        self,
        citations: list[Citation],
        seen: set[tuple[Any, ...]],
        *,
        url: str | None,
        title: str | None,
        source: str | None,
        citation_id: str | None = None,
        start_index: int | None = None,
        end_index: int | None = None,
        raw: dict[str, Any],
    ) -> None:
        item = Citation(
            provider=self.provider_name,
            citation_id=citation_id,
            url=url,
            title=title,
            source=source,
            start_index=start_index,
            end_index=end_index,
            raw=raw,
        )
        key = self._citation_key(item)
        if key in seen:
            return
        seen.add(key)
        citations.append(item)

    def _extract_citations_from_dict(
        self,
        payload: dict[str, Any],
        citations: list[Citation],
        seen: set[tuple[Any, ...]],
    ) -> None:
        for output in payload.get("output", []):
            if not isinstance(output, dict):
                continue
            if output.get("type") == "message":
                for content in output.get("content", []):
                    if not isinstance(content, dict):
                        continue
                    for annotation in content.get("annotations") or []:
                        if not isinstance(annotation, dict):
                            continue
                        url = annotation.get("url")
                        title = annotation.get("title")
                        self._append_citation(
                            citations,
                            seen,
                            url=url,
                            title=title,
                            source=url,
                            start_index=annotation.get("start_index"),
                            end_index=annotation.get("end_index"),
                            raw=annotation,
                        )

    def _extract_citations(self, response: Any) -> list[Citation]:
        citations: list[Citation] = []
        seen: set[tuple[Any, ...]] = set()

        # Inline citations contain structured metadata and positions.
        for inline in getattr(response, "inline_citations", []) or []:
            url = None
            title = getattr(inline, "title", None)
            source = None
            raw: dict[str, Any] = {
                "id": getattr(inline, "id", None),
                "start_index": getattr(inline, "start_index", None),
                "end_index": getattr(inline, "end_index", None),
                "title": title,
            }

            if hasattr(inline, "HasField") and inline.HasField("web_citation"):
                url = inline.web_citation.url
                source = url
            elif hasattr(inline, "HasField") and inline.HasField("x_citation"):
                url = inline.x_citation.url
                source = "x"
            elif hasattr(inline, "HasField") and inline.HasField("collections_citation"):
                source = "collections"
                raw["collections"] = {
                    "file_id": inline.collections_citation.file_id,
                    "chunk_id": inline.collections_citation.chunk_id,
                    "score": inline.collections_citation.score,
                }

            self._append_citation(
                citations,
                seen,
                url=url,
                title=title,
                source=source,
                citation_id=str(getattr(inline, "id", "")) or None,
                start_index=getattr(inline, "start_index", None),
                end_index=getattr(inline, "end_index", None),
                raw=raw,
            )

        raw_payload = self._raw_response(response)
        proto = raw_payload.get("proto")
        if isinstance(proto, dict):
            self._extract_citations_from_dict(proto, citations, seen)

        return citations

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
                "of the facts in the structured answer. Every factual claim must have an inline citation. "
                "If you cannot cite a claim, omit it.\n\n"
                f"Structured answer:\n{preview}"
            )
        else:
            original_prompt = prompt if isinstance(prompt, str) else "\n\n".join(str(item) for item in prompt)
            followup_prompt = (
                "Answer the original request again using web search. Every factual claim must have "
                "an inline citation. If you cannot cite a claim, omit it.\n\n"
                f"Original request:\n{original_prompt}\n\nPrevious uncited answer:\n{preview}"
            )

        create_kwargs: dict[str, Any] = {
            "model": model,
            "messages": self._build_messages(followup_prompt, None, require_search=True),
            "max_tokens": int(self.provider_settings.get("max_tokens", 8192)),
            "tools": [self.xai_tools.web_search()],
            "tool_choice": "required",
            "max_turns": int(self.provider_settings.get("max_turns", 12)),
            "include": ["inline_citations", "web_search_call_output"],
        }
        merge_reasoning_payload(create_kwargs, build_grok_reasoning_payload(reasoning_level, model=model))
        if adapter_options:
            passthrough = {
                key: value
                for key, value in adapter_options.items()
                if key not in {"include", "messages", "response_format", "tool_choice", "tools"}
            }
            create_kwargs.update(passthrough)

        chat = self.client.chat.create(**create_kwargs)
        response = chat.sample()
        followup_text = getattr(response, "content", "")
        return followup_text, self._extract_citations(response), self._raw_response(response)

    def _raw_response(self, response: Any) -> dict[str, Any]:
        raw: dict[str, Any] = {
            "id": getattr(response, "id", None),
            "content": getattr(response, "content", ""),
            "citations": list(getattr(response, "citations", []) or []),
        }
        try:
            from google.protobuf.json_format import MessageToDict

            raw["proto"] = MessageToDict(response.proto, preserving_proto_field_name=True)
        except Exception:
            pass
        return raw

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
        try:
            create_kwargs: dict[str, Any] = {
                "model": model,
                "messages": self._build_messages(prompt, files, require_search=require_search),
                "max_tokens": int(self.provider_settings.get("max_tokens", 8192)),
            }

            if require_search:
                create_kwargs["tools"] = [self.xai_tools.web_search()]
                create_kwargs["tool_choice"] = "required"
                create_kwargs["max_turns"] = int(self.provider_settings.get("max_turns", 12))

            if return_citations:
                include = ["inline_citations"]
                if require_search:
                    include.append("web_search_call_output")
                create_kwargs["include"] = include

            if output_format is not None:
                create_kwargs["response_format"] = output_format

            merge_reasoning_payload(create_kwargs, build_grok_reasoning_payload(reasoning_level, model=model))

            if adapter_options:
                create_kwargs.update(adapter_options)

            chat = self.client.chat.create(**create_kwargs)
            response = chat.sample()
            text = getattr(response, "content", "")
            citations = self._extract_citations(response) if return_citations else []
            raw = self._raw_response(response)

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
                    raw["citation_followup"] = followup_raw
                    if output_format is None and followup_text:
                        text = followup_text

            if return_citations and require_search and not citations:
                raise ProviderError(
                    "Grok did not return provider-anchored inline citations after a citation follow-up pass. "
                    "Bare response.citations URLs were treated as candidate sources, not citations."
                )

            return AdapterResponse(
                text=text,
                citations=citations,
                raw=raw,
            )

        except ProviderError:
            raise
        except Exception as exc:  # pragma: no cover - network/provider behavior
            raise ProviderError(f"Grok adapter failed: {exc}") from exc
