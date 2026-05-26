"""Google Gemini adapter using google-genai SDK."""

import logging
import mimetypes
import os
import sys
import uuid
import zipfile
from pathlib import Path
from typing import Any
from collections.abc import Sequence

# Vertex Gemini natively supports only specific mime types for binary parts.
# For others (like docx, pptx), we fall back to text extraction.
_GEMINI_SUPPORTED_MIME: dict[str, str] = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/plain",
    ".markdown": "text/plain",
    ".json": "application/json",
    ".html": "text/html",
    ".htm": "text/html",
    ".csv": "text/csv",
    ".xml": "application/xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".svg": "image/svg+xml",
}

from pydantic import BaseModel
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from simpleai.adapters.base import BaseAdapter
from simpleai.adapters.reasoning import ReasoningLevel, build_gemini_reasoning_config_kwargs
from simpleai.exceptions import FileExtractionError, ProviderError
from simpleai.files import extract_text_from_files
from simpleai.types import AdapterResponse, Citation, PromptInput

logger = logging.getLogger(__name__)


def _emit_gemini_file_event(message: str, level: int = logging.INFO) -> None:
    """Log and mirror to stderr so messages show in log files and the terminal."""
    logger.log(level, message)
    print(message, file=sys.stderr, flush=True)


def _sniff_gemini_media_mime(path: Path) -> str | None:
    """Infer MIME type from file contents for natively supported file parts."""
    try:
        header = path.read_bytes()[:4096]
    except OSError:
        return None
    if not header:
        return None
    if header.startswith(b"%PDF"):
        return "application/pdf"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if header[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if header.startswith(b"RIFF") and len(header) >= 12 and header[8:12] == b"WEBP":
        return "image/webp"
    return None


_GEMINI_SUPPORTED_PREFIXES = (
    "image/",
    "audio/",
    "video/",
    "text/",
)


def _gemini_media_mime_type(path: Path) -> str | None:
    """Return a Gemini-supported MIME type, or None for text fallback."""
    ext = path.suffix.lower()
    if ext in _GEMINI_SUPPORTED_MIME:
        return _GEMINI_SUPPORTED_MIME[ext]

    guessed, _ = mimetypes.guess_type(path.name)
    if guessed and (
        guessed.startswith(_GEMINI_SUPPORTED_PREFIXES) or guessed in {"application/pdf", "application/json"}
    ):
        return guessed

    sniffed = _sniff_gemini_media_mime(path)
    if sniffed and (
        sniffed.startswith(_GEMINI_SUPPORTED_PREFIXES) or sniffed in {"application/pdf", "application/json"}
    ):
        return sniffed

    return None


def _is_retryable_gemini_error(exc: BaseException) -> bool:
    """Check if the exception from Gemini is retryable (e.g., 503 or 429)."""
    exc_str = str(exc)
    return "503" in exc_str or "429" in exc_str or "UNAVAILABLE" in exc_str or "Too Many Requests" in exc_str


def _gemini_provider_setting(
    provider_settings: dict[str, Any],
    key: str,
    *,
    legacy_key: str,
    env_var: str,
    legacy_env_var: str | None = None,
) -> Any:
    """Read a Gemini enterprise setting; prefer new keys, then legacy vertexai_* names."""
    value = provider_settings.get(key)
    if value is None:
        value = provider_settings.get(legacy_key)
    if value is None:
        value = os.getenv(env_var)
    if value is None and legacy_env_var:
        value = os.getenv(legacy_env_var)
    return value


def _gemini_use_enterprise(provider_settings: dict[str, Any]) -> bool:
    raw = _gemini_provider_setting(
        provider_settings,
        "use_enterprise",
        legacy_key="use_vertexai",
        env_var="GEMINI_USE_ENTERPRISE",
        legacy_env_var="GEMINI_USE_VERTEXAI",
    )
    if raw is None:
        return False
    if isinstance(raw, bool):
        return raw
    return str(raw).lower() in ("true", "1", "yes")


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

        use_enterprise = _gemini_use_enterprise(provider_settings)

        if use_enterprise:
            project = _gemini_provider_setting(
                provider_settings,
                "enterprise_project",
                legacy_key="vertexai_project",
                env_var="GEMINI_ENTERPRISE_PROJECT",
                legacy_env_var="GEMINI_VERTEXAI_PROJECT",
            )
            location = _gemini_provider_setting(
                provider_settings,
                "enterprise_location",
                legacy_key="vertexai_location",
                env_var="GEMINI_ENTERPRISE_LOCATION",
                legacy_env_var="GEMINI_VERTEXAI_LOCATION",
            )
            self.client = genai.Client(enterprise=True, project=project, location=location)
            self._project = project
        else:
            api_key = provider_settings.get("api_key") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            self.client = genai.Client(api_key=api_key)

        self.types = types
        self._genai = genai
        self._use_enterprise = use_enterprise
        self._enterprise_gcs_bucket = _gemini_provider_setting(
            provider_settings,
            "enterprise_gcs_bucket",
            legacy_key="vertexai_gcs_bucket",
            env_var="GEMINI_ENTERPRISE_GCS_BUCKET",
            legacy_env_var="GEMINI_VERTEXAI_GCS_BUCKET",
        )
        self._enterprise_gcs_prefix = (
            str(
                _gemini_provider_setting(
                    provider_settings,
                    "enterprise_gcs_prefix",
                    legacy_key="vertexai_gcs_prefix",
                    env_var="GEMINI_ENTERPRISE_GCS_PREFIX",
                    legacy_env_var="GEMINI_VERTEXAI_GCS_PREFIX",
                )
                or "simpleai-uploads"
            )
        ).strip("/")
        self._enterprise_gcs_cleanup = (
            str(
                _gemini_provider_setting(
                    provider_settings,
                    "enterprise_gcs_cleanup",
                    legacy_key="vertexai_gcs_cleanup",
                    env_var="GEMINI_ENTERPRISE_GCS_CLEANUP",
                    legacy_env_var="GEMINI_VERTEXAI_GCS_CLEANUP",
                )
                or "always"
            )
            .strip()
            .lower()
        )
        if self._enterprise_gcs_cleanup not in {"always", "on_success", "never"}:
            raise ProviderError(
                "Invalid enterprise_gcs_cleanup value. Use one of: " "'always', 'on_success', 'never'."
            )
        self._storage_client: Any | None = None
        self._skip_citation_followup = bool(provider_settings.get("skip_citation_followup", False))
        # google-genai blocks Client.files.upload on the enterprise (GCP) path.
        # Enable true binary files only when a GCS bucket is configured.
        self.supports_binary_files = (not self._use_enterprise) or bool(self._enterprise_gcs_bucket)

    def _append_text_fallback_for_path(self, path: Path, contents: list[Any]) -> None:
        """Append one local file as extracted text (Vertex when GCS upload is skipped)."""
        try:
            extracted = extract_text_from_files([path])
            for item in extracted:
                contents.append(f"[File: {item.path.name}]\n{item.text}")
            _emit_gemini_file_event("Gemini adapter: text extraction fallback succeeded " f"path={path!s}")
        except FileExtractionError:
            _emit_gemini_file_event(
                "Gemini adapter: structured text extraction failed; " f"trying UTF-8 read path={path!s}",
                logging.WARNING,
            )
            try:
                raw = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                _emit_gemini_file_event(
                    "Gemini adapter: text fallback failed " f"path={path!s}: {exc}",
                    logging.ERROR,
                )
                raise ProviderError(f"Could not read file for text fallback: {path.name}") from exc
            contents.append(f"[File: {path.name}]\n{raw}")
            _emit_gemini_file_event("Gemini adapter: UTF-8 text fallback succeeded " f"path={path!s}")

    def _try_delete_gcs_object(self, object_name: str, *, reason: str) -> None:
        """Best-effort delete; avoids orphaned objects after failed or bad uploads."""
        if not self._enterprise_gcs_bucket:
            return
        try:
            storage_client = self._get_storage_client()
            bucket = storage_client.bucket(self._enterprise_gcs_bucket)
            blob = bucket.blob(object_name)
            if not blob.exists():
                return
            blob.delete()
            _emit_gemini_file_event(
                "Gemini adapter: best-effort GCS delete succeeded " f"({reason}) object={object_name!r}"
            )
        except Exception as exc:
            _emit_gemini_file_event(
                "Gemini adapter: best-effort GCS delete failed " f"({reason}) object={object_name!r}: {exc}",
                logging.WARNING,
            )

    def _get_storage_client(self) -> Any:
        if self._storage_client is not None:
            return self._storage_client
        try:
            from google.cloud import storage
        except Exception as exc:
            raise ProviderError(
                "google-cloud-storage is required for Vertex GCS uploads. "
                "Install it and configure enterprise_gcs_bucket."
            ) from exc
        self._storage_client = storage.Client(project=self._project)
        return self._storage_client

    def _upload_file_to_gcs(self, path: Path, *, mime_type: str) -> tuple[str, str]:
        if not self._enterprise_gcs_bucket:
            raise ProviderError("Missing enterprise_gcs_bucket for enterprise (GCP) file uploads.")
        try:
            local_size = path.stat().st_size
        except OSError as exc:
            raise ProviderError(f"Cannot stat file for GCS upload: {path}") from exc
        if local_size == 0:
            raise ProviderError("GCS upload must not be called for 0-byte files; use text fallback.")

        object_name = f"{self._enterprise_gcs_prefix}/{uuid.uuid4().hex}-{path.name}"
        _emit_gemini_file_event(
            "Gemini adapter: starting GCS upload "
            f"path={path!s} mime_type={mime_type} local_bytes={local_size} "
            f"bucket={self._enterprise_gcs_bucket!r} object={object_name!r}"
        )
        storage_client = self._get_storage_client()
        bucket = storage_client.bucket(self._enterprise_gcs_bucket)
        blob = bucket.blob(object_name)
        try:
            blob.upload_from_filename(str(path), content_type=mime_type)
        except Exception as exc:
            _emit_gemini_file_event(
                f"Gemini adapter: GCS upload failed path={path!s}: {exc}",
                logging.ERROR,
            )
            self._try_delete_gcs_object(
                object_name,
                reason=("upload failure (may leave 0-byte or partial object without " "cleanup)"),
            )
            raise
        try:
            blob.reload()
        except Exception as exc:
            _emit_gemini_file_event(
                f"Gemini adapter: GCS blob.reload() after upload failed "
                f"path={path!s} object={object_name!r}: {exc}",
                logging.ERROR,
            )
            self._try_delete_gcs_object(object_name, reason="reload failed after upload")
            raise ProviderError(f"Could not verify GCS upload for {path.name}") from exc

        remote_size = blob.size
        if remote_size is None:
            _emit_gemini_file_event(
                f"Gemini adapter: GCS blob has no size after upload " f"object={object_name!r}",
                logging.ERROR,
            )
            self._try_delete_gcs_object(object_name, reason="missing size after upload")
            raise ProviderError(f"GCS upload verification failed for {path.name}")

        if remote_size != local_size:
            _emit_gemini_file_event(
                "Gemini adapter: GCS size mismatch after upload "
                f"local_bytes={local_size} remote_bytes={remote_size} "
                f"object={object_name!r}. "
                "If remote is 0, common causes: interrupted upload, resumable "
                "upload abort leaving an empty object, or storage race.",
                logging.ERROR,
            )
            self._try_delete_gcs_object(object_name, reason="size mismatch after upload")
            raise ProviderError(
                f"GCS upload size mismatch for {path.name}: " f"local={local_size} remote={remote_size}"
            )

        uri = f"gs://{self._enterprise_gcs_bucket}/{object_name}"
        _emit_gemini_file_event(f"Gemini adapter: GCS upload verified uri={uri!s} " f"bytes={remote_size}")
        return uri, object_name

    def _cleanup_gcs_uploads(
        self,
        uploaded_objects: Sequence[str],
        *,
        generation_succeeded: bool,
    ) -> None:
        if not uploaded_objects:
            return
        if self._enterprise_gcs_cleanup == "never":
            return
        if self._enterprise_gcs_cleanup == "on_success" and not generation_succeeded:
            return
        try:
            storage_client = self._get_storage_client()
            bucket = storage_client.bucket(self._enterprise_gcs_bucket)
            for object_name in uploaded_objects:
                blob = bucket.blob(object_name)
                try:
                    if not blob.exists():
                        _emit_gemini_file_event(
                            "Gemini adapter: GCS cleanup object already absent "
                            f"object={object_name!r} bucket={self._enterprise_gcs_bucket!r}",
                            logging.WARNING,
                        )
                        continue
                    blob.reload()
                    size_before = blob.size
                    if size_before == 0:
                        _emit_gemini_file_event(
                            "Gemini adapter: GCS cleanup removing 0-byte object "
                            f"object={object_name!r} bucket={self._enterprise_gcs_bucket!r}. "
                            "Possible causes: empty local file was uploaded before "
                            "skip logic, interrupted upload, failed upload leaving a "
                            "placeholder, delete of a prior version failed, or "
                            "enterprise_gcs_cleanup prevented earlier removal.",
                            logging.WARNING,
                        )
                    blob.delete()
                    _emit_gemini_file_event(
                        "Gemini adapter: GCS cleanup deleted object "
                        f"object={object_name!r} bytes_before_delete={size_before}"
                    )
                except Exception as exc:
                    _emit_gemini_file_event(
                        "Gemini adapter: GCS cleanup failed for " f"object={object_name!r}: {exc}",
                        logging.ERROR,
                    )
        except Exception:
            logger.warning(
                "Failed cleaning up %d GCS upload(s) in bucket %s.",
                len(uploaded_objects),
                self._enterprise_gcs_bucket,
                exc_info=True,
            )

    def _build_contents(
        self,
        prompt: PromptInput,
        files: Sequence[Path] | None,
        client: Any = None,
    ) -> tuple[Any, list[str]]:
        client = client or self.client
        contents: list[Any] = []
        uploaded_objects: list[str] = []

        if files:
            if self._use_enterprise:
                if self._enterprise_gcs_bucket:
                    for path in files:
                        try:
                            local_bytes = path.stat().st_size
                        except OSError as exc:
                            _emit_gemini_file_event(
                                f"Gemini adapter: cannot stat file path={path!s}: {exc}",
                                logging.ERROR,
                            )
                            raise ProviderError(f"Cannot read file for Vertex upload: {path}") from exc

                        if local_bytes == 0:
                            _emit_gemini_file_event(
                                "Gemini adapter: local file is 0 bytes; skipping "
                                "GCS upload (would create a 0-byte object) "
                                f"path={path!s}. Using text extraction instead.",
                                logging.WARNING,
                            )
                            self._append_text_fallback_for_path(
                                path,
                                contents,
                            )
                            continue

                        mime_type = _gemini_media_mime_type(path)
                        if mime_type is None:
                            _emit_gemini_file_event(
                                "Gemini adapter: Vertex MIME type unknown; "
                                f"falling back to text extraction path={path!s}",
                                logging.WARNING,
                            )
                            self._append_text_fallback_for_path(
                                path,
                                contents,
                            )
                            continue
                        uri, object_name = self._upload_file_to_gcs(
                            path,
                            mime_type=mime_type,
                        )
                        uploaded_objects.append(object_name)
                        contents.append(
                            self.types.Part.from_uri(
                                file_uri=uri,
                                mime_type=mime_type,
                            )
                        )
                else:
                    _emit_gemini_file_event(
                        "Gemini adapter: enterprise mode has no enterprise_gcs_bucket; "
                        f"using text extraction for {len(files)} file(s)"
                    )
                    extracted = extract_text_from_files(files)
                    for item in extracted:
                        contents.append(f"[File: {item.path.name}]\n{item.text}")
                    _emit_gemini_file_event(
                        "Gemini adapter: text extraction (no GCS bucket) " f"succeeded for {len(files)} file(s)"
                    )
            else:

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
                    try:
                        local_bytes = path.stat().st_size
                    except OSError as exc:
                        _emit_gemini_file_event(
                            f"Gemini adapter: cannot stat file path={path!s}: {exc}",
                            logging.ERROR,
                        )
                        raise ProviderError(f"Cannot read file for Gemini upload: {path}") from exc

                    if local_bytes == 0:
                        _emit_gemini_file_event(
                            "Gemini adapter: local file is 0 bytes; skipping "
                            "Gemini Developer API file upload "
                            f"path={path!s}. Using text extraction instead.",
                            logging.WARNING,
                        )
                        self._append_text_fallback_for_path(
                            path,
                            contents,
                        )
                        continue

                    mime_type = _gemini_media_mime_type(path)
                    if mime_type is None:
                        _emit_gemini_file_event(
                            "Gemini adapter: Gemini Developer API MIME type unsupported; "
                            f"falling back to text extraction path={path!s}",
                            logging.WARNING,
                        )
                        self._append_text_fallback_for_path(
                            path,
                            contents,
                        )
                        continue

                    _emit_gemini_file_event(
                        "Gemini adapter: starting Gemini Developer API " f"file upload path={path!s}"
                    )
                    try:
                        uploaded = _upload(path)
                    except Exception as exc:
                        _emit_gemini_file_event(
                            "Gemini adapter: Gemini Developer API file upload " f"failed path={path!s}: {exc}",
                            logging.ERROR,
                        )
                        raise
                    _emit_gemini_file_event(
                        "Gemini adapter: Gemini Developer API file upload " f"succeeded path={path!s}"
                    )
                    contents.append(uploaded)

        if isinstance(prompt, str):
            contents.append(prompt)
        else:
            contents.extend(str(item) for item in prompt)

        if len(contents) == 1:
            return contents[0], uploaded_objects
        return contents, uploaded_objects

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
                        title=retrieved.get("title")
                        or retrieved.get("document_name")
                        or retrieved.get("documentName"),
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

    _SEARCH_INSTRUCTION = (
        "You are an expert researcher. You must ALWAYS use the Google Search tool to ground your answer, "
        "even if you think you already know the answer. Ensure that all cited URLs are publicly accessible. "
        "Do not cite links that result in a 404 or 5xx error. You must provide a robust, comprehensive list "
        "of citations for all factual claims. When possible, include inline citation markers (e.g. [1]) in the "
        "text that map to the sources you used."
    )

    def _collect_citations_followup(
        self,
        *,
        client: Any,
        model: str,
        text: str,
        adapter_options: dict[str, Any] | None,
        reasoning_level: ReasoningLevel | None,
    ) -> list[Citation]:
        preview = text.strip()[:4000]
        if not preview:
            return []

        followup_prompt = (
            "Use Google Search to find citations supporting this structured answer. "
            "Ensure that all cited URLs are publicly accessible. Prefer official sources and include "
            "company homepages when relevant.\n\n"
            f"Structured answer:\n{preview}"
        )
        config_kwargs: dict[str, Any] = {
            "tools": [self.types.Tool(google_search=self.types.GoogleSearch())],
            "system_instruction": self._SEARCH_INSTRUCTION,
        }
        reasoning_kwargs = build_gemini_reasoning_config_kwargs(reasoning_level, model=model)
        thinking_config = reasoning_kwargs.get("thinking_config")
        if thinking_config is not None:
            config_kwargs["thinking_config"] = self.types.ThinkingConfig(**thinking_config)
        if adapter_options:
            passthrough = {
                key: value
                for key, value in adapter_options.items()
                if key not in {"response_mime_type", "response_schema"}
            }
            config_kwargs.update(passthrough)

        config = self.types.GenerateContentConfig(**config_kwargs)
        response = client.models.generate_content(
            model=model,
            contents=followup_prompt,
            config=config,
        )
        response_dict = response.model_dump(mode="json") if hasattr(response, "model_dump") else {}
        return self._extract_citations(response_dict)

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
        generation_succeeded = False
        uploaded_objects: list[str] = []
        try:
            client = self.client
            if getattr(self, "_use_enterprise", False) and model.startswith("gemini-3.1"):
                client = self._genai.Client(enterprise=True, project=self._project, location="global")

            default_max_tokens = 65536 if ("gemini-3.5" in model or "gemini-3.1" in model) else 8192
            config_kwargs: dict[str, Any] = {
                "max_output_tokens": int(self.provider_settings.get("max_output_tokens", default_max_tokens)),
            }

            if require_search:
                config_kwargs["tools"] = [self.types.Tool(google_search=self.types.GoogleSearch())]
                config_kwargs.setdefault("system_instruction", self._SEARCH_INSTRUCTION)

            if output_format is not None:
                config_kwargs["response_mime_type"] = "application/json"
                config_kwargs["response_schema"] = output_format.model_json_schema()

            reasoning_kwargs = build_gemini_reasoning_config_kwargs(reasoning_level, model=model)
            if reasoning_kwargs:
                thinking_config = reasoning_kwargs.get("thinking_config")
                if thinking_config is not None:
                    config_kwargs["thinking_config"] = self.types.ThinkingConfig(**thinking_config)

            if adapter_options:
                config_kwargs.update(adapter_options)

            config = self.types.GenerateContentConfig(**config_kwargs)
            contents, uploaded_objects = self._build_contents(
                prompt,
                files,
                client=client,
            )

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
                        raise ProviderError(
                            f"Gemini hit MAX_TOKENS before finishing the JSON response. Try increasing max_output_tokens (currently {max_tokens} for model {model})."
                        )
                    else:
                        logger.warning("Gemini hit MAX_TOKENS. The response may be incomplete.")

            citations = self._extract_citations(response_dict) if return_citations else []
            if (
                return_citations
                and require_search
                and not citations
                and not self._skip_citation_followup
            ):
                citations = self._collect_citations_followup(
                    client=client,
                    model=model,
                    text=text,
                    adapter_options=adapter_options,
                    reasoning_level=reasoning_level,
                )
            generation_succeeded = True
            return AdapterResponse(text=text, citations=citations, raw=response_dict)

        except Exception as exc:  # pragma: no cover - network/provider behavior
            raise ProviderError(f"Gemini adapter failed: {exc}") from exc
        finally:
            if self._use_enterprise:
                self._cleanup_gcs_uploads(
                    uploaded_objects,
                    generation_succeeded=generation_succeeded,
                )
