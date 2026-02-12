"""Central logging adapter for prompt execution telemetry."""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4


# Global lock to ensure we only instrument once
_instrumentation_lock = threading.Lock()
_is_instrumented = False


def _is_django_configured() -> bool:
    try:
        from django.conf import settings as django_settings  # type: ignore
    except Exception:
        return False
    return bool(getattr(django_settings, "configured", False))


class PromptLogger:
    """Structured logger for `run_prompt` lifecycle events."""

    def __init__(self, logging_settings: dict[str, Any] | None) -> None:
        self.settings = logging_settings or {}
        self.enabled = bool(self.settings.get("enabled", False))
        self.network_logging = bool(self.settings.get("network_logging", False))
        self.logger: logging.Logger | None = None

        if not self.enabled:
            return

        self.logger = self._build_logger()

        if self.network_logging:
            _instrument_network_libs(self)

    def _build_logger(self) -> logging.Logger:
        if _is_django_configured():
            logger_name = str(self.settings.get("django_logfile") or "django")
            logger = logging.getLogger(logger_name)
            return logger

        logger = logging.getLogger("simpleai")
        logger.setLevel(logging.INFO)
        logger.propagate = False

        if not logger.handlers:
            logfile = Path(str(self.settings.get("logfile_location") or "./simpleai.log"))
            logfile.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(logfile)
            formatter = logging.Formatter("%(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _emit(self, payload: dict[str, Any]) -> None:
        if not self.enabled or self.logger is None:
            return

        payload.setdefault("ts", time.time())
        self.logger.info(json.dumps(payload, default=str, ensure_ascii=True, indent=2))

    def log_start(self, args: dict[str, Any], adapter_payload: dict[str, Any]) -> str:
        event_id = str(uuid4())
        self._emit(
            {
                "event": "run_prompt.start",
                "event_id": event_id,
                "args": args,
                "adapter_payload": adapter_payload,
            }
        )
        return event_id

    def log_end(
        self,
        event_id: str,
        started_at: float,
        result_preview: str,
        citations_count: int,
    ) -> None:
        ended_at = time.time()
        self._emit(
            {
                "event": "run_prompt.end",
                "event_id": event_id,
                "started_at": started_at,
                "ended_at": ended_at,
                "elapsed_seconds": ended_at - started_at,
                "result_preview": result_preview[:5000],
                "citations_count": citations_count,
            }
        )

    def log_error(
        self,
        event_id: str,
        started_at: float,
        error: Exception,
        context: dict[str, Any],
    ) -> None:
        ended_at = time.time()
        self._emit(
            {
                "event": "run_prompt.error",
                "event_id": event_id,
                "started_at": started_at,
                "ended_at": ended_at,
                "elapsed_seconds": ended_at - started_at,
                "error_type": type(error).__name__,
                "error": str(error),
                "context": context,
            }
        )


# --- Instrumentation Helpers ---


def _safe_header(key: str, value: Any) -> str:
    """Redact sensitive headers."""
    if key.lower() in ("authorization", "api-key", "x-api-key", "x-auth-token"):
        return "REDACTED"
    return str(value)


def _sanitize_headers(headers: Mapping) -> dict[str, str]:
    return {k: _safe_header(k, v) for k, v in headers.items()}


def _safe_body(body: Any, is_stream: bool = False) -> Any:
    if is_stream:
        return "<streaming_content>"
    if isinstance(body, bytes):
        try:
            return body.decode("utf-8")
        except Exception:
            return "<binary_content>"
    return body


def _log_httpx_exchange(logger: PromptLogger, request: Any, response: Any) -> None:
    try:
        # httpx request/response objects
        request_body = (
            _safe_body(request.content) if hasattr(request, "content") else None
        )

        # Check if response is stream
        is_stream = (
            getattr(response, "is_stream_consumed", False) is False
            and getattr(response, "is_closed", False) is False
        )
        # If headers indicate stream, logging body is risky
        if response.headers.get("content-type", "").startswith("text/event-stream"):
            is_stream = True

        response_body = _safe_body(getattr(response, "content", None), is_stream)

        logger._emit(
            {
                "event": "http_exchange",
                "library": "httpx",
                "url": str(request.url),
                "method": request.method,
                "status_code": response.status_code,
                "request_headers": _sanitize_headers(request.headers),
                "request_body": request_body,
                "response_headers": _sanitize_headers(response.headers),
                "response_body": response_body,
            }
        )
    except Exception:
        # Fail silently to avoid breaking the app integration
        pass


def _log_requests_exchange(logger: PromptLogger, response: Any) -> None:
    try:
        request = response.request

        request_body = _safe_body(request.body)

        # In requests, if stream=True in call, content might not be available
        # raw response usually handles this, but response.content accesses it.
        # We rely on checking if internal content has been consumed or flag
        response_body = "<streaming/binary>"
        if hasattr(response, "_content") and response._content is not None:
            response_body = _safe_body(response.content)
        elif hasattr(response, "_content_consumed") and not response._content_consumed:
            response_body = "<streaming_content>"

        logger._emit(
            {
                "event": "http_exchange",
                "library": "requests",
                "url": str(request.url),
                "method": request.method,
                "status_code": response.status_code,
                "request_headers": _sanitize_headers(request.headers),
                "request_body": request_body,
                "response_headers": _sanitize_headers(response.headers),
                "response_body": response_body,
            }
        )
    except Exception:
        pass


def _instrument_network_libs(logger_instance: PromptLogger) -> None:
    global _is_instrumented
    with _instrumentation_lock:
        if _is_instrumented:
            return

        # 1. Patch httpx (Used by OpenAI, Anthropic)
        try:
            import httpx

            _orig_httpx_send = httpx.Client.send
            _orig_httpx_asend = httpx.AsyncClient.send

            def _instrumented_httpx_send(self, request, *args, **kwargs):
                response = _orig_httpx_send(self, request, *args, **kwargs)
                _log_httpx_exchange(logger_instance, request, response)
                return response

            async def _instrumented_httpx_asend(self, request, *args, **kwargs):
                response = await _orig_httpx_asend(self, request, *args, **kwargs)
                _log_httpx_exchange(logger_instance, request, response)
                return response

            httpx.Client.send = _instrumented_httpx_send
            httpx.AsyncClient.send = _instrumented_httpx_asend
        except ImportError:
            pass

        # 2. Patch requests (Used by many other libs)
        try:
            import requests

            _orig_requests_request = requests.Session.request

            def _instrumented_requests_request(self, method, url, *args, **kwargs):
                response = _orig_requests_request(self, method, url, *args, **kwargs)
                _log_requests_exchange(logger_instance, response)
                return response

            requests.Session.request = _instrumented_requests_request
        except ImportError:
            pass

        _is_instrumented = True

