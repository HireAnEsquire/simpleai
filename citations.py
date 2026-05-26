"""Citation normalization and optional metadata enrichment."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from html.parser import HTMLParser

from .types import Citation

_UNKNOWN_PUBLICATION = "(unknown publication)"
_UNKNOWN_TITLE = "(unknown title)"
_UNKNOWN_URL = "(no url)"

_TRACKING_PARAM_PREFIXES = (
    "utm_",
    "mc_",
)
_TRACKING_PARAM_KEYS = {
    "fbclid",
    "gclid",
    "yclid",
    "msclkid",
    "ref",
    "ref_src",
    "ref_url",
    "source",
    "cmpid",
}

_GENERIC_SOURCE_VALUES = {
    "",
    "web",
    "source",
    "result",
    "search",
    "google_search_query",
    "google_maps",
    "x",
    "collections",
}

_MULTIPART_PUBLIC_SUFFIXES = {
    "co.uk",
    "org.uk",
    "gov.uk",
    "ac.uk",
    "com.au",
    "net.au",
    "org.au",
    "co.jp",
    "com.br",
}

_SOURCE_ALIAS_BY_DOMAIN = {
    "nytimes.com": "The New York Times",
    "wsj.com": "The Wall Street Journal",
    "ft.com": "Financial Times",
    "bbc.com": "BBC",
    "reuters.com": "Reuters",
    "apnews.com": "Associated Press",
    "bloomberg.com": "Bloomberg",
}


class _MetaParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_title = False
        self.title_parts: list[str] = []
        self.meta: dict[str, str] = {}
        self.ld_json_blocks: list[str] = []
        self._script_is_json_ld = False
        self._script_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {k.lower(): (v or "") for k, v in attrs}
        tag_l = tag.lower()
        if tag_l == "title":
            self.in_title = True
            return

        if tag_l == "meta":
            key = (attrs_dict.get("property") or attrs_dict.get("name") or "").strip().lower()
            value = (attrs_dict.get("content") or "").strip()
            if key and value:
                self.meta[key] = value
            return

        if tag_l == "script":
            script_type = (attrs_dict.get("type") or "").strip().lower()
            if script_type == "application/ld+json":
                self._script_is_json_ld = True
                self._script_parts = []

    def handle_endtag(self, tag: str) -> None:
        tag_l = tag.lower()
        if tag_l == "title":
            self.in_title = False
            return
        if tag_l == "script" and self._script_is_json_ld:
            payload = "".join(self._script_parts).strip()
            if payload:
                self.ld_json_blocks.append(payload)
            self._script_is_json_ld = False
            self._script_parts = []

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title_parts.append(data)
        if self._script_is_json_ld:
            self._script_parts.append(data)


def _clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = " ".join(value.split())
    return cleaned or None


def _looks_like_url(value: str | None) -> bool:
    cleaned = _clean_text(value)
    if not cleaned:
        return False
    lowered = cleaned.lower()
    if lowered.startswith(("http://", "https://", "www.")):
        return True
    parsed = urllib.parse.urlparse(cleaned)
    return bool(parsed.scheme and parsed.netloc)


def _extract_hostname(url: str | None) -> str | None:
    if not url:
        return None
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return None
    hostname = (parsed.hostname or "").strip().lower()
    if hostname.startswith("www."):
        hostname = hostname[4:]
    return hostname or None


def _registrable_domain(hostname: str | None) -> str | None:
    if not hostname:
        return None
    if re.fullmatch(r"\d+\.\d+\.\d+\.\d+", hostname):
        return hostname
    parts = hostname.split(".")
    if len(parts) <= 2:
        return hostname
    suffix2 = ".".join(parts[-2:])
    suffix3 = ".".join(parts[-3:])
    if suffix2 in _MULTIPART_PUBLIC_SUFFIXES and len(parts) >= 3:
        return suffix3
    return suffix2


def _normalize_alias_map(source_alias_by_domain: Mapping[str, str] | None) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    if not source_alias_by_domain:
        return alias_map
    for raw_domain, raw_name in source_alias_by_domain.items():
        domain = _clean_text(str(raw_domain))
        name = _clean_text(str(raw_name))
        if not domain or not name:
            continue
        normalized_domain = domain.lower().removeprefix("www.")
        alias_map[normalized_domain] = name
    return alias_map


def _prettify_domain(domain: str | None, *, alias_map: Mapping[str, str]) -> str | None:
    if not domain:
        return None
    alias = alias_map.get(domain)
    if alias:
        return alias
    labels = [label for label in domain.split(".") if label and label not in {"com", "org", "net", "io", "co"}]
    if not labels:
        labels = [domain.split(".")[0]]
    return " ".join(label.upper() if len(label) <= 3 else label.capitalize() for label in labels)


def _normalize_url(url: str | None) -> str | None:
    if not url:
        return None
    cleaned = _clean_text(url)
    if not cleaned:
        return None
    try:
        parsed = urllib.parse.urlparse(cleaned)
    except Exception:
        return None
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return None

    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if not host:
        return None

    port = parsed.port
    netloc = host
    if port and not ((parsed.scheme == "http" and port == 80) or (parsed.scheme == "https" and port == 443)):
        netloc = f"{host}:{port}"

    params = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    kept = []
    for key, value in params:
        key_l = key.lower()
        if key_l in _TRACKING_PARAM_KEYS:
            continue
        if any(key_l.startswith(prefix) for prefix in _TRACKING_PARAM_PREFIXES):
            continue
        kept.append((key, value))
    query = urllib.parse.urlencode(kept, doseq=True)

    normalized = urllib.parse.urlunparse(
        (
            parsed.scheme.lower(),
            netloc,
            parsed.path or "",
            "",
            query,
            "",
        )
    )
    return normalized


def _title_from_url_slug(url: str | None) -> str | None:
    if not url:
        return None
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return None
    path = (parsed.path or "").strip("/")
    if not path:
        return None
    slug = path.split("/")[-1]
    slug = re.sub(r"\.[a-zA-Z0-9]{1,5}$", "", slug)
    slug = urllib.parse.unquote(slug)
    slug = re.sub(r"[-_]+", " ", slug)
    slug = re.sub(r"\s+", " ", slug).strip()
    if not slug:
        return None
    if len(slug) < 4:
        return None
    return slug[:1].upper() + slug[1:]


def _is_valid_title(candidate: str | None, *, hostname: str | None) -> bool:
    value = _clean_text(candidate)
    if not value:
        return False
    lowered = value.lower()
    if _looks_like_url(value):
        return False
    if hostname and lowered in {hostname, hostname.replace(".", " ")}:
        return False
    if lowered in _GENERIC_SOURCE_VALUES:
        return False
    if len(value) < 3:
        return False
    return True


def _is_valid_source(candidate: str | None, *, title: str | None) -> bool:
    value = _clean_text(candidate)
    if not value:
        return False
    lowered = value.lower()
    if _looks_like_url(value):
        return False
    if "." in value and " " not in value:
        return False
    if lowered in _GENERIC_SOURCE_VALUES:
        return False
    if title and lowered == title.lower():
        return False
    return True


def _extract_json_ld_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = _clean_text(value)
    if not cleaned:
        return None
    return cleaned


def _extract_source_from_json_ld_payload(payload: object) -> str | None:
    if isinstance(payload, dict):
        publisher = payload.get("publisher")
        if isinstance(publisher, dict):
            name = _extract_json_ld_string(publisher.get("name"))
            if name:
                return name
        if isinstance(publisher, list):
            for item in publisher:
                if isinstance(item, dict):
                    name = _extract_json_ld_string(item.get("name"))
                    if name:
                        return name
        source = _extract_json_ld_string(payload.get("sourceOrganization"))
        if source:
            return source
        source = _extract_json_ld_string(payload.get("author"))
        if source and len(source.split()) <= 4:
            return source
        graph = payload.get("@graph")
        if isinstance(graph, list):
            for node in graph:
                extracted = _extract_source_from_json_ld_payload(node)
                if extracted:
                    return extracted
    elif isinstance(payload, list):
        for item in payload:
            extracted = _extract_source_from_json_ld_payload(item)
            if extracted:
                return extracted
    return None


def _fetch_url_metadata(url: str, *, timeout: float) -> tuple[str | None, str | None]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            content_type = (response.headers.get("Content-Type") or "").lower()
            if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
                return None, None
            raw = response.read(300_000)
            charset = response.headers.get_content_charset() or "utf-8"
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError):
        return None, None
    except Exception:
        return None, None

    try:
        html = raw.decode(charset, errors="replace")
    except Exception:
        html = raw.decode("utf-8", errors="replace")

    parser = _MetaParser()
    try:
        parser.feed(html)
    except Exception:
        return None, None

    title = (
        _clean_text(parser.meta.get("og:title"))
        or _clean_text(parser.meta.get("twitter:title"))
        or _clean_text("".join(parser.title_parts))
    )
    source = (
        _clean_text(parser.meta.get("og:site_name"))
        or _clean_text(parser.meta.get("application-name"))
        or _clean_text(parser.meta.get("publisher"))
    )

    if not source and parser.ld_json_blocks:
        for payload in parser.ld_json_blocks:
            try:
                parsed = json.loads(payload)
            except Exception:
                continue
            source = _extract_source_from_json_ld_payload(parsed)
            if source:
                break
    return title, source


def normalize_citations(
    citations: list[Citation],
    *,
    enrich_citations: bool = False,
    enrichment_timeout: float = 3.0,
    enrichment_max_workers: int = 6,
    source_alias_by_domain: Mapping[str, str] | None = None,
) -> list[Citation]:
    """Return citations normalized into consistent url/title/source semantics."""
    normalized: list[Citation] = []
    enrichment_by_url: dict[str, tuple[str | None, str | None]] = {}
    alias_map = dict(_SOURCE_ALIAS_BY_DOMAIN)
    alias_map.update(_normalize_alias_map(source_alias_by_domain))

    candidates_for_enrichment: set[str] = set()

    for citation in citations:
        original_url = _clean_text(citation.url)
        original_title = _clean_text(citation.title)
        original_source = _clean_text(citation.source)

        url = _normalize_url(original_url)
        hostname = _extract_hostname(url or original_url)
        domain = _registrable_domain(hostname)

        title = original_title if _is_valid_title(original_title, hostname=hostname) else None
        source = original_source if _is_valid_source(original_source, title=title) else None
        title_confidence = "provider" if title else "fallback_unknown"
        source_confidence = "provider" if source else "fallback_unknown"

        if not title:
            title = _title_from_url_slug(url or original_url)
            if title:
                title_confidence = "heuristic_slug"

        if not source:
            alias_source = None
            if domain:
                alias_source = alias_map.get(domain)
            if not alias_source and hostname:
                alias_source = alias_map.get(hostname)
            prettified = alias_source or _prettify_domain(domain, alias_map=alias_map)
            if prettified:
                source = prettified
                source_confidence = "mapped_domain" if alias_source else "heuristic_domain"

        if enrich_citations and url:
            candidates_for_enrichment.add(url)

        source_key = domain or "unknown_publication"
        analytics_source = source or _prettify_domain(domain, alias_map=alias_map) or _UNKNOWN_PUBLICATION
        analytics_title = title or _UNKNOWN_TITLE
        analytics_url = url or original_url or _UNKNOWN_URL

        normalized.append(
            replace(
                citation,
                url=url,
                title=title,
                source=source,
                original_url=original_url,
                original_title=original_title,
                original_source=original_source,
                source_key=source_key,
                analytics_source=analytics_source,
                analytics_title=analytics_title,
                analytics_url=analytics_url,
                normalization_version="v1",
                source_confidence=source_confidence,
                title_confidence=title_confidence,
            )
        )

    if enrich_citations and candidates_for_enrichment:
        workers = max(1, min(enrichment_max_workers, len(candidates_for_enrichment)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_pairs = {
                executor.submit(_fetch_url_metadata, url, timeout=enrichment_timeout): url
                for url in candidates_for_enrichment
            }
            for future, url in future_pairs.items():
                try:
                    enrichment_by_url[url] = future.result()
                except Exception:
                    enrichment_by_url[url] = (None, None)

        enriched_items: list[Citation] = []
        for item in normalized:
            if not item.url:
                enriched_items.append(item)
                continue
            enriched_title, enriched_source = enrichment_by_url.get(item.url, (None, None))
            title = item.title
            source = item.source
            title_confidence = item.title_confidence or "fallback_unknown"
            source_confidence = item.source_confidence or "fallback_unknown"
            if (
                item.title_confidence != "provider"
                and _is_valid_title(enriched_title, hostname=_extract_hostname(item.url))
            ):
                title = _clean_text(enriched_title)
                title_confidence = "html_meta"
            if _is_valid_source(enriched_source, title=title):
                source = _clean_text(enriched_source)
                source_confidence = "html_meta"
            analytics_source = source or item.analytics_source or _UNKNOWN_PUBLICATION
            analytics_title = title or item.analytics_title or _UNKNOWN_TITLE
            enriched_items.append(
                replace(
                    item,
                    title=title,
                    source=source,
                    analytics_source=analytics_source,
                    analytics_title=analytics_title,
                    source_confidence=source_confidence,
                    title_confidence=title_confidence,
                )
            )
        normalized = enriched_items

    return normalized
