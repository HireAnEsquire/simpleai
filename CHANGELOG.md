# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- `default_reasoning_level` settings option (Django `SIMPLEAI` / `ai_settings.json`). When unset or empty, providers use their standard model defaults.
- Anthropic rate limiting support for Tier 1 accounts:
  - Automatic retry using `retry-after` header from 429 responses
  - Configurable retry count (`max_retries`)
  - Option to skip secondary citation API call (`skip_citation_followup`)

### Fixed
- Citation reliability for latest default models when structured output is requested:
  - Grok: fall back to `response.citations` and parse `web_search_call` output when inline markers are absent.
  - Gemini: secondary search pass when JSON schema responses omit grounding metadata.
  - Perplexity: use search tool results from structured-output responses and secondary search pass when needed.

- Gemini GCP settings renamed from `vertexai_*` / `use_vertexai` to `enterprise_*` / `use_enterprise` (aligned with `google-genai` 2.x). Legacy `vertexai_*` keys and `GEMINI_USE_VERTEXAI` / `GEMINI_VERTEXAI_*` env vars remain supported. Search grounding still uses standard `GoogleSearch`.
- Require `google-genai` 2.x (`>=2.0.0,<3.0.0`). v2 breaking changes are limited to the Interactions API; `GenerateContent` usage in `GeminiAdapter` is unchanged.
- Default models updated to latest provider releases (2026-05-26):
  - Gemini: `gemini-3.5-flash`
  - OpenAI: `gpt-5.5`
  - Claude: `claude-opus-4-7`
  - Grok: `grok-4.3`
  - Perplexity: `sonar-deep-research` (unchanged)

## [0.1.0] - 2026-02-06

### Added
- Initial `simpleai` release with one public API: `run_prompt`.
- Provider adapters for OpenAI, Anthropic Claude, Google Gemini, xAI Grok, and Perplexity.
- Model/provider resolution with aliases and heuristics, including provider aliases such as `chatgpt`, `claude`, `xai`, and `perplexityai`.
- Search grounding and normalized citations across providers.
- Structured output support via Pydantic models, with provider-compatible schema handling.
- File handling with binary upload when supported, plus text-extraction fallback.
- Text extraction support for `pdf`, `doc`, `docx`, `md`, `txt`, `json`, and `rtf`.
- Settings system that loads from Django settings first, then `ai_settings.json` (including common app-root discovery paths).
- Bundled settings examples for Django and non-Django usage.
- API-key preflight validation with provider env-var aliases.
- Centralized logging adapter for prompt lifecycle, payload metadata, timing, and error capture.
- Catch-all `SimpleAIException` surface for `run_prompt`, with original exception preserved for debugging.
- Manual cross-provider smoke runner tooling:
  - Shared runner module (`simpleai/provider_smoke.py`)
  - Standalone script (`scripts/run_provider_smoke.py`)
  - Django management command (`run_provider_smoke`)
  - Colorized summary and per-provider file-handling mode output (`binary upload` vs `parsed text`)
  - Bundled sample resume file (`simpleai/samples/functionalsample.pdf`)
- Documentation set including `README.md`, `README_API_KEYS.md`, and `README_PYPI.md`.
- Packaging and distribution files for pip/PyPI (`pyproject.toml`, `MANIFEST.in`, `requirements.txt`).
- Django app integration (`SimpleAIConfig`).
- Test suite compatible with pytest.
- MIT license and project `.gitignore`.

### Defaults
- Default provider order: `gemini`, `openai`, `claude`, `grok`, `perplexity`.
- Default models at release time:
  - Gemini: `gemini-3.1-pro-preview`
  - OpenAI: `gpt-5.2`
  - Claude: `claude-opus-4-6`
  - Grok: `grok-4-1-fast-reasoning`
  - Perplexity: `deep-research`
