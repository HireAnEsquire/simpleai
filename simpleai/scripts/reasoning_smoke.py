"""Terminal smoke test for reasoning_level across SimpleAI providers.

Uses each provider's ``default_model`` from settings (Django / ``ai_settings.json``).
Override per provider with ``SIMPLEAI_SMOKE_<PROVIDER>_MODEL`` env vars if needed.

Usage:
    python -m simpleai.scripts.reasoning_smoke
    python -m simpleai.scripts.reasoning_smoke --provider openai --level high
    python -m simpleai.scripts.reasoning_smoke --single-level --level medium
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

from simpleai.adapters import ADAPTER_CLASSES, REASONING_LEVELS
from simpleai.api import run_prompt
from simpleai.model_registry import resolve_provider_and_model
from simpleai.settings import _load_from_django, _load_from_json, load_settings

DEFAULT_PROMPT = "Reply with exactly one word: pong."

# Optional per-provider overrides (otherwise uses providers.<name>.default_model from settings).
_SMOKE_MODEL_ENV_VARS: dict[str, str] = {
    "openai": "SIMPLEAI_SMOKE_OPENAI_MODEL",
    "claude": "SIMPLEAI_SMOKE_CLAUDE_MODEL",
    "gemini": "SIMPLEAI_SMOKE_GEMINI_MODEL",
    "grok": "SIMPLEAI_SMOKE_GROK_MODEL",
    "perplexity": "SIMPLEAI_SMOKE_PERPLEXITY_MODEL",
}


def _settings_source(settings_file: str | None) -> str:
    if _load_from_django() is not None:
        return "django (SIMPLEAI / SIMPLEAI_SETTINGS)"
    if _load_from_json(settings_file) is not None:
        if settings_file:
            return f"ai_settings.json ({settings_file})"
        env_path = os.getenv("SIMPLEAI_SETTINGS_FILE")
        if env_path:
            return f"ai_settings.json ({env_path})"
        return "ai_settings.json (discovered from cwd / app root)"
    return "simpleai built-in DEFAULT_SETTINGS"


def _resolve_smoke_model(settings: dict, provider: str) -> str:
    env_var = _SMOKE_MODEL_ENV_VARS.get(provider)
    if env_var:
        override = os.getenv(env_var)
        if override:
            return override
    _, model = resolve_provider_and_model(settings, provider)
    return model


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        choices=sorted(ADAPTER_CLASSES),
        help="Run a single provider (default: all configured providers)",
    )
    parser.add_argument(
        "--level",
        choices=list(REASONING_LEVELS),
        default=os.getenv("SIMPLEAI_SMOKE_REASONING_LEVEL", "low"),
        help="Reasoning level to send (default: low)",
    )
    parser.add_argument(
        "--single-level",
        action="store_true",
        help="Only run the selected --level value",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt text for the smoke call",
    )
    parser.add_argument(
        "--settings-file",
        default=os.getenv("SIMPLEAI_SETTINGS_FILE"),
        help="Optional path to ai_settings.json",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Print resolved models and settings source, then exit",
    )
    return parser.parse_args(argv)


def _providers_to_run(selected: str | None) -> list[str]:
    if selected:
        return [selected]
    return sorted(ADAPTER_CLASSES)


def _levels_to_run() -> list[str]:
    return list(REASONING_LEVELS)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = load_settings(args.settings_file)
    providers = _providers_to_run(args.provider)

    if args.show_config:
        import simpleai

        print(f"simpleai package: {simpleai.__file__}", flush=True)
        print(f"reasoning_smoke: {__file__}", flush=True)
        print(f"settings source: {_settings_source(args.settings_file)}", flush=True)
        for provider in providers:
            model = _resolve_smoke_model(settings, provider)
            env_var = _SMOKE_MODEL_ENV_VARS.get(provider, "")
            override = os.getenv(env_var) if env_var else None
            suffix = f" (env {env_var}={override!r})" if override else ""
            print(f"  {provider}: {model}{suffix}", flush=True)
        return 0

    levels = [args.level] if args.single_level else _levels_to_run()

    failures = 0
    for provider in providers:
        model = _resolve_smoke_model(settings, provider)
        for level in levels:
            label = f"{provider} model={model} reasoning_level={level}"
            print(f"\n=== {label} ===", flush=True)
            try:
                text = run_prompt(
                    args.prompt,
                    model=model,
                    require_search=False,
                    return_citations=False,
                    reasoning_level=level,
                    settings_file=args.settings_file,
                )
            except Exception as exc:  # pragma: no cover - manual smoke script
                failures += 1
                print(f"FAIL: {exc}", file=sys.stderr, flush=True)
                continue

            preview = str(text).strip().replace("\n", " ")[:200]
            print(f"OK: {preview!r}", flush=True)

    print(f"\nDone. failures={failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

