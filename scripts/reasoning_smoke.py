#!/usr/bin/env python3
"""Terminal smoke test for reasoning_level across SimpleAI providers.

Usage:
    python -m simpleai.scripts.reasoning_smoke
    python -m simpleai.scripts.reasoning_smoke --provider openai --level high
    python -m simpleai.scripts.reasoning_smoke --single-level --level medium
"""

import argparse
import os
import sys
from collections.abc import Sequence

from simpleai.adapters import ADAPTER_CLASSES, REASONING_LEVELS
from simpleai.api import run_prompt

DEFAULT_PROMPT = "Reply with exactly one word: pong."


PROVIDER_MODELS: dict[str, str] = {
    "openai": os.getenv("SIMPLEAI_SMOKE_OPENAI_MODEL", "gpt-5.4-mini"),
    "claude": os.getenv("SIMPLEAI_SMOKE_CLAUDE_MODEL", "claude-sonnet-4-6"),
    "gemini": os.getenv("SIMPLEAI_SMOKE_GEMINI_MODEL", "gemini-2.5-flash"),
    "grok": os.getenv("SIMPLEAI_SMOKE_GROK_MODEL", "grok-4-1-fast-reasoning"),
    "perplexity": os.getenv("SIMPLEAI_SMOKE_PERPLEXITY_MODEL", "fast-search"),
}


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
    return parser.parse_args(argv)


def _providers_to_run(selected: str | None) -> list[str]:
    if selected:
        return [selected]
    return sorted(ADAPTER_CLASSES)


def _levels_to_run() -> list[str]:
    return list(REASONING_LEVELS)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    providers = _providers_to_run(args.provider)
    levels = [args.level] if args.single_level else _levels_to_run()

    failures = 0
    for provider in providers:
        model = PROVIDER_MODELS[provider]
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
