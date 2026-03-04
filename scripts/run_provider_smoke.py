#!/usr/bin/env python3
"""Manual cross-provider smoke runner for simpleai."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simpleai.provider_smoke import resolve_sample_file_path, run_provider_matrix



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the same resume/search/citation prompt across all providers."
    )
    parser.add_argument(
        "--file",
        help="Path to resume PDF. Defaults to detected functionalsample.pdf locations.",
    )
    parser.add_argument(
        "--settings-file",
        help="Optional ai_settings.json override path.",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        help="Optional subset: openai anthropic gemini grok xai perplexity",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colors in output.",
    )
    return parser



def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        file_path = resolve_sample_file_path(args.file)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    results = run_provider_matrix(
        file_path=file_path,
        settings_file=args.settings_file,
        providers=args.providers,
        use_color=not args.no_color,
    )

    if any(item.status == "failed" for item in results):
        return 1
    if any(item.status == "missing_key" for item in results):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
