#!/usr/bin/env python3
"""Compatibility wrapper for the packaged provider smoke runner."""

from simpleai.scripts.run_provider_smoke import build_parser, main


if __name__ == "__main__":
    raise SystemExit(main())
