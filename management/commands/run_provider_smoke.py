"""Django command to run provider smoke checks."""

from __future__ import annotations

from django.core.management.base import BaseCommand, CommandError

from simpleai.provider_smoke import resolve_sample_file_path, run_provider_matrix


class Command(BaseCommand):
    help = "Run resume/search/citation smoke checks across AI providers using simpleai.run_prompt"

    def add_arguments(self, parser) -> None:
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

    def handle(self, *args, **options):
        try:
            file_path = resolve_sample_file_path(options.get("file"))
        except FileNotFoundError as exc:
            raise CommandError(str(exc)) from exc

        results = run_provider_matrix(
            file_path=file_path,
            settings_file=options.get("settings_file"),
            providers=options.get("providers"),
            emit=self.stdout.write,
            use_color=not bool(options.get("no_color", False)),
        )

        if any(item.status == "failed" for item in results):
            raise CommandError("One or more providers failed smoke validation.")

        if any(item.status == "missing_key" for item in results):
            self.stderr.write(self.style.WARNING("Some providers were skipped due to missing API keys."))
