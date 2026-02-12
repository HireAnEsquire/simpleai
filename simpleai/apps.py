"""Django app configuration for SimpleAI."""

from __future__ import annotations

from django.apps import AppConfig


class SimpleAIConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "simpleai"
    verbose_name = "Simple AI"
