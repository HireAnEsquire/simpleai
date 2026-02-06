"""SimpleAI exception hierarchy."""

from __future__ import annotations


class SimpleAIError(Exception):
    """Base error for SimpleAI."""


class SettingsError(SimpleAIError):
    """Raised for invalid or missing configuration."""


class ProviderError(SimpleAIError):
    """Raised when a provider adapter fails."""


class ModelResolutionError(SimpleAIError):
    """Raised when model/provider resolution fails."""


class FileExtractionError(SimpleAIError):
    """Raised when file extraction fails."""
