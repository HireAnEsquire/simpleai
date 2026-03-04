"""Example Django settings for SimpleAI."""

SIMPLEAI = {
    "defaults": ["gemini", "openai", "claude", "grok", "perplexity"],
    "providers": {
        "gemini": {
            "api_key": "YOUR_GEMINI_API_KEY",
            "default_model": "gemini-3.1-pro-preview",
            "max_output_tokens": 65536,
            "use_vertexai": False,  # Set to True to use Vertex AI instead of standard API
            "vertexai_project": "YOUR_GOOGLE_CLOUD_PROJECT_ID",  # Required if use_vertexai=True
            "vertexai_location": "us-central1",  # Required if use_vertexai=True
        },
        "claude": {
            "api_key": "YOUR_ANTHROPIC_API_KEY",
            "default_model": "claude-opus-4-6",
            "max_tokens": 128000,
            "max_retries": 3,  # retries on 429 errors (uses retry-after header)
            "skip_citation_followup": False,  # skip extra API call for citations
        },
        "openai": {
            "api_key": "YOUR_OPENAI_API_KEY",
            "default_model": "gpt-5.2",
            "max_output_tokens": 128000,
            "base_url": None,
        },
        "grok": {
            "api_key": "YOUR_XAI_API_KEY",
            "default_model": "grok-4-1-fast-reasoning",
            "max_tokens": 8192,
        },
        "perplexity": {
            "api_key": "YOUR_PERPLEXITY_API_KEY",
            "default_model": "sonar-deep-research",
            "max_output_tokens": 128000,
        },
    },
    "logging": {
        "enabled": True,
        "network_logging": True,
        "django_logfile": "django",
        "logfile_location": "./simpleai.log",
    },
}
