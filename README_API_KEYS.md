# API Key Setup Guide

This guide explains how to get credentials for each provider used by `simpleai`.

## Summary

For standard API usage in this library, each provider uses an API key.
No separate app ID/client ID is required for the default SDK flows in this project.

## OpenAI

1. Create/sign in to your OpenAI account.
2. Open the API keys page and create a key.
3. Set `OPENAI_API_KEY` or place it in settings.

Docs:
- [OpenAI quickstart](https://platform.openai.com/docs/quickstart)

## Anthropic (Claude)

1. Create/sign in to Anthropic Console.
2. Create an API key.
3. Set `ANTHROPIC_API_KEY` or place it in settings.

Docs:
- [Anthropic API getting started](https://docs.anthropic.com/en/api/getting-started)

## Google Gemini (Standard API vs Vertex AI)

SimpleAI supports both the standard Google AI Studio API (default) and Google Cloud Vertex AI.

### Option 1: Standard API (Default)

1. Open [Google AI Studio](https://aistudio.google.com/).
2. Generate a Gemini API key.
3. Set `GEMINI_API_KEY` in your environment or place it in settings.

Docs:
- [Gemini API quickstart](https://ai.google.dev/gemini-api/docs/quickstart)

### Option 2: Vertex AI

If you need enterprise features or specific compliance guarantees, you can switch the adapter to use Vertex AI. 

For full step-by-step instructions on setting up a Google Cloud Project, Service Account, and downloading credentials, see the **[Vertex AI Setup Guide](README_VERTEX_AI.md)**.

**Configuration:**

Once your credentials are set up (via the `GOOGLE_APPLICATION_CREDENTIALS` environment variable), you can switch to Vertex AI by configuring `.env` variables or through Django/`ai_settings.json`.

**Via Environment Variables (`.env`):**
```env
GEMINI_USE_VERTEXAI=true
GEMINI_VERTEXAI_PROJECT=your-google-cloud-project-id
GEMINI_VERTEXAI_LOCATION=us-central1
```

**Via Django `settings.py` / `ai_settings.json`:**
```json
{
  "providers": {
    "gemini": {
      "use_vertexai": true,
      "vertexai_project": "your-google-cloud-project-id",
      "vertexai_location": "us-central1"
    }
  }
}
```

*Note: When `use_vertexai` is `true`, the `api_key` setting is ignored and standard GCP authentication is used instead.*

## xAI (Grok)

1. Create/sign in to xAI Console.
2. Generate an API key.
3. Set `XAI_API_KEY` or place it in settings.

Accepted env vars in this library: `XAI_API_KEY`, `GROK_API_KEY`.

Accepted provider aliases in `run_prompt(model=...)`: `grok`, `xai`.

Docs:
- [xAI API key docs](https://docs.x.ai/docs/overview/api-key)

## Perplexity

1. Create/sign in to Perplexity API dashboard.
2. Create an API key.
3. Set `PERPLEXITY_API_KEY` or place it in settings.

Accepted env vars in this library: `PERPLEXITY_API_KEY`, `PPLX_API_KEY`.

Docs:
- [Perplexity quickstart](https://docs.perplexity.ai/getting-started/quickstart)

## Example (`ai_settings.json`)

```json
{
  "providers": {
    "openai": {"api_key": "YOUR_OPENAI_API_KEY"},
    "claude": {"api_key": "YOUR_ANTHROPIC_API_KEY"},
    "gemini": {"api_key": "YOUR_GEMINI_API_KEY"},
    "grok": {"api_key": "YOUR_XAI_API_KEY"},
    "perplexity": {"api_key": "YOUR_PERPLEXITY_API_KEY"}
  }
}
```
