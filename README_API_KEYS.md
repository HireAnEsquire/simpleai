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

## Google Gemini

1. Open Google AI Studio.
2. Generate a Gemini API key.
3. Set `GEMINI_API_KEY` or place it in settings.

Docs:
- [Gemini API quickstart](https://ai.google.dev/gemini-api/docs/quickstart)

Notes:
- If you use Gemini via Vertex AI instead of AI Studio key auth, project/location and Google credentials are required by that platform.

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
