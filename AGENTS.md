# AGENTS.md

## Project purpose

`simpleai` provides a single public function, `run_prompt`, to run prompts across multiple GenAI providers with a unified interface.

## Public API contract

- Only public callable: `simpleai.run_prompt`.
- Keep backward compatibility for:
  - required `prompt` positional arg
  - `require_search`, `return_citations`, `file/files`, `binary_files`, `model`, `output_format`
- Return behavior:
  - plain result when citations are disabled
  - `(result, citations)` tuple when citations are enabled

## Architecture

- `simpleai/api.py`: orchestration and top-level `run_prompt`
- `simpleai/settings.py`: Django-first settings loader + JSON fallback
- `simpleai/model_registry.py`: model/provider resolution
- `simpleai/adapters/`: provider adapters and logging adapter
- `simpleai/files/extractor.py`: text extraction fallback pipeline

## Provider adapter expectations

- Adapters must implement `BaseAdapter.run(...)` and return `AdapterResponse`.
- Keep provider SDK imports inside adapter init (or lazy) for clear errors.
- Keep citations normalized through `simpleai.types.Citation`.

## File handling expectations

- If `binary_files=True` and adapter supports binary attachments, upload files.
- Otherwise extract text and append to prompt context.
- Supported extraction formats: `pdf`, `doc`, `docx`, `md`, `txt`, `json`, `rtf`.

## Testing

- Run tests with `pytest`.
- Tests should remain offline; mock provider SDK client calls.
- Add tests for any behavior change in:
  - settings loading
  - model resolution
  - run_prompt return shape
  - citations normalization
  - file extraction fallback

## Packaging

Before release verify:
- `pyproject.toml` version
- `CHANGELOG.md` entry
- `requirements.txt` coverage
- included settings examples and `LICENSE`

## Conventions

- Keep code ASCII unless required by existing files.
- Prefer small cohesive modules.
- Raise typed errors from `simpleai.exceptions`.
