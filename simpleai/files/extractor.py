"""Text extraction helpers for supported file formats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader
from striprtf.striprtf import rtf_to_text

from simpleai.exceptions import FileExtractionError
from simpleai.types import ExtractedFile

_TEXT_EXTENSIONS = {".txt", ".md"}
_SUPPORTED_EXTENSIONS = {".pdf", ".doc", ".docx", ".md", ".txt", ".json", ".rtf"}


def collect_file_paths(
    file: str | Path | None = None,
    files: str | Path | Iterable[str | Path] | None = None,
) -> list[Path]:
    """Normalize `file`/`files` args into a de-duplicated Path list."""

    values: list[str | Path] = []
    if file is not None:
        values.append(file)

    if files is not None:
        if isinstance(files, (str, Path)):
            values.append(files)
        else:
            values.extend(list(files))

    normalized: list[Path] = []
    seen: set[str] = set()
    for value in values:
        path = Path(value).expanduser().resolve()
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(path)

    return normalized



def extract_text_from_file(path: str | Path) -> str:
    """Extract plain text from supported file types."""

    file_path = Path(path)
    if not file_path.exists():
        raise FileExtractionError(f"File does not exist: {file_path}")

    ext = file_path.suffix.lower()
    if ext not in _SUPPORTED_EXTENSIONS:
        raise FileExtractionError(
            f"Unsupported file extension '{ext}' for {file_path}. Supported: {sorted(_SUPPORTED_EXTENSIONS)}"
        )

    try:
        if ext in _TEXT_EXTENSIONS:
            return file_path.read_text(encoding="utf-8")

        if ext == ".json":
            payload = json.loads(file_path.read_text(encoding="utf-8"))
            return json.dumps(payload, indent=2, sort_keys=True)

        if ext == ".rtf":
            return rtf_to_text(file_path.read_text(encoding="utf-8", errors="ignore"))

        if ext == ".pdf":
            reader = PdfReader(str(file_path))
            parts: list[str] = []
            for page in reader.pages:
                parts.append(page.extract_text() or "")
            return "\n".join(parts).strip()

        if ext == ".docx":
            from docx import Document  # lazy import for smaller import surface

            doc = Document(str(file_path))
            return "\n".join(paragraph.text for paragraph in doc.paragraphs).strip()

        if ext == ".doc":
            try:
                import textract  # type: ignore

                content = textract.process(str(file_path))
                return content.decode("utf-8", errors="ignore").strip()
            except Exception:
                # Fallback when optional doc extractors are unavailable.
                return file_path.read_bytes().decode("latin-1", errors="ignore").strip()

    except Exception as exc:
        raise FileExtractionError(f"Failed extracting text from {file_path}: {exc}") from exc

    raise FileExtractionError(f"Unable to extract text from {file_path}")



def extract_text_from_files(paths: Iterable[str | Path]) -> list[ExtractedFile]:
    """Extract text from a list of files."""

    extracted: list[ExtractedFile] = []
    for item in paths:
        path = Path(item).expanduser().resolve()
        extracted.append(ExtractedFile(path=path, text=extract_text_from_file(path)))
    return extracted
