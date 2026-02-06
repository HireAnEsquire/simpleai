from __future__ import annotations

import json
from pathlib import Path

import pytest
from pypdf import PdfWriter

from simpleai.exceptions import FileExtractionError
from simpleai.files.extractor import collect_file_paths, extract_text_from_file, extract_text_from_files


def test_collect_file_paths_deduplicates(tmp_path: Path) -> None:
    a = tmp_path / "a.txt"
    a.write_text("hello", encoding="utf-8")

    paths = collect_file_paths(file=a, files=[a, str(a)])
    assert len(paths) == 1
    assert paths[0] == a.resolve()


def test_extract_text_plain_and_json(tmp_path: Path) -> None:
    txt = tmp_path / "test.txt"
    txt.write_text("plain", encoding="utf-8")

    js = tmp_path / "test.json"
    js.write_text(json.dumps({"x": 1}), encoding="utf-8")

    assert extract_text_from_file(txt) == "plain"
    assert '"x": 1' in extract_text_from_file(js)


def test_extract_rtf(tmp_path: Path) -> None:
    rtf = tmp_path / "sample.rtf"
    rtf.write_text(r"{\rtf1\ansi This is {\b bold}.}", encoding="utf-8")

    extracted = extract_text_from_file(rtf)
    assert "This is" in extracted


def test_extract_docx(tmp_path: Path) -> None:
    from docx import Document

    path = tmp_path / "sample.docx"
    doc = Document()
    doc.add_paragraph("Hello from docx")
    doc.save(path)

    extracted = extract_text_from_file(path)
    assert "Hello from docx" in extracted


def test_extract_pdf_supported(tmp_path: Path) -> None:
    path = tmp_path / "sample.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with path.open("wb") as handle:
        writer.write(handle)

    extracted = extract_text_from_file(path)
    assert isinstance(extracted, str)


def test_extract_text_from_files(tmp_path: Path) -> None:
    path = tmp_path / "a.md"
    path.write_text("# Header", encoding="utf-8")

    extracted = extract_text_from_files([path])
    assert len(extracted) == 1
    assert extracted[0].path == path.resolve()
    assert "Header" in extracted[0].text


def test_unsupported_extension_raises(tmp_path: Path) -> None:
    path = tmp_path / "a.csv"
    path.write_text("x,y", encoding="utf-8")

    with pytest.raises(FileExtractionError):
        extract_text_from_file(path)
