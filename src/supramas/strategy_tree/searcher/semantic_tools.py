"""Optional helpers for loading PDFs with GROBID."""

from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import GrobidParser


def load_pdf_directory(path: str | Path, glob: str = "*"):
    """Load PDFs from a directory using the caller's local GROBID setup."""
    loader = GenericLoader.from_filesystem(
        str(path),
        glob=glob,
        suffixes=[".pdf"],
        parser=GrobidParser(segment_sentences=False),
    )
    return loader.load()

