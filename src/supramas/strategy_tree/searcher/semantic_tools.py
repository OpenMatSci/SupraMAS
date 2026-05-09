"""Optional document-loading extension point."""

from __future__ import annotations

from pathlib import Path


def load_pdf_directory(path: str | Path, glob: str = "*"):
    raise NotImplementedError("PDF loading is not included in the public skeleton.")

