#!/usr/bin/env python3
"""Export a simple file manifest for a directory."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Export relative path and file size manifest.")
    parser.add_argument("root", help="Directory to scan")
    parser.add_argument("--output", default="file_manifest.csv", help="Output CSV path")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = Path(args.output)
    with output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["relative_path", "size_bytes"])
        for path in sorted(root.rglob("*")):
            if path.is_file():
                writer.writerow([path.relative_to(root).as_posix(), path.stat().st_size])
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

