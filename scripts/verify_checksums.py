#!/usr/bin/env python3
"""Verify SHA256 checksums for a SupraMAS data package."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify checksums_sha256.csv.")
    parser.add_argument("checksum_csv", help="Path to data_availability/metadata/checksums_sha256.csv")
    args = parser.parse_args()

    checksum_csv = Path(args.checksum_csv).resolve()
    base_dir = checksum_csv.parents[1]
    failures: list[str] = []

    with checksum_csv.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            rel = row["relative_path"]
            expected = row["sha256"]
            path = base_dir / rel
            if not path.exists():
                failures.append(f"missing: {rel}")
                continue
            actual = sha256(path)
            if actual != expected:
                failures.append(f"mismatch: {rel}")

    if failures:
        for failure in failures:
            print(failure)
        return 1
    print(f"All checksums verified under {base_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

