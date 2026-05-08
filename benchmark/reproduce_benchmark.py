#!/usr/bin/env python3
"""Compatibility wrapper for scripts/reproduce_benchmark.py."""

from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/reproduce_benchmark.py"), run_name="__main__")

