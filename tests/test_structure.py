from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_required_public_files_exist():
    for rel in [
        "README.md",
        "LICENSE",
        "CITATION.cff",
        "pyproject.toml",
        "configs/supramas.example.yaml",
        "benchmark/data/benchmark_topics.json",
        "data_availability/README.md",
    ]:
        assert (ROOT / rel).exists(), rel


def test_no_local_secret_files():
    forbidden = [".env", "config.yaml", "SupraMAS_data_availability(1).rar"]
    found = [p.name for p in ROOT.rglob("*") if p.name in forbidden]
    assert not found

