# Security And Release Hygiene

This public release excludes real local configuration files, API keys, private model gateways, and old Git history.

Before pushing a new release:

```bash
rg -n "sk-|AKLT|api[_-]?key|password|secret|token" .
python -m compileall src benchmark scripts
python scripts/verify_checksums.py data_availability/metadata/checksums_sha256.csv
```

Expected matches should be limited to placeholder configuration keys, code that reads environment variables, and benchmark/data text that is not credential material.

