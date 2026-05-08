# Examples

- `quickstart/`: a minimal topic file for smoke testing the CLI.
- `case_studies/`: released generated protocols and strategy records from the manuscript data package.

Use `--dry-run` first:

```bash
PYTHONPATH=src python -m supramas.cli --topic "$(python - <<'PY'
import json
print(json.load(open('examples/quickstart/topic.json'))['topic'])
PY
)" --config configs/supramas.example.yaml --dry-run
```

