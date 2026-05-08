# Data Availability

The `data_availability/` directory mirrors the manuscript data package. It includes:

- source data for tables and figure panels
- optimization trajectories
- benchmark inputs, labels, and results
- expert evaluation files
- generated protocols
- interpretability records
- metadata, manifests, and checksums

Use:

```bash
python scripts/verify_checksums.py data_availability/metadata/checksums_sha256.csv
```

