#!/usr/bin/env bash
set -euo pipefail
curl --fail --location --retry 3 "https://huggingface.co/datasets/cuichaowei/supramas-data-availability/resolve/main/SupraMAS_Supporting_Data.zip?download=true" --output SupraMAS_Supporting_Data.zip
