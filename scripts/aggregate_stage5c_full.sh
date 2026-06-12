#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OVERWRITE_FLAG=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OVERWRITE_FLAG=(--overwrite)
fi

python aggregate_block1_results.py \
  --datasets Reuters BDGP Wikipedia Handwritten \
  --missing-rates 0 10 30 50 70 90 \
  --methods llm_gimvc statistical_only mica jga_imvc freecsl \
  --seeds 0 1 2 3 4 \
  --missing-pattern MCAR

python tools/stage5c/validate_aggregation.py \
  --expected-runs 600 \
  --report-json refine-logs/stage5c/full_aggregation_check.json \
  --report-md refine-logs/stage5c/full_aggregation_check.md \
  "${OVERWRITE_FLAG[@]}"

python tools/stage5c/build_paper_tables.py "${OVERWRITE_FLAG[@]}"
