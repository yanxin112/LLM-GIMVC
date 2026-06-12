#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${LLM_PROVIDER:?Set LLM_PROVIDER to a real provider before a full run.}"
: "${EMBEDDING_PROVIDER:?Set EMBEDDING_PROVIDER to a real provider before a full run.}"

OVERWRITE_FLAG=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OVERWRITE_FLAG=(--overwrite)
fi

mkdir -p refine-logs/experiment_queue/stage5c_full_logs refine-logs/stage5c

python tools/stage5c/verify_external_baselines.py "${OVERWRITE_FLAG[@]}" || {
  echo "External baseline verification failed. Full queue was not launched." >&2
  exit 1
}

python tools/experiment_queue/build_manifest.py \
  --config refine-logs/experiment_queue/block1_grid_spec.json \
  --output refine-logs/experiment_queue/block1_manifest.json \
  "${OVERWRITE_FLAG[@]}"

python tools/stage5c/check_stage5c_manifest.py \
  --manifest refine-logs/experiment_queue/block1_manifest.json \
  --expected-total 600 \
  --missing-rates 0,10,30,50,70,90 \
  --seeds 0,1,2,3,4 \
  --report-json refine-logs/stage5c/full_manifest_check.json \
  --report-md refine-logs/stage5c/full_manifest_check.md \
  "${OVERWRITE_FLAG[@]}"

python tools/experiment_queue/queue_manager.py \
  --manifest refine-logs/experiment_queue/block1_manifest.json \
  --state refine-logs/experiment_queue/stage5c_full_queue_state.json \
  --log-dir refine-logs/experiment_queue/stage5c_full_logs \
  --max-parallel "${MAX_PARALLEL:-1}" \
  --conda-env "${CONDA_ENV:-llm-gimvc}"
