#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OVERWRITE_FLAG=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OVERWRITE_FLAG=(--overwrite)
fi

mkdir -p refine-logs/stage5c/smoke_logs refine-logs/experiment_queue

python tools/stage5c/verify_external_baselines.py "${OVERWRITE_FLAG[@]}" || {
  echo "External baseline verification failed; smoke test was not started." >&2
  exit 1
}

python tools/experiment_queue/build_manifest.py \
  --config refine-logs/experiment_queue/block1_grid_spec_stage5c_smoke.json \
  --output refine-logs/experiment_queue/block1_manifest_stage5c_smoke.json \
  "${OVERWRITE_FLAG[@]}"

python tools/stage5c/check_stage5c_manifest.py \
  --manifest refine-logs/experiment_queue/block1_manifest_stage5c_smoke.json \
  --expected-total 20 \
  --missing-rates 50 \
  --seeds 0 \
  --report-json refine-logs/stage5c/smoke_manifest_check.json \
  --report-md refine-logs/stage5c/smoke_manifest_check.md \
  "${OVERWRITE_FLAG[@]}"

set +e
python tools/experiment_queue/queue_manager.py \
  --manifest refine-logs/experiment_queue/block1_manifest_stage5c_smoke.json \
  --state refine-logs/experiment_queue/stage5c_smoke_queue_state.json \
  --log-dir refine-logs/stage5c/smoke_logs \
  --max-parallel "${MAX_PARALLEL:-1}" \
  --conda-env "${CONDA_ENV:-llm-gimvc}"
QUEUE_RC=$?

python tools/stage5c/summarize_queue_state.py \
  --state refine-logs/experiment_queue/stage5c_smoke_queue_state.json \
  --report-json refine-logs/stage5c/smoke_run_summary.json \
  --report-md refine-logs/stage5c/smoke_run_summary.md \
  --require-outputs \
  "${OVERWRITE_FLAG[@]}"
SUMMARY_RC=$?
set -e

if [[ "$QUEUE_RC" -ne 0 || "$SUMMARY_RC" -ne 0 ]]; then
  echo "Smoke jobs failed or expected metrics are missing; aggregation and full launch are blocked." >&2
  exit 1
fi

python aggregate_block1_results.py \
  --datasets Reuters BDGP Wikipedia Handwritten \
  --missing-rates 50 \
  --methods llm_gimvc statistical_only mica jga_imvc freecsl \
  --seeds 0 \
  --missing-pattern MCAR

python tools/stage5c/validate_aggregation.py \
  --expected-runs 20 \
  "${OVERWRITE_FLAG[@]}"
