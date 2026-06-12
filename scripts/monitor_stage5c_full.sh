#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

STATE="${1:-refine-logs/experiment_queue/stage5c_full_queue_state.json}"
LOG_DIR="${2:-refine-logs/experiment_queue/stage5c_full_logs}"
OVERWRITE_FLAG=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OVERWRITE_FLAG=(--overwrite)
fi

python - "$STATE" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"Queue state not found: {path}")
with path.open("r", encoding="utf-8-sig") as handle:
    state = json.load(handle)
jobs = state.get("jobs", [])
if isinstance(jobs, dict):
    jobs = [{"id": key, **value} for key, value in jobs.items()]
counts = Counter(str(job.get("status", "unknown")).lower() for job in jobs)
print(f"total jobs: {len(jobs)}")
print(f"pending jobs: {counts.get('pending', 0)}")
print(f"running jobs: {counts.get('running', 0)}")
print(f"succeeded jobs: {counts.get('succeeded', 0)}")
print(f"failed jobs: {counts.get('failed', 0)}")
print("recent 20 failed jobs:")
failed = [job for job in jobs if str(job.get("status", "")).lower() == "failed"]
for job in failed[-20:]:
    print(
        f"  {job.get('id')} dataset={job.get('dataset')} rate={job.get('missing_rate')} "
        f"method={job.get('method')} seed={job.get('seed')} rc={job.get('returncode')} "
        f"log={job.get('log_path')}"
    )
PY

python tools/stage5c/analyze_failed_jobs.py \
  --state "$STATE" \
  --log-dir "$LOG_DIR" \
  "${OVERWRITE_FLAG[@]}"
