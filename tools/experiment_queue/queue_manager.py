#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


STATE_LOCK = threading.Lock()


def parse_args():
    parser = argparse.ArgumentParser(description="Run or resume jobs from an experiment manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument("--conda-env", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun-failed", action="store_true")
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    os.replace(temp_path, path)


def valid_metrics(path):
    path = Path(path)
    if not path.is_file():
        return False
    try:
        payload = load_json(path)
        metrics = payload.get("metrics", payload)
        return all(metrics.get(key) is not None for key in ("NMI", "ARI", "ACC", "Purity"))
    except Exception:
        return False


def now():
    return datetime.now(timezone.utc).isoformat()


def initial_state(manifest, manifest_path):
    return {
        "manifest": Path(manifest_path).as_posix(),
        "total_jobs": len(manifest["jobs"]),
        "updated_at": now(),
        "jobs": [
            {
                **job,
                "status": "pending",
                "returncode": None,
                "log_path": None,
                "started_at": None,
                "finished_at": None,
            }
            for job in manifest["jobs"]
        ],
    }


def merge_state(manifest, existing, manifest_path):
    if not existing:
        return initial_state(manifest, manifest_path)
    previous = {job.get("id"): job for job in existing.get("jobs", [])}
    merged = initial_state(manifest, manifest_path)
    for job in merged["jobs"]:
        old = previous.get(job["id"])
        if old:
            for key in ("status", "returncode", "log_path", "started_at", "finished_at", "error"):
                if key in old:
                    job[key] = old[key]
    return merged


def save_state(state_path, state):
    with STATE_LOCK:
        state["updated_at"] = now()
        atomic_write_json(state_path, state)


def command_for_job(job, conda_env):
    command = job["cmd"]
    if conda_env:
        return f'conda run -n "{conda_env}" {command}'
    return command


def run_job(job, root, log_dir, conda_env, gpu):
    expected = root / job["expected_output"]
    log_path = log_dir / f"{job['id']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    command = command_for_job(job, conda_env)
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"command: {command}\n")
        log.write(f"cwd: {root.as_posix()}\n")
        log.write(f"expected_output: {expected.as_posix()}\n\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=root,
            env=env,
            shell=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    output_ok = valid_metrics(expected)
    return {
        "returncode": int(completed.returncode),
        "log_path": log_path.as_posix(),
        "output_ok": output_ok,
        "status": "succeeded" if completed.returncode == 0 and output_ok else "failed",
        "error": None if output_ok else f"Expected valid metrics missing: {expected.as_posix()}",
    }


def run_job_with_state(job, state_job, state, state_path, root, log_dir, conda_env, gpu):
    state_job.update(status="running", started_at=now(), finished_at=None, error=None)
    save_state(state_path, state)
    return run_job(job, root, log_dir, conda_env, gpu)


def resolve_root(manifest):
    configured = manifest.get("cwd")
    if not configured or configured == "__REMOTE_PROJECT_DIR__":
        return Path.cwd()
    return Path(configured).expanduser().resolve()


def main():
    args = parse_args()
    manifest = load_json(args.manifest)
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("Manifest must contain a 'jobs' list.")
    state_path = Path(args.state)
    log_dir = Path(args.log_dir)
    root = resolve_root(manifest)
    existing = load_json(state_path) if state_path.exists() else None
    state = merge_state(manifest, existing, args.manifest)
    state_by_id = {job["id"]: job for job in state["jobs"]}

    max_parallel = args.max_parallel or int(os.environ.get("MAX_PARALLEL", manifest.get("max_parallel", 1)))
    if max_parallel < 1:
        raise ValueError("max_parallel must be at least 1.")
    conda_env = args.conda_env or os.environ.get("CONDA_ENV") or manifest.get("conda")
    gpu_env = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpus = [item.strip() for item in gpu_env.split(",") if item.strip()] if gpu_env else manifest.get("gpus", [])

    runnable = []
    for index, job in enumerate(jobs):
        state_job = state_by_id[job["id"]]
        expected = root / job["expected_output"]
        if valid_metrics(expected):
            state_job.update(status="succeeded", returncode=0, finished_at=now(), error=None)
            continue
        if state_job.get("status") == "failed" and not args.rerun_failed:
            continue
        if state_job.get("status") == "succeeded":
            state_job["status"] = "pending"
        runnable.append((index, job))
    save_state(state_path, state)

    if args.dry_run:
        print(f"total jobs: {len(jobs)}")
        print(f"runnable jobs: {len(runnable)}")
        return

    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {}
        for index, job in runnable:
            state_job = state_by_id[job["id"]]
            gpu = gpus[index % len(gpus)] if gpus else None
            future = executor.submit(
                run_job_with_state,
                job,
                state_job,
                state,
                state_path,
                root,
                log_dir,
                conda_env,
                gpu,
            )
            futures[future] = state_job

        for future in as_completed(futures):
            state_job = futures[future]
            try:
                result = future.result()
                state_job.update(result)
            except Exception as exc:
                state_job.update(status="failed", returncode=-1, error=str(exc))
            state_job["finished_at"] = now()
            save_state(state_path, state)

    counts = {}
    for job in state["jobs"]:
        counts[job["status"]] = counts.get(job["status"], 0) + 1
    print(json.dumps(counts, indent=2))
    if counts.get("failed", 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
