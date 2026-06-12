#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path


CATEGORIES = [
    ("missing external repo", ("external baseline repo", "repo was not found", "no such file or directory")),
    ("missing entrypoint", ("entrypoint", "cannot open file", "module not found")),
    ("CUDA OOM", ("cuda out of memory", "outofmemoryerror", "cublas_status_alloc_failed")),
    ("metrics.json missing", ("expected valid metrics missing", "metrics.json missing")),
    ("pred_labels missing", ("prediction file missing", "pred_labels", "no usable output")),
    ("metric adapter failed", ("adapter_error", "adapt_results", "metric adapter")),
    ("dataset load failed", ("dataset", "loadmat", "data bundle not found", "baseline data missing")),
    ("LLM provider/API failed", ("api key", "provider", "rate limit", "authentication", "connection error")),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Classify failed Stage 5C queue jobs from state and logs.")
    parser.add_argument(
        "--state",
        default="refine-logs/experiment_queue/stage5c_full_queue_state.json",
    )
    parser.add_argument("--log-dir", default="refine-logs/experiment_queue/stage5c_full_logs")
    parser.add_argument("--report-json", default="refine-logs/stage5c/failed_jobs_analysis.json")
    parser.add_argument("--report-md", default="refine-logs/stage5c/failed_jobs_analysis.md")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def state_jobs(payload):
    jobs = payload.get("jobs", [])
    if isinstance(jobs, dict):
        normalized = []
        for job_id, job in jobs.items():
            item = dict(job)
            item.setdefault("id", job_id)
            normalized.append(item)
        return normalized
    if isinstance(jobs, list):
        return jobs
    raise ValueError("queue state 'jobs' must be a list or object.")


def read_log(job, log_dir):
    candidates = []
    if job.get("log_path"):
        candidates.append(Path(job["log_path"]))
    if job.get("id"):
        candidates.append(Path(log_dir) / f"{job['id']}.log")
    for path in candidates:
        if path.is_file():
            return path, path.read_text(encoding="utf-8", errors="replace")
    return (candidates[0] if candidates else None), ""


def classify(text, job):
    haystack = " ".join(
        [
            text,
            str(job.get("error", "")),
            str(job.get("stderr_tail", "")),
            str(job.get("stdout_tail", "")),
        ]
    ).lower()
    for category, markers in CATEGORIES:
        if any(marker in haystack for marker in markers):
            return category
    return "unknown"


def write_output(path, content, overwrite):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Report already exists: {path}. Pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, str):
        path.write_text(content, encoding="utf-8")
    else:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(content, handle, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    state_path = Path(args.state)
    if not state_path.is_file():
        raise FileNotFoundError(f"Queue state not found: {state_path}")
    payload = load_json(state_path)
    failed = [job for job in state_jobs(payload) if str(job.get("status", "")).lower() == "failed"]
    results = []
    for job in failed:
        log_path, text = read_log(job, args.log_dir)
        category = classify(text, job)
        results.append(
            {
                "id": job.get("id"),
                "dataset": job.get("dataset"),
                "missing_rate": job.get("missing_rate"),
                "method": job.get("method"),
                "seed": job.get("seed"),
                "returncode": job.get("returncode"),
                "category": category,
                "log_path": log_path.as_posix() if log_path else None,
                "error": job.get("error"),
                "log_tail": text[-4000:],
            }
        )
    counts = Counter(item["category"] for item in results)
    report = {
        "state": state_path.as_posix(),
        "total_jobs": len(state_jobs(payload)),
        "failed_jobs": len(results),
        "category_counts": dict(sorted(counts.items())),
        "jobs": results,
    }
    lines = [
        "# Stage 5C Failed Jobs Analysis",
        "",
        f"- Queue state: `{state_path.as_posix()}`",
        f"- Total jobs: {report['total_jobs']}",
        f"- Failed jobs: {report['failed_jobs']}",
        "",
        "## Failure Categories",
        "",
        "| category | count |",
        "|---|---:|",
    ]
    for category, count in sorted(counts.items()):
        lines.append(f"| {category} | {count} |")
    lines.extend(["", "## Failed Jobs", ""])
    if results:
        lines.extend(
            [
                "| id | dataset | missing_rate | method | seed | returncode | category | log |",
                "|---|---|---:|---|---:|---:|---|---|",
            ]
        )
        for item in results:
            lines.append(
                f"| {item['id']} | {item['dataset']} | {item['missing_rate']} | "
                f"{item['method']} | {item['seed']} | {item['returncode']} | "
                f"{item['category']} | {item['log_path'] or ''} |"
            )
    else:
        lines.append("No failed jobs.")
    lines.extend(
        [
            "",
            "Fix failures by category and rerun only failed jobs; do not restart the full matrix blindly.",
            "",
        ]
    )
    write_output(args.report_json, report, args.overwrite)
    write_output(args.report_md, "\n".join(lines), args.overwrite)
    print(json.dumps(report["category_counts"], indent=2))


if __name__ == "__main__":
    main()
