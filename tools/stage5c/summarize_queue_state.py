#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Create JSON and Markdown summaries for a queue run.")
    parser.add_argument("--state", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--report-md", required=True)
    parser.add_argument("--require-outputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def valid_metrics(path):
    path = Path(path)
    if not path.is_file():
        return False
    try:
        values = load_json(path)
        values = values.get("metrics", values)
        return all(values.get(name) is not None for name in ("NMI", "ARI", "ACC", "Purity"))
    except Exception:
        return False


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
    state = load_json(args.state)
    jobs = state.get("jobs", [])
    if isinstance(jobs, dict):
        jobs = [{"id": key, **value} for key, value in jobs.items()]
    counts = Counter(str(job.get("status", "unknown")).lower() for job in jobs)
    details = []
    for job in jobs:
        expected = job.get("expected_output")
        output_valid = valid_metrics(expected) if expected else False
        details.append(
            {
                "id": job.get("id"),
                "dataset": job.get("dataset"),
                "missing_rate": job.get("missing_rate"),
                "method": job.get("method"),
                "seed": job.get("seed"),
                "status": job.get("status"),
                "returncode": job.get("returncode"),
                "log_path": job.get("log_path"),
                "expected_output": expected,
                "output_valid": output_valid,
            }
        )
    invalid_outputs = [item for item in details if not item["output_valid"]]
    passed = counts.get("failed", 0) == 0 and counts.get("succeeded", 0) == len(jobs)
    if args.require_outputs:
        passed = passed and not invalid_outputs
    report = {
        "state": Path(args.state).as_posix(),
        "total_jobs": len(jobs),
        "counts": dict(sorted(counts.items())),
        "invalid_outputs": len(invalid_outputs),
        "passed": passed,
        "jobs": details,
    }
    lines = [
        "# Stage 5C Queue Run Summary",
        "",
        f"- Total jobs: {len(jobs)}",
        f"- Pending jobs: {counts.get('pending', 0)}",
        f"- Running jobs: {counts.get('running', 0)}",
        f"- Succeeded jobs: {counts.get('succeeded', 0)}",
        f"- Failed jobs: {counts.get('failed', 0)}",
        f"- Missing/invalid expected outputs: {len(invalid_outputs)}",
        f"- Status: {'PASS' if passed else 'FAILED'}",
        "",
        "| dataset | missing_rate | method | seed | status | return code | output valid | log |",
        "|---|---:|---|---:|---|---:|---:|---|",
    ]
    for item in details:
        lines.append(
            f"| {item['dataset']} | {item['missing_rate']} | {item['method']} | {item['seed']} | "
            f"{item['status']} | {item['returncode']} | {item['output_valid']} | {item['log_path'] or ''} |"
        )
    lines.append("")
    write_output(args.report_json, report, args.overwrite)
    write_output(args.report_md, "\n".join(lines), args.overwrite)
    print("PASS" if passed else "FAILED")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
