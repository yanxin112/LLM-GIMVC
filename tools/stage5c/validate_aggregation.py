#!/usr/bin/env python3
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


METRICS = ("NMI", "ARI", "ACC", "Purity")


def parse_args():
    parser = argparse.ArgumentParser(description="Validate Stage 5C aggregate CSV outputs.")
    parser.add_argument("--summary-dir", default="results/block1_summary")
    parser.add_argument("--expected-runs", type=int, required=True)
    parser.add_argument("--report-json", default="refine-logs/stage5c/smoke_aggregation_check.json")
    parser.add_argument("--report-md", default="refine-logs/stage5c/smoke_aggregation_check.md")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def missing_number(value):
    if value is None or str(value).strip() == "":
        return True
    try:
        return not math.isfinite(float(value))
    except ValueError:
        return True


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
    root = Path(args.summary_dir)
    paths = {
        "all_runs": root / "block1_all_runs.csv",
        "mean_std": root / "block1_mean_std.csv",
        "delta": root / "block1_delta_vs_statistical.csv",
        "summary": root / "block1_summary.json",
    }
    errors = [f"missing output: {path}" for path in paths.values() if not path.is_file()]
    rows = read_csv(paths["all_runs"]) if paths["all_runs"].is_file() else []
    delta_rows = read_csv(paths["delta"]) if paths["delta"].is_file() else []
    required = {"dataset", "missing_rate", "method", "seed", *METRICS}
    if rows and not required.issubset(rows[0]):
        errors.append(f"all_runs missing columns: {sorted(required - set(rows[0]))}")
    if len(rows) < args.expected_runs:
        errors.append(f"all_runs has {len(rows)} rows; expected at least {args.expected_runs}")
    keys = [(row.get("dataset"), row.get("missing_rate"), row.get("method"), row.get("seed")) for row in rows]
    duplicates = [key for key, count in Counter(keys).items() if count > 1]
    if duplicates:
        errors.append(f"duplicate run rows: {duplicates}")
    invalid_metrics = []
    for index, row in enumerate(rows, start=2):
        for metric in METRICS:
            if missing_number(row.get(metric)):
                invalid_metrics.append({"row": index, "metric": metric, "value": row.get(metric)})
    if invalid_metrics:
        errors.append(f"NaN/empty metrics: {len(invalid_metrics)}")
    if not delta_rows:
        errors.append("delta_vs_statistical is empty")
    elif any(missing_number(row.get(f"{metric}_delta")) for row in delta_rows for metric in METRICS):
        errors.append("delta_vs_statistical contains empty or NaN deltas")

    passed = not errors
    report = {
        "summary_dir": root.as_posix(),
        "expected_runs": args.expected_runs,
        "all_run_rows": len(rows),
        "delta_rows": len(delta_rows),
        "duplicate_rows": [list(key) for key in duplicates],
        "invalid_metrics": invalid_metrics,
        "passed": passed,
        "errors": errors,
    }
    lines = [
        "# Stage 5C Smoke Aggregation Check",
        "",
        f"- Summary directory: `{root.as_posix()}`",
        f"- Expected smoke runs: {args.expected_runs}",
        f"- Found all-run rows: {len(rows)}",
        f"- Delta rows: {len(delta_rows)}",
        f"- Duplicate rows: {len(duplicates)}",
        f"- Empty/NaN metrics: {len(invalid_metrics)}",
        f"- Status: {'PASS' if passed else 'FAILED'}",
    ]
    if errors:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in errors)
    lines.append("")
    write_output(args.report_json, report, args.overwrite)
    write_output(args.report_md, "\n".join(lines), args.overwrite)
    print("PASS" if passed else "FAILED")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
