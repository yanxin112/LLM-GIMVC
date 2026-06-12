#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path


def csv_values(value, cast=str):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Validate a Stage 5C smoke or full manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-total", type=int, required=True)
    parser.add_argument("--datasets", default="Reuters,BDGP,Wikipedia,Handwritten")
    parser.add_argument("--missing-rates", required=True)
    parser.add_argument("--methods", default="llm_gimvc,statistical_only,mica,jga_imvc,freecsl")
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--missing-pattern", default="MCAR")
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--report-md", required=True)
    parser.add_argument("--check-outputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_file(path, content, overwrite):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Report already exists: {path}. Pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, str):
        path.write_text(content, encoding="utf-8")
    else:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(content, handle, indent=2, ensure_ascii=False)


def valid_metrics(path):
    if not path.is_file():
        return False
    try:
        payload = load_json(path)
        values = payload.get("metrics", payload)
        return all(values.get(name) is not None for name in ("NMI", "ARI", "ACC", "Purity"))
    except Exception:
        return False


def main():
    args = parse_args()
    manifest = load_json(args.manifest)
    jobs = manifest.get("jobs", [])
    datasets = csv_values(args.datasets)
    rates = csv_values(args.missing_rates, int)
    methods = csv_values(args.methods)
    seeds = csv_values(args.seeds, int)
    expected_keys = {
        (dataset, rate, method, seed)
        for dataset in datasets
        for rate in rates
        for method in methods
        for seed in seeds
    }
    actual_keys = []
    errors = []
    output_checks = []
    for job in jobs:
        try:
            key = (
                str(job["dataset"]),
                int(job["missing_rate"]),
                str(job["method"]),
                int(job["seed"]),
            )
        except Exception as exc:
            errors.append(f"job missing identity fields: {job.get('id')}: {exc}")
            continue
        actual_keys.append(key)
        expected_output = (
            f"results/block1/{key[0]}/{args.missing_pattern}/missing_{key[1]}/"
            f"{key[2]}/seed_{key[3]}/metrics.json"
        )
        if job.get("expected_output") != expected_output:
            errors.append(
                f"{job.get('id')}: expected_output mismatch: "
                f"{job.get('expected_output')} != {expected_output}"
            )
        if args.check_outputs:
            output_checks.append(
                {
                    "job_id": job.get("id"),
                    "path": expected_output,
                    "valid": valid_metrics(Path(expected_output)),
                }
            )

    duplicate_keys = [list(key) for key, count in Counter(actual_keys).items() if count > 1]
    if duplicate_keys:
        errors.append(f"duplicate job combinations: {duplicate_keys}")
    missing_keys = sorted(expected_keys - set(actual_keys))
    unexpected_keys = sorted(set(actual_keys) - expected_keys)
    if len(jobs) != args.expected_total:
        errors.append(f"total_jobs mismatch: {len(jobs)} != {args.expected_total}")
    if missing_keys:
        errors.append(f"missing combinations: {len(missing_keys)}")
    if unexpected_keys:
        errors.append(f"unexpected combinations: {len(unexpected_keys)}")
    missing_outputs = [item for item in output_checks if not item["valid"]]
    if args.check_outputs and missing_outputs:
        errors.append(f"missing or invalid expected outputs: {len(missing_outputs)}")

    per_dataset = Counter(key[0] for key in actual_keys)
    passed = not errors
    payload = {
        "manifest": Path(args.manifest).as_posix(),
        "expected_total": args.expected_total,
        "actual_total": len(jobs),
        "passed": passed,
        "per_dataset": dict(sorted(per_dataset.items())),
        "duplicate_combinations": duplicate_keys,
        "missing_combinations": [list(key) for key in missing_keys],
        "unexpected_combinations": [list(key) for key in unexpected_keys],
        "output_checks": output_checks,
        "errors": errors,
    }
    lines = [
        "# Stage 5C Manifest Check",
        "",
        f"- Manifest: `{payload['manifest']}`",
        f"- Expected jobs: {args.expected_total}",
        f"- Actual jobs: {len(jobs)}",
        f"- Status: {'PASS' if passed else 'FAILED'}",
        f"- Per dataset: `{json.dumps(payload['per_dataset'], sort_keys=True)}`",
        f"- Duplicate combinations: {len(duplicate_keys)}",
        f"- Missing combinations: {len(missing_keys)}",
        f"- Unexpected combinations: {len(unexpected_keys)}",
    ]
    if args.check_outputs:
        lines.append(f"- Missing/invalid outputs: {len(missing_outputs)}")
    if errors:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in errors)
    lines.append("")
    write_file(args.report_json, payload, args.overwrite)
    write_file(args.report_md, "\n".join(lines), args.overwrite)
    print("PASS" if passed else "FAILED")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
