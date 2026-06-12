#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


METHODS = ("llm_gimvc", "statistical_only", "mica", "jga_imvc", "freecsl")
METRICS = ("NMI", "ARI", "ACC", "Purity")


def parse_args():
    parser = argparse.ArgumentParser(description="Build Stage 5C paper tables and final report.")
    parser.add_argument("--summary-dir", default="results/block1_summary")
    parser.add_argument("--expected-jobs", type=int, default=600)
    parser.add_argument("--report-json", default="refine-logs/stage5c/stage5c_final_report.json")
    parser.add_argument("--report-md", default="refine-logs/stage5c/stage5c_final_report.md")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fieldnames, overwrite):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}. Pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def formatted(mean_row, metric):
    mean = mean_row.get(f"{metric}_mean")
    std = mean_row.get(f"{metric}_std")
    if mean in (None, "") or std in (None, ""):
        return ""
    return f"{float(mean):.4f}+/-{float(std):.4f}"


def main():
    args = parse_args()
    root = Path(args.summary_dir)
    all_runs_path = root / "block1_all_runs.csv"
    mean_path = root / "block1_mean_std.csv"
    delta_path = root / "block1_delta_vs_statistical.csv"
    for path in (all_runs_path, mean_path, delta_path, root / "block1_summary.json"):
        if not path.is_file():
            raise FileNotFoundError(f"Required aggregate output missing: {path}")
    all_runs = read_csv(all_runs_path)
    mean_rows = read_csv(mean_path)
    delta_rows = read_csv(delta_path)
    by_mean = {
        (row["dataset"], int(float(row["missing_rate"])), row["method"]): row
        for row in mean_rows
    }
    by_delta = {
        (row["dataset"], int(float(row["missing_rate"])), row["method"]): row
        for row in delta_rows
    }
    dataset_rates = sorted({(row["dataset"], int(float(row["missing_rate"]))) for row in mean_rows})
    table_paths = {}
    for metric in METRICS:
        rows = []
        for dataset, rate in dataset_rates:
            row = {"dataset": dataset, "missing_rate": rate}
            for method in METHODS:
                row[f"{method} mean+/-std"] = formatted(by_mean.get((dataset, rate, method), {}), metric)
            delta = by_delta.get((dataset, rate, "llm_gimvc"), {}).get(f"{metric}_delta")
            row["delta_vs_statistical"] = "" if delta in (None, "") else float(delta)
            rows.append(row)
        path = root / f"block1_paper_table_{metric.lower()}.csv"
        fields = [
            "dataset",
            "missing_rate",
            *[f"{method} mean+/-std" for method in METHODS],
            "delta_vs_statistical",
        ]
        write_csv(path, rows, fields, args.overwrite)
        table_paths[metric] = path.as_posix()

    success = len(all_runs)
    failed = max(args.expected_jobs - success, 0)
    complete_five_seeds = all(int(row.get("num_seeds") or 0) == 5 for row in mean_rows)
    high_missing = [
        row for row in delta_rows
        if row["method"] == "llm_gimvc" and int(float(row["missing_rate"])) in (50, 70, 90)
    ]
    nmi_deltas = [float(row["NMI_delta"]) for row in high_missing if row.get("NMI_delta") not in (None, "")]
    supports_claim = bool(nmi_deltas) and complete_five_seeds and all(value > 0 for value in nmi_deltas)
    report = {
        "expected_jobs": args.expected_jobs,
        "successful_jobs": success,
        "failed_or_missing_jobs": failed,
        "complete_five_seeds": complete_five_seeds,
        "high_missing_llm_vs_statistical": high_missing,
        "supports_high_missing_claim": supports_claim,
        "paper_tables": table_paths,
    }
    lines = [
        "# Stage 5C Final Report",
        "",
        f"- Total expected jobs: {args.expected_jobs}",
        f"- Successful jobs: {success}",
        f"- Failed or missing jobs: {failed}",
        f"- All five seeds complete: {complete_five_seeds}",
        f"- High-missing-rate claim supported: {supports_claim}",
        "",
        "## Dataset / Missing Rate / Method Mean and Standard Deviation",
        "",
        "| dataset | missing_rate | method | NMI mean+/-std | ARI mean+/-std | ACC mean+/-std | Purity mean+/-std | seeds |",
        "|---|---:|---|---|---|---|---|---:|",
    ]
    for row in mean_rows:
        lines.append(
            f"| {row['dataset']} | {row['missing_rate']} | {row['method']} | "
            f"{formatted(row, 'NMI')} | {formatted(row, 'ARI')} | "
            f"{formatted(row, 'ACC')} | {formatted(row, 'Purity')} | {row.get('num_seeds', '')} |"
        )
    lines.extend(
        [
            "",
            "## LLM-GIMVC vs Statistical-Only at 50/70/90 Missing Rates",
            "",
            "| dataset | missing_rate | NMI delta | ARI delta | ACC delta | Purity delta |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in high_missing:
        lines.append(
            f"| {row['dataset']} | {row['missing_rate']} | {row.get('NMI_delta', '')} | "
            f"{row.get('ARI_delta', '')} | {row.get('ACC_delta', '')} | {row.get('Purity_delta', '')} |"
        )
    lines.extend(
        [
            "",
            "Stable conclusions require all five seeds for every matrix cell.",
            "External baseline results must not enter the main paper table unless independent verification passes.",
            "",
        ]
    )
    write_output(args.report_json, report, args.overwrite)
    write_output(args.report_md, "\n".join(lines), args.overwrite)
    print(f"paper tables written: {len(table_paths)}")
    if not complete_five_seeds or failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
