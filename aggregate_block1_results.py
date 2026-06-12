import argparse
import csv
from pathlib import Path

from block1_utils import get_block1_metrics_path, normalize_missing_rate, read_json, write_json


METRIC_NAMES = ["NMI", "ARI", "ACC", "Purity"]


def _parse_args():
    parser = argparse.ArgumentParser(description="Aggregate Stage 5A Block 1 results.")
    parser.add_argument("--results-dir", type=str, default="results/block1")
    parser.add_argument("--datasets", nargs="+", default=["BDGP"])
    parser.add_argument("--missing-rates", nargs="+", type=float, default=[50])
    parser.add_argument("--methods", nargs="+", default=["llm_gimvc", "statistical_only"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--missing-pattern", type=str, default="mcar")
    parser.add_argument("--output-dir", type=str, default="results/block1_summary")
    return parser.parse_args()


def _mean(values):
    return sum(values) / len(values) if values else None


def _std(values):
    if len(values) <= 1:
        return 0.0 if values else None
    mean = _mean(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


def _write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _collect_runs(args):
    all_rows = []
    missing_runs = []
    for dataset in args.datasets:
        for missing_rate in args.missing_rates:
            rate = normalize_missing_rate(missing_rate)
            for method in args.methods:
                for seed in args.seeds:
                    metrics_path = get_block1_metrics_path(
                        args.results_dir,
                        dataset,
                        args.missing_pattern,
                        rate["percent"],
                        method,
                        seed,
                    )
                    metrics_obj = read_json(metrics_path)
                    if metrics_obj is None:
                        missing_runs.append(
                            {
                                "dataset": dataset,
                                "missing_rate": rate["percent"],
                                "method": method,
                                "seed": seed,
                                "expected_metrics_path": metrics_path.as_posix(),
                            }
                        )
                        continue
                    metrics = metrics_obj.get("metrics", {})
                    row = {
                        "dataset": dataset,
                        "missing_pattern": args.missing_pattern,
                        "missing_rate": rate["percent"],
                        "method": method,
                        "seed": seed,
                        "metrics_path": metrics_path.as_posix(),
                        "debug_only": bool(metrics_obj.get("debug_only", False)),
                    }
                    for metric_name in METRIC_NAMES:
                        row[metric_name] = metrics.get(metric_name)
                    all_rows.append(row)
    return all_rows, missing_runs


def _mean_std_rows(args, all_rows):
    rows = []
    for dataset in args.datasets:
        for missing_rate in args.missing_rates:
            rate = normalize_missing_rate(missing_rate)
            for method in args.methods:
                group = [
                    row
                    for row in all_rows
                    if row["dataset"] == dataset
                    and int(row["missing_rate"]) == rate["percent"]
                    and row["method"] == method
                ]
                found_seeds = {int(row["seed"]) for row in group}
                missing_seeds = [seed for seed in args.seeds if seed not in found_seeds]
                out = {
                    "dataset": dataset,
                    "missing_pattern": args.missing_pattern,
                    "missing_rate": rate["percent"],
                    "method": method,
                    "num_seeds": len(group),
                    "missing_seeds": " ".join(str(seed) for seed in missing_seeds),
                }
                for metric_name in METRIC_NAMES:
                    values = [float(row[metric_name]) for row in group if row.get(metric_name) is not None]
                    out[f"{metric_name}_mean"] = _mean(values)
                    out[f"{metric_name}_std"] = _std(values)
                rows.append(out)
    return rows


def _delta_rows(args, mean_rows):
    by_key = {
        (row["dataset"], int(row["missing_rate"]), row["method"]): row
        for row in mean_rows
    }
    rows = []
    for dataset in args.datasets:
        for missing_rate in args.missing_rates:
            rate = normalize_missing_rate(missing_rate)
            stat = by_key.get((dataset, rate["percent"], "statistical_only"))
            missing_baseline = stat is None or stat.get("NMI_mean") is None
            for method in args.methods:
                if method == "statistical_only":
                    continue
                method_row = by_key.get((dataset, rate["percent"], method))
                row = {
                    "dataset": dataset,
                    "missing_pattern": args.missing_pattern,
                    "missing_rate": rate["percent"],
                    "method": method,
                    "reference_method": "statistical_only",
                    "missing_baseline": bool(missing_baseline),
                    "method_NMI_mean": method_row.get("NMI_mean") if method_row else None,
                    "statistical_only_NMI_mean": stat.get("NMI_mean") if stat else None,
                }
                for metric_name in METRIC_NAMES:
                    method_value = method_row.get(f"{metric_name}_mean") if method_row else None
                    stat_value = stat.get(f"{metric_name}_mean") if stat else None
                    row[f"{metric_name}_delta"] = (
                        None if method_value is None or stat_value is None else float(method_value) - float(stat_value)
                    )
                rows.append(row)
    return rows


def main():
    args = _parse_args()
    output_dir = Path(args.output_dir)
    all_runs_csv = output_dir / "block1_all_runs.csv"
    mean_std_csv = output_dir / "block1_mean_std.csv"
    delta_csv = output_dir / "block1_delta_vs_statistical.csv"
    summary_json = output_dir / "block1_summary.json"

    print("=" * 60)
    print("Block 1 Aggregation - Stage 5A")
    print("=" * 60)
    all_rows, missing_runs = _collect_runs(args)
    mean_rows = _mean_std_rows(args, all_rows)
    delta_rows = _delta_rows(args, mean_rows)

    _write_csv(
        all_runs_csv,
        all_rows,
        ["dataset", "missing_pattern", "missing_rate", "method", "seed", *METRIC_NAMES, "metrics_path", "debug_only"],
    )
    _write_csv(
        mean_std_csv,
        mean_rows,
        [
            "dataset",
            "missing_pattern",
            "missing_rate",
            "method",
            "NMI_mean",
            "NMI_std",
            "ARI_mean",
            "ARI_std",
            "ACC_mean",
            "ACC_std",
            "Purity_mean",
            "Purity_std",
            "num_seeds",
            "missing_seeds",
        ],
    )
    _write_csv(
        delta_csv,
        delta_rows,
        [
            "dataset",
            "missing_pattern",
            "missing_rate",
            "method",
            "reference_method",
            "NMI_delta",
            "ARI_delta",
            "ACC_delta",
            "Purity_delta",
            "method_NMI_mean",
            "statistical_only_NMI_mean",
            "missing_baseline",
        ],
    )

    num_expected = len(args.datasets) * len(args.missing_rates) * len(args.methods) * len(args.seeds)
    summary = {
        "stage": "stage5a_block1_aggregation",
        "results_dir": args.results_dir,
        "output_dir": output_dir.as_posix(),
        "datasets": args.datasets,
        "missing_rates": [normalize_missing_rate(x)["percent"] for x in args.missing_rates],
        "methods": args.methods,
        "seeds": args.seeds,
        "num_expected_runs": num_expected,
        "num_found_runs": len(all_rows),
        "num_missing_runs": len(missing_runs),
        "missing_runs": missing_runs,
        "outputs": {
            "all_runs_csv": all_runs_csv.as_posix(),
            "mean_std_csv": mean_std_csv.as_posix(),
            "delta_csv": delta_csv.as_posix(),
            "summary_json": summary_json.as_posix(),
        },
    }
    write_json(summary_json, summary)

    print(f"found runs: {len(all_rows)}")
    print(f"missing runs: {len(missing_runs)}")
    print(f"saved all runs: {all_runs_csv.as_posix()}")
    print(f"saved mean/std: {mean_std_csv.as_posix()}")
    print(f"saved delta table: {delta_csv.as_posix()}")
    print(f"saved summary: {summary_json.as_posix()}")


if __name__ == "__main__":
    main()
