#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from llm_gimvc.baselines.data_export import export_block1_data_bundle


REQUIRED_KEYS = {
    "complete_views",
    "masked_views",
    "available_mask",
    "missing_mask",
    "labels",
    "metadata",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Export a unified Stage 5B external-baseline data bundle.")
    parser.add_argument("--dataset", default="BDGP")
    parser.add_argument("--missing-pattern", default="MCAR")
    parser.add_argument("--missing-rate", type=float, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", default="results/baseline_data")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_bundle(path):
    with np.load(path, allow_pickle=True) as bundle:
        missing = sorted(REQUIRED_KEYS - set(bundle.files))
        if missing:
            raise ValueError(f"Exported bundle is missing required keys: {missing}")
        labels = np.asarray(bundle["labels"]).reshape(-1)
        available_mask = np.asarray(bundle["available_mask"])
        missing_mask = np.asarray(bundle["missing_mask"])
        complete_views = list(bundle["complete_views"])
        masked_views = list(bundle["masked_views"])
        metadata = json.loads(str(np.asarray(bundle["metadata"]).item()))
    if available_mask.shape != missing_mask.shape:
        raise ValueError("available_mask and missing_mask shapes differ.")
    if not np.array_equal(1.0 - available_mask, missing_mask):
        raise ValueError("missing_mask is not the complement of available_mask.")
    if len(complete_views) != len(masked_views):
        raise ValueError("complete_views and masked_views have different view counts.")
    if any(np.asarray(view).shape[0] != labels.size for view in complete_views + masked_views):
        raise ValueError("At least one view has a sample count different from labels.")
    return {
        "keys": sorted(REQUIRED_KEYS),
        "num_samples": int(labels.size),
        "num_views": len(complete_views),
        "view_dims": [int(np.asarray(view).shape[1]) for view in complete_views],
        "mask_shape": list(available_mask.shape),
        "metadata": metadata,
    }


def main():
    args = parse_args()
    if args.missing_pattern.upper() != "MCAR":
        raise ValueError("The current project loader supports the Stage 5B export only for MCAR.")
    result = export_block1_data_bundle(
        dataset=args.dataset,
        missing_rate=args.missing_rate,
        missing_pattern=args.missing_pattern,
        seed=args.seed,
        output_root=args.output_root,
        export_format="npz",
        device=args.device,
        force=args.overwrite,
    )
    validation = validate_bundle(result["data_path"])
    print(f"data path: {result['data_path'].as_posix()}")
    print(f"metadata path: {result['metadata_path'].as_posix()}")
    print(f"reused: {str(result['reused']).lower()}")
    print(json.dumps(validation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
