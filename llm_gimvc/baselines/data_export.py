import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np
import torch

from block1_utils import normalize_missing_rate, write_json
from configure import get_default_config
from datasets import get_loader


def get_baseline_data_dir(data_root, dataset, missing_pattern, missing_rate, seed):
    rate = normalize_missing_rate(missing_rate)
    return Path(data_root) / dataset / missing_pattern / rate["dir_name"] / f"seed_{seed}"


def _infer_config_from_data(config, features, labels):
    config["Module"]["in_dim"] = [int(feature.shape[1]) for feature in features]
    config["Dataset"]["num_views"] = len(features)
    config["Dataset"]["num_sample"] = int(features[0].shape[0])
    config["Dataset"]["num_classes"] = int(np.unique(labels).size)
    return config


def _object_array(values):
    array = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        array[index] = value
    return array


def _maybe_export_mat(path, x_list, x_missing_list, labels, available_mask, missing_mask, num_clusters):
    try:
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            import scipy.io as sio

            sio.savemat(
                path,
                {
                    "X": np.asarray(x_list, dtype=object),
                    "X_missing": np.asarray(x_missing_list, dtype=object),
                    "labels": labels,
                    "available_mask": available_mask,
                    "missing_mask": missing_mask,
                    "num_clusters": int(num_clusters),
                },
            )
        return True, None
    except Exception as exc:
        return False, str(exc)


def export_block1_data_bundle(
    dataset,
    missing_rate,
    missing_pattern,
    seed,
    output_root="results/baseline_data",
    export_format="npz",
    device="cpu",
    force=False,
):
    rate = normalize_missing_rate(missing_rate)
    output_dir = get_baseline_data_dir(output_root, dataset, missing_pattern, rate["percent"], seed)
    data_path = output_dir / "data.npz"
    metadata_path = output_dir / "metadata.json"
    if data_path.exists() and metadata_path.exists() and not force:
        return {
            "output_dir": output_dir,
            "data_path": data_path,
            "metadata_path": metadata_path,
            "reused": True,
        }

    config = get_default_config(dataset)
    config["Dataset"]["name"] = dataset
    config["Dataset"]["missing_rate"] = rate["fraction"]
    config["training"]["seed"] = int(seed)
    load_device = torch.device("cpu" if device == "cpu" or not torch.cuda.is_available() else device)
    loader, features, labels, inc_mask, masked_x = get_loader(config, load_device)
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    config = _infer_config_from_data(config, features, labels)

    available_mask = np.asarray(inc_mask, dtype=np.float32)
    missing_mask = 1.0 - available_mask
    x_list = [np.asarray(feature, dtype=np.float32) for feature in features]
    x_missing_list = [np.asarray(view, dtype=np.float32) for view in masked_x]
    num_views = len(x_list)
    num_samples = int(x_list[0].shape[0])
    view_dims = [int(x.shape[1]) for x in x_list]
    num_clusters = int(np.unique(labels).size)
    metadata = {
        "schema_version": "stage5b_baseline_data_v2",
        "dataset": dataset,
        "missing_pattern": missing_pattern,
        "missing_rate": rate["percent"],
        "missing_rate_fraction": rate["fraction"],
        "seed": int(seed),
        "num_samples": num_samples,
        "num_views": num_views,
        "num_clusters": num_clusters,
        "view_dims": view_dims,
        "label_usage": (
            "labels are included only for post-hoc evaluation and must not be passed "
            "to external baseline training or inference"
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    npz_payload = {
        "complete_views": _object_array(x_list),
        "masked_views": _object_array(x_missing_list),
        "labels": labels,
        "available_mask": available_mask,
        "missing_mask": missing_mask,
        "metadata": np.asarray(json.dumps(metadata, sort_keys=True)),
        "num_views": np.asarray(num_views, dtype=np.int64),
        "num_samples": np.asarray(num_samples, dtype=np.int64),
        "num_clusters": np.asarray(num_clusters, dtype=np.int64),
        "view_dims": np.asarray(view_dims, dtype=np.int64),
        "dataset": np.asarray(dataset),
        "missing_rate": np.asarray(rate["percent"], dtype=np.int64),
        "missing_rate_fraction": np.asarray(rate["fraction"], dtype=np.float32),
        "missing_pattern": np.asarray(missing_pattern),
        "seed": np.asarray(seed, dtype=np.int64),
    }
    for view_idx, (x, x_missing) in enumerate(zip(x_list, x_missing_list)):
        npz_payload[f"X_{view_idx}"] = x
        npz_payload[f"X_missing_{view_idx}"] = x_missing
    np.savez_compressed(data_path, **npz_payload)

    mat_path = output_dir / "data.mat"
    mat_exported = False
    mat_export_error = None
    if export_format in ["mat", "npz"]:
        mat_exported, mat_export_error = _maybe_export_mat(
            mat_path,
            x_list,
            x_missing_list,
            labels,
            available_mask,
            missing_mask,
            num_clusters,
        )

    metadata = {
        **metadata,
        "files": {
            "npz": "data.npz",
            "mat": "data.mat" if mat_exported else None,
        },
        "mat_exported": bool(mat_exported),
        "mat_export_error": mat_export_error,
    }
    write_json(metadata_path, metadata)
    return {
        "output_dir": output_dir,
        "data_path": data_path,
        "metadata_path": metadata_path,
        "reused": False,
    }


def _parse_args():
    parser = argparse.ArgumentParser(description="Export a Block 1 baseline data bundle.")
    parser.add_argument("--dataset", type=str, default="BDGP")
    parser.add_argument("--missing-rate", type=float, default=50)
    parser.add_argument("--missing-pattern", type=str, default="mcar")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", type=str, default="results/baseline_data")
    parser.add_argument("--export-format", choices=["npz", "mat"], default="npz")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    result = export_block1_data_bundle(
        dataset=args.dataset,
        missing_rate=args.missing_rate,
        missing_pattern=args.missing_pattern,
        seed=args.seed,
        output_root=args.output_root,
        export_format=args.export_format,
        device=args.device,
        force=args.force,
    )
    print(f"data path: {result['data_path'].as_posix()}")
    print(f"metadata path: {result['metadata_path'].as_posix()}")
    print(f"reused: {str(result['reused']).lower()}")


if __name__ == "__main__":
    main()
