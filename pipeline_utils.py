import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def normalize_missing_rate(missing_rate):
    """
    Accept either fraction or percent.

    Examples:
        0     -> percent=0,  fraction=0.0, dir_name=missing_0
        0.0   -> percent=0,  fraction=0.0, dir_name=missing_0
        10    -> percent=10, fraction=0.1, dir_name=missing_10
        0.1   -> percent=10, fraction=0.1, dir_name=missing_10
        50    -> percent=50, fraction=0.5, dir_name=missing_50
        0.5   -> percent=50, fraction=0.5, dir_name=missing_50
        90    -> percent=90, fraction=0.9, dir_name=missing_90
        0.9   -> percent=90, fraction=0.9, dir_name=missing_90
    """
    rate = float(missing_rate)
    if rate <= 1.0:
        fraction = rate
        percent = int(round(rate * 100))
    else:
        percent = int(round(rate))
        fraction = percent / 100.0

    return {
        "percent": percent,
        "fraction": fraction,
        "dir_name": f"missing_{percent}",
    }


def get_missing_rate_dir(missing_rate):
    return normalize_missing_rate(missing_rate)["dir_name"]


assert get_missing_rate_dir(0.5) == "missing_50"
assert get_missing_rate_dir(50) == "missing_50"
assert get_missing_rate_dir(0.9) == "missing_90"
assert get_missing_rate_dir(90) == "missing_90"


def get_stage1_dir(dataset, missing_rate, seed):
    return Path("outputs/statistical_path") / dataset / get_missing_rate_dir(missing_rate) / f"seed_{seed}"


def get_stage2a_dir(dataset, missing_rate, seed):
    return Path("outputs/llm_semantic_path") / dataset / get_missing_rate_dir(missing_rate) / f"seed_{seed}"


def get_fusion_dir(dataset, missing_rate, seed, gate_mode):
    return (
        Path("outputs/fusion_gate")
        / dataset
        / get_missing_rate_dir(missing_rate)
        / f"seed_{seed}"
        / gate_mode
    )


def get_clustering_head_dir(dataset, missing_rate, seed, gate_mode, input_source, head_type, representation):
    return (
        Path("outputs/clustering_head")
        / dataset
        / get_missing_rate_dir(missing_rate)
        / f"seed_{seed}"
        / gate_mode
        / input_source
        / head_type
        / representation
    )


def get_method_dir(dataset, missing_rate, seed, gate_mode, input_source, head_type, representation):
    return (
        Path("outputs/llm_gimvc_method")
        / dataset
        / get_missing_rate_dir(missing_rate)
        / f"seed_{seed}"
        / gate_mode
        / input_source
        / head_type
        / representation
    )


def _check_files(root, filenames):
    root = Path(root)
    missing = [name for name in filenames if not (root / name).exists()]
    return {
        "complete": len(missing) == 0,
        "missing": missing,
        "dir": root.as_posix(),
    }


def stage1_complete(stage1_dir):
    return _check_files(stage1_dir, ["model.pt", "y_stat.pt"])


def stage2a_complete(stage2a_dir):
    return _check_files(stage2a_dir, ["y_llm.pt", "c_llm.pt", "s_cons.pt", "query_mask.pt", "run_summary.json"])


def fusion_complete(fusion_dir):
    return _check_files(fusion_dir, ["y_final.pt", "fusion_summary.json"])


def clustering_head_complete(head_dir):
    return _check_files(head_dir, ["metrics.json", "head_summary.json", "model.pt"])


def read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def make_json_safe(obj):
    try:
        import torch
    except Exception:
        torch = None

    if isinstance(obj, dict):
        return {str(key): make_json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(value) for value in obj]
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch is not None and torch.is_tensor(obj):
        if obj.numel() == 1:
            return obj.detach().cpu().item()
        return obj.detach().cpu().tolist()
    return obj


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(obj), f, indent=2, ensure_ascii=False)


def _command_to_text(cmd):
    if isinstance(cmd, str):
        return cmd
    return subprocess.list2cmdline([str(part) for part in cmd])


def _secret_values(env):
    values = []
    source = dict(os.environ)
    if env:
        source.update(env)
    for key, value in source.items():
        key_upper = key.upper()
        if any(token in key_upper for token in ["KEY", "TOKEN", "SECRET", "PASSWORD"]):
            if value and len(str(value)) >= 8:
                values.append(str(value))
    return values


def _sanitize_text(text, secrets):
    if text is None:
        return ""
    text = str(text)
    for secret in secrets:
        text = text.replace(secret, "[REDACTED]")
    return text


def run_command(cmd, cwd=None, env=None, fail_fast=True, timeout_seconds=None):
    """
    Execute a stage command and return a compact command log record.
    """
    secrets = _secret_values(env)
    cmd_text = _sanitize_text(_command_to_text(cmd), secrets)
    run_env = os.environ.copy()
    if env:
        run_env.update(env)

    try:
        completed = subprocess.run(
            cmd,
            cwd=cwd,
            env=run_env,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        stdout_tail = _sanitize_text((exc.stdout or "")[-4000:], secrets)
        stderr_tail = _sanitize_text((exc.stderr or "")[-4000:], secrets)
        result = {
            "cmd": cmd_text,
            "returncode": -1,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
            "ok": False,
            "error": f"Command timed out after {timeout_seconds} seconds",
        }
        if fail_fast:
            raise RuntimeError(result["error"])
        return result
    stdout_tail = _sanitize_text(completed.stdout[-4000:], secrets)
    stderr_tail = _sanitize_text(completed.stderr[-4000:], secrets)
    result = {
        "cmd": cmd_text,
        "returncode": int(completed.returncode),
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
        "ok": completed.returncode == 0,
    }
    if fail_fast and completed.returncode != 0:
        raise RuntimeError(
            "Command failed: "
            f"{cmd_text}\nreturncode: {completed.returncode}\nstderr_tail:\n{stderr_tail}"
        )
    return result


def python_cmd(script_name, *args):
    return [sys.executable, script_name, *[str(arg) for arg in args]]
