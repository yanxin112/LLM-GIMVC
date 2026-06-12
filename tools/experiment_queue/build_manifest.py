#!/usr/bin/env python3
import argparse
import itertools
import json
import re
from datetime import datetime, timezone
from pathlib import Path


PLACEHOLDER_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build an experiment manifest from a phase/grid JSON config.")
    parser.add_argument("--config", required=True, help="Grid specification JSON.")
    parser.add_argument("--output", required=True, help="Manifest JSON to create.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing manifest.")
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def render(value, variables):
    if not isinstance(value, str):
        return value

    def replace(match):
        key = match.group(1)
        if key not in variables:
            raise KeyError(f"Template variable '{key}' has no grid value.")
        return str(variables[key])

    return PLACEHOLDER_RE.sub(replace, value)


def expand_phase(phase):
    name = phase.get("name")
    grid = phase.get("grid")
    template = phase.get("template")
    if not name or not isinstance(grid, dict) or not isinstance(template, dict):
        raise ValueError("Each phase requires name, grid, and template objects.")
    if not grid:
        raise ValueError(f"Phase '{name}' has an empty grid.")

    keys = list(grid)
    values = []
    for key in keys:
        entries = grid[key]
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"Phase '{name}' grid field '{key}' must be a non-empty list.")
        values.append(entries)

    jobs = []
    for combination in itertools.product(*values):
        variables = dict(zip(keys, combination))
        job = {key: render(value, variables) for key, value in template.items()}
        job.update(variables)
        job["phase"] = name
        required = ["id", "cmd", "expected_output"]
        missing = [key for key in required if not job.get(key)]
        if missing:
            raise ValueError(f"Phase '{name}' produced a job missing fields: {missing}")
        jobs.append(job)
    return jobs


def main():
    args = parse_args()
    config_path = Path(args.config)
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Manifest already exists: {output_path}. Pass --overwrite to replace it.")

    config = load_json(config_path)
    phases = config.get("phases")
    if not isinstance(phases, list) or not phases:
        raise ValueError("Config must contain a non-empty 'phases' list.")

    jobs = []
    for phase in phases:
        jobs.extend(expand_phase(phase))
    job_ids = [job["id"] for job in jobs]
    if len(job_ids) != len(set(job_ids)):
        raise ValueError("Manifest contains duplicate job ids.")

    passthrough = {
        key: value
        for key, value in config.items()
        if key not in {"phases"}
    }
    manifest = {
        **passthrough,
        "config_path": config_path.as_posix(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_jobs": len(jobs),
        "jobs": jobs,
    }
    write_json(output_path, manifest)
    print(f"built manifest: {output_path.as_posix()}")
    print(f"total jobs: {len(jobs)}")


if __name__ == "__main__":
    main()
