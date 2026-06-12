from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import shlex
import sys

from .adapters import adapt_external_result


@dataclass
class BaselineJob:
    method: str
    dataset: str
    missing_rate: int
    missing_rate_fraction: float
    missing_pattern: str
    seed: int
    data_path: Path
    metadata_path: Path
    raw_output_dir: Path
    block1_job_dir: Path
    device: str
    config: dict


@dataclass
class BaselineRunResult:
    method: str
    ok: bool
    returncode: int
    command: str
    raw_output_dir: Path
    stdout_tail: str
    stderr_tail: str
    error: Optional[str] = None


class ExternalBaselineAdapter:
    method_name: str

    def build_command(self, job: BaselineJob) -> list[str]:
        raise NotImplementedError

    def validate_environment(self, job: BaselineJob) -> None:
        raise NotImplementedError

    def find_outputs(self, job: BaselineJob) -> dict:
        raise NotImplementedError

    def adapt_results(self, job: BaselineJob) -> dict:
        raise NotImplementedError


class CommandTemplateBaselineAdapter(ExternalBaselineAdapter):
    display_name: str = "external baseline"

    def _method_cfg(self, job: BaselineJob):
        return job.config["ExternalBaselines"][self.method_name]

    def _repo_dir(self, job: BaselineJob):
        cfg = self._method_cfg(job)
        repo_root = Path(job.config["ExternalBaselines"].get("repo_root", "external_baselines"))
        repo_dir = Path(cfg["repo_dir"])
        if not repo_dir.is_absolute() and repo_dir.parts and repo_dir.parts[0] == "external_baselines":
            repo_dir = repo_root / Path(*repo_dir.parts[1:])
        return repo_dir

    def _entrypoint(self, job: BaselineJob):
        cfg = self._method_cfg(job)
        return self._repo_dir(job) / cfg["entrypoint"]

    def validate_environment(self, job: BaselineJob) -> None:
        repo_dir = self._repo_dir(job)
        entrypoint = self._entrypoint(job)
        if not repo_dir.exists():
            raise FileNotFoundError(
                f"External baseline repo for {self.display_name} was not found at {repo_dir.as_posix()}. "
                f"Please clone or place the official implementation there, or override --repo-root."
            )
        if not entrypoint.exists():
            raise FileNotFoundError(
                f"Entrypoint {entrypoint.name} was not found under {repo_dir.as_posix()}."
            )

    def build_command(self, job: BaselineJob) -> list[str]:
        cfg = self._method_cfg(job)
        entrypoint = Path(cfg["entrypoint"])
        values = {
            "python": sys.executable,
            "entrypoint": entrypoint.as_posix(),
            "data_path": job.data_path.resolve().as_posix(),
            "raw_output_dir": job.raw_output_dir.resolve().as_posix(),
            "dataset": job.dataset,
            "missing_rate": job.missing_rate,
            "missing_rate_fraction": job.missing_rate_fraction,
            "missing_pattern": job.missing_pattern,
            "seed": job.seed,
            "device": job.device,
        }
        command = cfg["command_template"].format(**values)
        return shlex.split(command, posix=False)

    def find_outputs(self, job: BaselineJob) -> dict:
        outputs = {}
        checks = {
            "metrics_json": ["metrics.json", "result.json"],
            "metrics_csv": ["metrics.csv", "result.csv"],
            "pred_labels": ["pred_labels.npy", "y_pred.npy", "labels_pred.npy", "pred.csv", "y_pred.csv"],
            "result_mat": ["result.mat"],
        }
        for key, filenames in checks.items():
            for filename in filenames:
                path = job.raw_output_dir / filename
                if path.exists():
                    outputs[key] = path.as_posix()
                    break
        return outputs

    def adapt_results(self, job: BaselineJob) -> dict:
        return adapt_external_result(self.method_name, job.data_path, job.raw_output_dir)
