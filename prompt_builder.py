import torch


def _format_float(value):
    return f"{float(value):.4f}"


def summarize_latent_view(z, top_k=8):
    z = z.detach().float().cpu()
    top_k = min(top_k, z.numel())
    top_values, top_indices = torch.topk(torch.abs(z), k=top_k)
    return {
        "mean": float(torch.mean(z).item()),
        "std": float(torch.std(z, unbiased=False).item()),
        "min": float(torch.min(z).item()),
        "max": float(torch.max(z).item()),
        "top_abs_indices": [int(idx) for idx in top_indices.tolist()],
        "top_abs_values": [float(value) for value in top_values.tolist()],
    }


class FixedPromptBuilder:
    def __init__(self, dataset_name, num_views, prompt_mode="fixed_template"):
        if prompt_mode != "fixed_template":
            raise ValueError(f"Unsupported prompt_mode for Stage 2A: {prompt_mode}")
        self.dataset_name = dataset_name
        self.num_views = num_views
        self.prompt_mode = prompt_mode

    def _format_summary(self, view_idx, summary):
        top_indices = ", ".join(str(idx) for idx in summary["top_abs_indices"])
        top_values = ", ".join(_format_float(value) for value in summary["top_abs_values"])
        return (
            f"- view {view_idx}: "
            f"mean={_format_float(summary['mean'])}, "
            f"std={_format_float(summary['std'])}, "
            f"min={_format_float(summary['min'])}, "
            f"max={_format_float(summary['max'])}, "
            f"top_abs_indices=[{top_indices}], "
            f"top_abs_values=[{top_values}]"
        )

    def build_prompt(
        self,
        sample_index,
        target_view,
        available_view_indices,
        available_view_summaries,
        dataset_metadata=None,
    ):
        metadata = dataset_metadata or {}
        metadata_lines = []
        for key in sorted(metadata):
            metadata_lines.append(f"{key}: {metadata[key]}")
        metadata_block = "\n".join(metadata_lines) if metadata_lines else "None"

        summary_lines = []
        for view_idx in available_view_indices:
            summary_lines.append(self._format_summary(view_idx, available_view_summaries[view_idx]))
        summaries = "\n".join(summary_lines) if summary_lines else "- none"

        return "\n".join(
            [
                "You are a semantic missing-view recovery assistant for incomplete multi-view clustering.",
                "",
                f"Dataset: {self.dataset_name}",
                f"Sample index: {int(sample_index)}",
                f"Target missing view: view {int(target_view)}",
                f"Available views: {[int(view_idx) for view_idx in available_view_indices]}",
                "",
                "Dataset metadata:",
                metadata_block,
                "",
                "Available view summaries:",
                summaries,
                "",
                "Task:",
                "Infer a concise semantic description for the missing target view.",
                "Return strict JSON with this schema:",
                (
                    '{"recovered_text": "...", "semantic_label": "...", '
                    '"confidence": 0.0, "rationale": "...", "should_abstain": false}'
                ),
            ]
        )
