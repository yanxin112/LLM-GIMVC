import argparse
import json
from pathlib import Path

import numpy as np
import torch

from configure import get_default_config
from datasets import get_loader
from embedding_providers import build_embedding_provider
from llm_providers import build_llm_provider
from llm_semantic_path import LLMSemanticViewRecovery, format_missing_rate
from model import get_statistical_generator
from prompt_builder import FixedPromptBuilder
from util import set_seed


def _device_from_arg(device_arg):
    if device_arg == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_arg)


def _infer_config_from_data(config, features, labels):
    config["Module"]["in_dim"] = [feature.shape[1] for feature in features]
    config["Dataset"]["num_views"] = len(features)
    config["Dataset"]["num_sample"] = features[0].shape[0]
    config["Dataset"]["num_classes"] = int(np.unique(labels).size)
    config["LLM"]["output_dim"] = int(config["Module"].get("trans_dim", config["Module"].get("d_model", 512)))
    if config.get("Embedding", {}).get("target_dim") is None:
        config["Embedding"]["target_dim"] = config["LLM"]["output_dim"]
    return config


def _stage1_model_path(config, explicit_path=None):
    if explicit_path is not None:
        return Path(explicit_path)

    dataset = config["Dataset"]["name"]
    rate = format_missing_rate(config["Dataset"]["missing_rate"])
    seed = config["training"]["seed"]
    path = Path("outputs") / "statistical_path" / dataset / f"missing_{rate}" / f"seed_{seed}" / "model.pt"
    if path.exists():
        return path

    legacy_rate = config["Dataset"]["missing_rate"]
    legacy_path = (
        Path("outputs")
        / "statistical_path"
        / dataset
        / f"missing_{legacy_rate}"
        / f"seed_{seed}"
        / "model.pt"
    )
    if legacy_path.exists():
        print(f"Using legacy Stage 1 path missing_{legacy_rate}. Please migrate to missing_{rate}.")
        return legacy_path

    return path


def _base_output_dir(config):
    dataset = config["Dataset"]["name"]
    rate = format_missing_rate(config["Dataset"]["missing_rate"])
    seed = config["training"]["seed"]
    return Path("outputs") / "llm_semantic_path" / dataset / f"missing_{rate}" / f"seed_{seed}"


def _output_dir(config):
    base_dir = _base_output_dir(config)
    max_samples = config["LLM"].get("max_samples")
    if max_samples is not None:
        return base_dir / f"partial_max_samples_{int(max_samples)}"
    return base_dir


def _is_sample_partial(config):
    return config["LLM"].get("max_samples") is not None


def _partial_warning_text(config, output_dir):
    reason = "max_samples" if _is_sample_partial(config) else "max_llm_queries/API failures"
    return "\n".join(
        [
            "This directory contains partial Stage 2A/2C outputs.",
            f"Partial reason: {reason}",
            "Do NOT use these tensors as Fusion Gate inputs unless you explicitly allow debug partial outputs.",
            "",
            f"Dataset: {config['Dataset']['name']}",
            f"Missing rate: {config['Dataset']['missing_rate']}",
            f"Seed: {config['training']['seed']}",
            f"Output directory: {output_dir.as_posix()}",
            "",
        ]
    )


def _write_partial_warning(output_dir, config, is_partial):
    if not is_partial:
        return None
    warning_path = output_dir / "PARTIAL_RUN_DO_NOT_USE_FOR_FUSION.txt"
    with open(warning_path, "w", encoding="utf-8") as f:
        f.write(_partial_warning_text(config, output_dir))
    return warning_path


def is_safe_stage2a_output(summary_path):
    with open(summary_path, "r", encoding="utf-8-sig") as f:
        summary = json.load(f)
    return bool(summary.get("safe_for_fusion", False)) and not bool(summary.get("is_partial", False))


def _load_stage1_model(config, model_path, device):
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    model_cfg = checkpoint.get("config", config)
    input_dims = checkpoint.get("input_dims") or config["Module"]["in_dim"]
    trans_dim = model_cfg["Module"].get("trans_dim", config["Module"]["trans_dim"])
    trans_layers = model_cfg["Module"].get("trans_layers", config["Module"]["trans_layers"])
    trans_headers = model_cfg["Module"].get("trans_headers", config["Module"]["trans_headers"])
    trans_dropout = model_cfg["Module"].get("trans_dropout", config["Module"]["trans_dropout"])
    model = get_statistical_generator(
        input_dims,
        d_model=trans_dim,
        n_layers=trans_layers,
        heads=trans_headers,
        dropout=trans_dropout,
        device=device,
    )
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    model.eval()
    return model


def _build_latent_fea(model, features, available_mask, device):
    all_x = [torch.from_numpy(feature).to(device) for feature in features]
    with torch.no_grad():
        return model.encode_views(all_x, available_mask)


def _write_records(records, records_path):
    with open(records_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")


def _load_inputs(config, device, stage1_model_path):
    set_seed(config["training"]["seed"])
    loader, features, labels, inc_mask, masked_x = get_loader(config, device)
    config = _infer_config_from_data(config, features, labels)
    available_mask = torch.from_numpy(inc_mask).to(device)
    stage1_model = _load_stage1_model(config, stage1_model_path, device)
    latent_fea = _build_latent_fea(stage1_model, features, available_mask, device)
    return config, latent_fea, available_mask


def _build_recovery(config, device, llm_provider=None, embedding_provider=None):
    llm_cfg = config["LLM"]
    emb_cfg = config["Embedding"]
    prompt_builder = FixedPromptBuilder(
        dataset_name=config["Dataset"]["name"],
        num_views=config["Dataset"]["num_views"],
        prompt_mode=llm_cfg["prompt_mode"],
    )
    if llm_provider is None:
        provider = str(llm_cfg["provider"]).lower()
        model = llm_cfg.get("model")
        if provider == "qwen" and (not model or model == "mock-llm"):
            model = llm_cfg.get("qwen_model", "qwen-plus")
        elif provider == "openai" and (not model or model == "mock-llm"):
            model = llm_cfg.get("openai_model", "gpt-4o-mini")
        elif not model:
            model = llm_cfg.get("llm_model", "mock-llm")
        llm_provider = build_llm_provider(
            provider=provider,
            model=model,
            temperature=llm_cfg.get("openai_temperature", llm_cfg.get("temperature", 0.0)),
            api_key_env=llm_cfg.get("openai_api_key_env", "OPENAI_API_KEY"),
            qwen_api_key_env=llm_cfg.get("qwen_api_key_env", "DASHSCOPE_API_KEY"),
            qwen_base_url=llm_cfg.get("qwen_base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            qwen_timeout=llm_cfg.get("qwen_timeout", 60),
            qwen_max_retries=llm_cfg.get("qwen_max_retries", 3),
            qwen_temperature=llm_cfg.get("qwen_temperature", 0.0),
            qwen_max_output_tokens=llm_cfg.get("qwen_max_output_tokens", 800),
            timeout=llm_cfg.get("openai_timeout", 60),
            max_retries=llm_cfg.get("openai_max_retries", 3),
            max_output_tokens=llm_cfg.get("openai_max_output_tokens", 800),
            use_structured_outputs=llm_cfg.get("use_structured_outputs", True),
            schema_version=llm_cfg.get("response_schema_version", "semantic_recovery_v1"),
            cache_root=llm_cfg.get("cache_root", "outputs/llm_cache"),
            use_cache=llm_cfg.get("use_cache", True),
            force_refresh_cache=llm_cfg.get("force_refresh_cache", False),
            cache_only=llm_cfg.get("cache_only", False),
        )
    if embedding_provider is None:
        emb_provider = str(emb_cfg["provider"]).lower()
        emb_model = emb_cfg.get("model")
        if emb_provider == "qwen" and (not emb_model or emb_model == "mock-embedding"):
            emb_model = emb_cfg.get("qwen_model", "text-embedding-v4")
        elif emb_provider == "openai" and (not emb_model or emb_model == "mock-embedding"):
            emb_model = emb_cfg.get("openai_model", "text-embedding-3-small")
        elif not emb_model:
            emb_model = emb_cfg.get("embedding_model", "mock-embedding")
        embedding_provider = build_embedding_provider(
            provider=emb_provider,
            model=emb_model,
            output_dim=llm_cfg["output_dim"],
            target_dim=emb_cfg.get("target_dim") or llm_cfg["output_dim"],
            api_key_env=emb_cfg.get("openai_api_key_env", "OPENAI_API_KEY"),
            qwen_api_key_env=emb_cfg.get("qwen_api_key_env", "DASHSCOPE_API_KEY"),
            qwen_base_url=emb_cfg.get("qwen_base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            qwen_timeout=emb_cfg.get("qwen_timeout", emb_cfg.get("openai_timeout", 60)),
            qwen_max_retries=emb_cfg.get("qwen_max_retries", emb_cfg.get("openai_max_retries", 3)),
            use_dimensions_parameter=emb_cfg.get("use_dimensions_parameter", True),
            normalize_embedding=emb_cfg.get("normalize_embedding", True),
            timeout=emb_cfg.get("openai_timeout", 60),
            max_retries=emb_cfg.get("openai_max_retries", 3),
            cache_root=emb_cfg.get("cache_root", "outputs/embedding_cache"),
            use_cache=emb_cfg.get("use_cache", True),
            force_refresh_cache=emb_cfg.get("force_refresh_cache", False),
            cache_only=emb_cfg.get("cache_only", False),
        )
    return LLMSemanticViewRecovery(
        config=config,
        llm_provider=llm_provider,
        embedding_provider=embedding_provider,
        prompt_builder=prompt_builder,
        device=device,
    )


def run_stage2a(config, device, stage1_model_path):
    config, latent_fea, available_mask = _load_inputs(config, device, stage1_model_path)
    recovery = _build_recovery(config, device)
    result = recovery.recover(
        latent_fea=latent_fea,
        available_mask=available_mask,
        max_samples=config["LLM"].get("max_samples"),
        max_llm_queries=config["LLM"].get("max_llm_queries"),
    )
    return config, latent_fea, available_mask, result


def dry_run_api(config, device, stage1_model_path):
    config, latent_fea, available_mask = _load_inputs(config, device, stage1_model_path)

    class _DummyProvider:
        name = config["LLM"]["provider"]
        model = config["LLM"].get("model") or config["LLM"].get("qwen_model") or config["LLM"].get("openai_model")

    class _DummyEmbedding:
        name = config["Embedding"]["provider"]
        model = config["Embedding"].get("model") or config["Embedding"].get("qwen_model") or config["Embedding"].get("openai_model")

    recovery = _build_recovery(config, device, llm_provider=_DummyProvider(), embedding_provider=_DummyEmbedding())
    query_items, previews = recovery.build_prompt_previews(
        latent_fea=latent_fea,
        available_mask=available_mask,
        max_samples=config["LLM"].get("max_samples"),
        limit=3,
    )
    max_llm_queries = config["LLM"].get("max_llm_queries")
    print("dry-run-api: no external LLM or embedding API call will be made")
    print(f"eligible queries: {len(query_items)}")
    print(f"max_llm_queries: {max_llm_queries}")
    print(f"llm cache root: {config['LLM'].get('cache_root')}")
    print(f"embedding cache root: {config['Embedding'].get('cache_root')}")
    print("prompt previews:")
    for idx, preview in enumerate(previews, start=1):
        prompt_text = preview["prompt"]
        if len(prompt_text) > 1200:
            prompt_text = prompt_text[:1200] + "\n...[truncated]"
        print(f"--- prompt {idx} sample={preview['sample_idx']} view={preview['view_idx']} hash={preview['prompt_hash']} ---")
        print(prompt_text)
    return {
        "eligible_queries": len(query_items),
        "previews": previews,
    }


def _bool_arg(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in ["1", "true", "yes", "y"]:
        return True
    if value in ["0", "false", "no", "n"]:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value}")


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate Stage 2A/2C LLM semantic y_llm path.")
    parser.add_argument("--dataset", type=str, default="BDGP")
    parser.add_argument("--missing-rate", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--stage1-model", type=str, default=None)
    parser.add_argument("--provider", choices=["mock", "openai", "deepseek", "gemini", "qwen"], default="mock")
    parser.add_argument("--embedding-provider", choices=["mock", "openai", "gemini", "qwen"], default="mock")
    parser.add_argument("--llm-model", type=str, default=None)
    parser.add_argument("--embedding-model", type=str, default=None)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-llm-queries", type=int, default=None)
    parser.add_argument("--query-order", choices=["sequential", "random"], default="sequential")
    parser.add_argument("--use-cache", dest="use_cache", action="store_true")
    parser.add_argument("--no-cache", dest="use_cache", action="store_false")
    parser.set_defaults(use_cache=True)
    parser.add_argument("--force-refresh-cache", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--openai-timeout", type=float, default=60)
    parser.add_argument("--openai-max-retries", type=int, default=3)
    parser.add_argument("--openai-temperature", type=float, default=0.0)
    parser.add_argument("--openai-max-output-tokens", type=int, default=800)
    parser.add_argument("--qwen-base-url", type=str, default=None)
    parser.add_argument("--fail-fast", nargs="?", const=True, default=False, type=_bool_arg)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--dry-run-api", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def _apply_args_to_config(config, args):
    config["Dataset"]["name"] = args.dataset
    config["Dataset"]["missing_rate"] = args.missing_rate
    config["training"]["seed"] = args.seed

    target_dim = args.embedding_dim or int(config["Module"].get("trans_dim", config["Module"].get("d_model", 512)))
    config["LLM"]["provider"] = args.provider
    if args.provider == "qwen":
        default_llm_model = config["LLM"].get("qwen_model", "qwen-plus")
    elif args.provider == "openai":
        default_llm_model = config["LLM"].get("openai_model", "gpt-4o-mini")
    else:
        default_llm_model = config["LLM"].get("llm_model", "mock-llm")
    config["LLM"]["model"] = args.llm_model or default_llm_model
    config["LLM"]["llm_model"] = config["LLM"]["model"]
    config["LLM"]["output_dim"] = target_dim
    config["LLM"]["max_samples"] = args.max_samples
    config["LLM"]["max_llm_queries"] = args.max_llm_queries
    config["LLM"]["query_order"] = args.query_order
    config["LLM"]["fail_fast"] = bool(args.fail_fast)
    config["LLM"]["allow_partial"] = bool(args.allow_partial)
    config["LLM"]["use_cache"] = bool(args.use_cache)
    config["LLM"]["force_refresh_cache"] = bool(args.force_refresh_cache)
    config["LLM"]["cache_only"] = bool(args.cache_only)
    config["LLM"]["openai_timeout"] = args.openai_timeout
    config["LLM"]["openai_max_retries"] = args.openai_max_retries
    config["LLM"]["openai_temperature"] = args.openai_temperature
    config["LLM"]["openai_max_output_tokens"] = args.openai_max_output_tokens
    config["LLM"]["qwen_timeout"] = args.openai_timeout
    config["LLM"]["qwen_max_retries"] = args.openai_max_retries
    config["LLM"]["qwen_temperature"] = args.openai_temperature
    config["LLM"]["qwen_max_output_tokens"] = args.openai_max_output_tokens
    if args.qwen_base_url:
        config["LLM"]["qwen_base_url"] = args.qwen_base_url
    if args.no_resume:
        config["LLM"]["resume"] = False
        config["LLM"]["use_cache"] = False

    config.setdefault("Embedding", {})
    config["Embedding"]["provider"] = args.embedding_provider
    if args.embedding_provider == "qwen":
        default_embedding_model = config["Embedding"].get("qwen_model", "text-embedding-v4")
    elif args.embedding_provider == "openai":
        default_embedding_model = config["Embedding"].get("openai_model", "text-embedding-3-small")
    else:
        default_embedding_model = config["Embedding"].get("embedding_model", "mock-embedding")
    config["Embedding"]["model"] = args.embedding_model or default_embedding_model
    config["Embedding"]["embedding_model"] = config["Embedding"]["model"]
    config["Embedding"]["target_dim"] = target_dim
    config["Embedding"]["use_cache"] = bool(args.use_cache)
    config["Embedding"]["force_refresh_cache"] = bool(args.force_refresh_cache)
    config["Embedding"]["cache_only"] = bool(args.cache_only)
    config["Embedding"]["openai_timeout"] = args.openai_timeout
    config["Embedding"]["openai_max_retries"] = args.openai_max_retries
    config["Embedding"]["qwen_timeout"] = args.openai_timeout
    config["Embedding"]["qwen_max_retries"] = args.openai_max_retries
    if args.qwen_base_url:
        config["Embedding"]["qwen_base_url"] = args.qwen_base_url

    config["LLM"]["embedding_provider"] = args.embedding_provider
    config["LLM"]["embedding_model"] = config["Embedding"]["model"]
    return config


def main():
    args = _parse_args()
    config = _apply_args_to_config(get_default_config(args.dataset), args)
    device = _device_from_arg(args.device)
    stage1_model_path = _stage1_model_path(config, args.stage1_model)
    if not stage1_model_path.exists():
        raise FileNotFoundError(f"Stage 1 model not found: {stage1_model_path}")

    print(f"dataset: {args.dataset}")
    print(f"missing rate: {args.missing_rate}")
    print(f"seed: {args.seed}")
    print(f"device: {device}")
    print(f"stage1 model: {stage1_model_path.as_posix()}")
    print(f"provider: {config['LLM']['provider']}")
    print(f"llm model: {config['LLM']['model']}")
    print(f"embedding provider: {config['Embedding']['provider']}")
    print(f"embedding model: {config['Embedding']['model']}")
    print(f"embedding dim: {config['Embedding']['target_dim']}")
    print(f"prompt mode: {config['LLM']['prompt_mode']}")
    print(f"max_samples: {config['LLM'].get('max_samples')}")
    print(f"max_llm_queries: {config['LLM'].get('max_llm_queries')}")

    if args.dry_run_api:
        dry_run_api(config, device, stage1_model_path)
        return

    config, latent_fea, available_mask, result = run_stage2a(config, device, stage1_model_path)

    y_llm = result["y_llm"]
    c_llm = result["c_llm"]
    s_cons = result["s_cons"]
    query_mask = result["query_mask"]
    records = result["records"]

    output_dir = _output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    y_llm_path = output_dir / "y_llm.pt"
    c_llm_path = output_dir / "c_llm.pt"
    s_cons_path = output_dir / "s_cons.pt"
    query_mask_path = output_dir / "query_mask.pt"
    records_path = output_dir / "llm_records.jsonl"
    summary_path = output_dir / "run_summary.json"

    allow_partial = bool(config["LLM"].get("allow_partial", False))
    is_partial = bool(result["is_partial"])
    safe_for_fusion = bool((not is_partial) or allow_partial)
    warning_path = _write_partial_warning(output_dir, config, is_partial)

    torch.save(y_llm.cpu(), y_llm_path)
    torch.save(c_llm.cpu(), c_llm_path)
    torch.save(s_cons.cpu(), s_cons_path)
    torch.save(query_mask.cpu(), query_mask_path)
    _write_records(records, records_path)

    num_missing_entries = int((available_mask == 0).sum().detach().cpu().item())
    summary = {
        "stage": "stage2c_real_llm_semantic_path"
        if config["LLM"]["provider"] in {"openai", "qwen"} or config["Embedding"]["provider"] in {"openai", "qwen"}
        else "stage2a_mock_semantic_path",
        "dataset": config["Dataset"]["name"],
        "missing_rate": config["Dataset"]["missing_rate"],
        "missing_rate_dir": f"missing_{format_missing_rate(config['Dataset']['missing_rate'])}",
        "seed": config["training"]["seed"],
        "provider": config["LLM"]["provider"],
        "llm_model": config["LLM"]["model"],
        "embedding_provider": config["Embedding"]["provider"],
        "embedding_model": config["Embedding"]["model"],
        "embedding_dim": int(config["Embedding"]["target_dim"]),
        "num_samples": config["Dataset"]["num_sample"],
        "num_views": config["Dataset"]["num_views"],
        "latent_dim": int(y_llm.shape[-1]),
        "num_missing_entries": num_missing_entries,
        "num_eligible_queries": int(result["num_eligible_queries"]),
        "max_llm_queries": config["LLM"].get("max_llm_queries"),
        "num_attempted_queries": int(result["num_attempted_queries"]),
        "num_successful_queries": int(result["num_successful_queries"]),
        "num_failed_queries": int(result["num_failed_queries"]),
        "num_cached_queries": int(result["num_cached_queries"]),
        "num_embedding_cached_queries": int(result["num_embedding_cached_queries"]),
        "max_samples": config["LLM"].get("max_samples"),
        "is_partial": is_partial,
        "allow_partial": allow_partial,
        "safe_for_fusion": safe_for_fusion,
        "partial_warning_path": warning_path.as_posix() if warning_path else None,
        "output_dir": output_dir.as_posix(),
        "device": str(device),
        "prompt_mode": config["LLM"]["prompt_mode"],
        "output_dim": config["LLM"]["output_dim"],
        "num_queries": len(records),
        "num_cache_hits": int(result["num_cache_hits"]),
        "num_cache_misses": int(result["num_cache_misses"]),
        "token_usage": result["token_usage"],
        "y_llm_shape": list(y_llm.shape),
        "c_llm_shape": list(c_llm.shape),
        "s_cons_shape": list(s_cons.shape),
        "query_mask_shape": list(query_mask.shape),
        "stage1_model_path": stage1_model_path.as_posix(),
        "output_paths": {
            "y_llm": y_llm_path.as_posix(),
            "c_llm": c_llm_path.as_posix(),
            "s_cons": s_cons_path.as_posix(),
            "query_mask": query_mask_path.as_posix(),
            "llm_records": records_path.as_posix(),
            "run_summary": summary_path.as_posix(),
        },
        "y_llm_path": y_llm_path.as_posix(),
        "c_llm_path": c_llm_path.as_posix(),
        "s_cons_path": s_cons_path.as_posix(),
        "query_mask_path": query_mask_path.as_posix(),
        "records_path": records_path.as_posix(),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"latent_fea shape: {tuple(latent_fea.shape)}")
    print(f"available_mask shape: {tuple(available_mask.shape)}")
    print(f"y_llm shape: {tuple(y_llm.shape)}")
    print(f"c_llm shape: {tuple(c_llm.shape)}")
    print(f"s_cons shape: {tuple(s_cons.shape)}")
    print(f"query_mask shape: {tuple(query_mask.shape)}")
    print(f"query_mask sum: {float(query_mask.sum().detach().cpu()):.0f}")
    print(f"num attempted queries: {result['num_attempted_queries']}")
    print(f"num successful queries: {result['num_successful_queries']}")
    print(f"num failed queries: {result['num_failed_queries']}")
    print(f"num cached queries: {result['num_cached_queries']}")
    print(f"is_partial: {str(is_partial).lower()}")
    print(f"safe_for_fusion: {str(safe_for_fusion).lower()}")
    print(f"saved y_llm: {y_llm_path.as_posix()}")
    print(f"saved c_llm: {c_llm_path.as_posix()}")
    print(f"saved s_cons: {s_cons_path.as_posix()}")
    print(f"saved query_mask: {query_mask_path.as_posix()}")
    print(f"saved records: {records_path.as_posix()}")
    print(f"saved summary: {summary_path.as_posix()}")
    if warning_path is not None:
        print(f"saved partial warning: {warning_path.as_posix()}")


if __name__ == "__main__":
    main()
