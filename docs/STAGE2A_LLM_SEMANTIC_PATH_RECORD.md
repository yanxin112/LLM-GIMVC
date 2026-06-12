# Stage 2A LLM Semantic Path Record

## Status
- Stage 2A LLM semantic recovery branch: DONE
- Current provider: mock
- Current embedding provider: mock
- No external API is used in Stage 2A by default

## Purpose
Generate y_llm / c_llm / s_cons tensors aligned with Stage 1 y_stat.

## Input Protocol
- Stage 1 model.pt
- features from datasets.get_loader()
- available_mask / inc_mask
- latent_fea from StatisticalViewGenerator.encode_views()

## Output Protocol
- y_llm: Tensor [N, V, 512]
- c_llm: Tensor [N, V]
- s_cons: Tensor [N, V]
- query_mask: Tensor [N, V]
- llm_records.jsonl
- run_summary.json

## Current Limitations
- fixed_template prompt only
- mock LLM provider by default
- mock embedding provider by default
- no Prompt Adapter training yet
- no Fusion Gate yet
- no clustering head yet

## Partial Runs with --max-samples
- `--max-samples` is only for smoke tests.
- Partial outputs are saved under:
  `outputs/llm_semantic_path/{dataset}/missing_{rate}/seed_{seed}/partial_max_samples_{K}/`
- Partial outputs are not safe for Fusion Gate input.
- Full outputs are saved under:
  `outputs/llm_semantic_path/{dataset}/missing_{rate}/seed_{seed}/`
- `run_summary.json` contains:
  - `is_partial`
  - `max_samples`
  - `safe_for_fusion`
  - `num_processed_samples`
  - `num_total_samples`
- Partial directories include:
  `PARTIAL_RUN_DO_NOT_USE_FOR_FUSION.txt`
- Fusion Gate must reject partial outputs by default.

## Next Stage
Stage 2B / Stage 3 will implement Fusion Gate:
[y_stat, y_llm, c_llm, s_cons] -> y_final
