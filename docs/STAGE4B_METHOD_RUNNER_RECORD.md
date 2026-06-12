# Stage 4B Method Runner Record

## Status
Stage 4B packages the current LLM-GIMVC method into a single runner.

## Pipeline

Stage 1:
available views -> statistical Transformer generator -> y_stat

Stage 2A:
available views -> prompt builder / LLM provider / embedding provider -> y_llm, c_llm, s_cons

Stage 2B:
y_stat + y_llm + c_llm + s_cons -> fusion gate -> y_final

Stage 4A:
y_final -> DCP-style clustering head -> final clustering metrics

## Main Entry

python run_llm_gimvc_method.py \
  --dataset BDGP \
  --missing-rate 0.5 \
  --seed 0 \
  --provider mock \
  --embedding-provider mock \
  --gate-mode heuristic \
  --input-source fusion \
  --head-type dcp \
  --representation mean \
  --device cuda:0

## Output

outputs/llm_gimvc_method/{dataset}/missing_{rate}/seed_{seed}/{gate_mode}/{input_source}/{head_type}/{representation}/

Contains:
- method_metrics.json
- method_summary.json
- command_log.json

## Safety

The runner refuses unsafe Stage 2A outputs by default.

The runner refuses debug-only Fusion outputs by default.

Unsafe or debug outputs can only be used with explicit flags:
- --allow-unsafe-stage2a
- --allow-debug-fusion

## Current Limitations

- This is not Block 1 sweep.
- This is not external baseline comparison.
- This does not implement MICA / JGA-IMVC / FreeCSL.
- This does not train Prompt Adapter.
- This does not train Fusion Gate MLP.
- Mock LLM outputs are diagnostic only and cannot be used as final paper evidence.

## Next Stage

Stage 5A should create the Block 1 experiment runner:
- multiple seeds
- multiple missing rates
- fusion vs statistical_only comparison
- result table aggregation
- CSV / JSON summary for paper analysis
