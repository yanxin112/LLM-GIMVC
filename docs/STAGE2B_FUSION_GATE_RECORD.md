# Stage 2B Fusion Gate Record

## Status
- Stage 2B Fusion Gate minimal implementation: DONE
- Current default gate: heuristic confidence-consistency gate
- Trainable MLP gate is defined for future Stage 3 but not trained in Stage 2B

## Inputs
- Stage 1 y_stat: Tensor [N,V,D]
- Stage 2A y_llm: Tensor [N,V,D]
- Stage 2A c_llm: Tensor [N,V]
- Stage 2A s_cons: Tensor [N,V]
- Stage 2A query_mask: Tensor [N,V]
- available_mask / inc_mask: Tensor [N,V]
- latent_fea from Stage 1 encoder: Tensor [N,V,D]

## Outputs
- y_final: Tensor [N,V,D]
- gate_weight: Tensor [N,V]
- source_mask: Tensor [N,V]
- eligible_mask: Tensor [N,V]
- fusion_summary.json

## Stage 2A Safety Check
Before Fusion Gate runs, it reads:
`outputs/llm_semantic_path/{dataset}/missing_{rate}/seed_{seed}/run_summary.json`

Required fields:
- `is_partial`
- `safe_for_fusion`
- `max_samples`

Default behavior:
- If `is_partial=True`, Fusion Gate raises RuntimeError.
- If `safe_for_fusion=False`, Fusion Gate raises RuntimeError.
- If safety fields are missing, Fusion Gate raises RuntimeError and asks to re-run Stage 2A.
- Partial outputs can only be used with `--allow-partial-stage2a`, and the resulting fusion output is debug-only.

## Fusion Rule
For missing and queried entries:
y_final = w * y_llm + (1 - w) * y_stat

where:
w = c_llm * s_cons
w = 0 if c_llm < abstention_threshold
w = 0 if query_mask == 0
w = 0 for observed views

For observed views:
y_final = latent_fea

## Source Mask
- 0 = observed original latent_fea
- 1 = statistical fallback y_stat
- 2 = fused with LLM contribution

## Current Limitations
- No clustering head yet
- No end-to-end gate training yet
- MLP gate is debug-only unless trained in Stage 3
- mock y_llm is not semantically meaningful for final experiments

## Next Stage
Stage 3 will attach clustering head / reconstruction or contrastive objective and train the gate.
