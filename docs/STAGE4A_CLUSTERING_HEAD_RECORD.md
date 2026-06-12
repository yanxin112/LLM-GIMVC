# Stage 4A Clustering Head Record

## Status
- Stage 4A trainable clustering head: DONE
- This stage trains a minimal DCP-style clustering head on recovered multi-view latent tensors.
- No real LLM API call.
- No Prompt Adapter training.
- No Fusion Gate MLP training.
- No Block 1 sweep.
- No external baselines yet.

## Input Sources

### fusion
Uses Stage 2B y_final. This corresponds to the current LLM-GIMVC recovered tensor.

### statistical_only
Observed views use latent_fea; missing views use y_stat.

### observed_only
Observed latent_fea only; missing views are zero-filled.

### llm_only
Observed views use latent_fea; missing views use y_llm. Unqueried missing views use zero fallback by default.

## Head

### dcp
Minimal DCP-style trainable head:
- shared view projector
- multi-view contrastive loss
- DEC-style clustering KL
- cluster balance regularization
- KMeans initialization of cluster centers

## Metrics
- NMI
- ARI
- ACC
- Purity

## Reported Evaluation
- head_assignment metrics
- kmeans_on_head_representation metrics

## Safety
- Refuse partial Stage 2A by default.
- Refuse debug-only Fusion by default.
- Allow debug only with explicit flags.

## Current Limitations
- This is a minimal DCP-style head, not a full reproduction of original DCP.
- Completer full implementation is reserved for the next stage if needed.
- Mock LLM outputs are not semantically meaningful.
- Results are diagnostic until real LLM / text-embedding-3 outputs are used.

## Next Stage
Stage 4B should either:
1. add Completer-style head if required, or
2. package Stage 1 -> Stage 2A -> Stage 2B -> Stage 4A into a single method runner for Block 1.
