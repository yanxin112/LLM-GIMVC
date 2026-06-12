# Stage 3B Comparison Evaluation Record

## Status
- Stage 3B comparison evaluation: DONE
- This stage evaluates multiple recovery sources with the same KMeans protocol.
- No DCP / Completer training.
- No Fusion Gate MLP training.
- No Prompt Adapter training.

## Methods

### observed_only
Observed original latent_fea only. Missing views are ignored for mean/sum via mask-aware aggregation and zero-filled for concat.

### statistical_only
Observed views use latent_fea; missing views use y_stat.

### llm_only
Observed views use latent_fea; missing views use y_llm. Unqueried missing views use zero fallback by default.

### fusion
Uses Stage 2B y_final.

## Metrics
- NMI
- ARI
- ACC
- Purity

## Reference
Default reference method:
- statistical_only

The main diagnostic is:
- fusion NMI - statistical_only NMI

## Safety
- Refuse partial Stage 2A by default.
- Refuse debug-only Fusion by default.
- Allow debug only with explicit flags.

## Current Limitations
- Uses KMeans only.
- Mock LLM outputs are not semantically meaningful.
- No trainable clustering head yet.
- No official baselines MICA / JGA-IMVC / FreeCSL yet.

## Next Stage
Stage 3C will package the current pipeline into a single method runner:
method=llm_gimvc / statistical_only / observed_only / llm_only
so it can later become the Block 1 experiment entrypoint.
