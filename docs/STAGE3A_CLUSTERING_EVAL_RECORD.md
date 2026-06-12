# Stage 3A Clustering Evaluation Record

## Status
- Stage 3A minimal clustering evaluation: DONE
- This stage does not train DCP / Completer.
- This stage does not train Fusion Gate MLP.
- This stage only evaluates y_final with KMeans.

## Inputs
- Stage 2B y_final: Tensor [N,V,D]
- Dataset labels from datasets.get_loader()
- Fusion summary for safety check

## Representation
Default:
- mean over views: y_final.mean(dim=1) -> [N,D]
- L2 normalize before KMeans

Optional:
- concat views: [N,V*D]
- sum views: [N,D]

## Metrics
- NMI
- ARI
- ACC
- Purity

## Safety
- Refuse debug-only Fusion output by default.
- Refuse Fusion output generated from partial Stage 2A by default.
- Use --allow-debug-fusion only for debugging.

## Current Limitations
- KMeans-only evaluation.
- No DCP / Completer head yet.
- No end-to-end training.
- Mock y_llm is not semantically meaningful for paper results.

## Next Stage
Stage 3B will add comparison modes:
- y_stat-only clustering
- y_llm-only clustering
- y_final clustering
- observed-only clustering

Stage 3C or Stage 4 will add trainable clustering head / DCP-style training.
