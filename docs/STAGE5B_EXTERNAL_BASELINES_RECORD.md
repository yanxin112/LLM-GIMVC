# Stage 5B External Baselines Record

## Status

Stage 5B implements external baseline interfaces and result adapters.

Supported external baseline methods:

- mica
- jga_imvc
- freecsl

## Purpose

This stage does not reimplement the baseline algorithms.

It provides:

- standardized data export
- external command wrapper
- raw output collection
- prediction / metric adapter
- Block 1 metrics.json writer

## Data Export

All baselines use the same data bundle:

results/baseline_data/{dataset}/{missing_pattern}/missing_{rate}/seed_{seed}/data.npz

The bundle contains:

- complete views
- masked views
- available_mask
- missing_mask
- labels
- metadata

Labels are included only for post-hoc evaluation.

## Output Format

All methods write:

results/block1/{dataset}/{missing_pattern}/missing_{rate}/{method}/seed_{seed}/metrics.json

with metrics:

- NMI
- ARI
- ACC
- Purity

## External Repositories

Default expected layout:

external_baselines/
  MICA/
  JGA-IMVC/
  FreeCSL/

If a repo or entrypoint is missing, the job fails clearly.

No fake metrics are generated.

## Adapter Sources

The adapter can read:

- raw metrics from metrics.json / csv / mat
- predicted labels from pred_labels.npy / csv / mat

If only predictions are found, metrics are computed with project clustering_metrics.

## Current Limitations

- Stage 5B does not verify whether external repos use labels correctly internally.
- Stage 5B does not tune baseline hyperparameters.
- Stage 5B does not launch the full 600-job Block 1.
- Stage 5B does not rewrite MICA / JGA-IMVC / FreeCSL algorithms.

## Next Stage

Stage 5C should rebuild and launch the full Block 1 manifest:

4 datasets x 6 missing rates x 5 methods x 5 seeds = 600 jobs.
