# Stage 5A Block 1 Runner Record

## Status

Stage 5A implements the Block 1 experiment entrance.

It supports:

- Single Block 1 job:
  python -m llm_gimvc.experiments.block1

- Local sweep:
  python run_block1_sweep.py

- Result aggregation:
  python aggregate_block1_results.py

## Runnable Methods

Stage 5A currently supports:

- llm_gimvc
- statistical_only

External baselines are reserved:

- mica
- jga_imvc
- freecsl

These baselines must raise NotImplementedError in Stage 5A.

## Pipeline Mapping

### llm_gimvc

Runs Stage 4B with:

input_source=fusion

This corresponds to:

available views
-> statistical generator
-> LLM semantic path
-> fusion gate
-> DCP-style clustering head

### statistical_only

Runs Stage 4B with:

input_source=statistical_only

This corresponds to:

available views
-> statistical generator
-> DCP-style clustering head

In Stage 5A, statistical_only may still reuse the full Stage 4B runner for engineering consistency.

## Output Structure

results/block1/{dataset}/{missing_pattern}/missing_{rate}/{method}/seed_{seed}/

Each job writes:

- metrics.json
- job_summary.json
- command_log.json

## Aggregated Outputs

results/block1_summary/

Contains:

- block1_all_runs.csv
- block1_mean_std.csv
- block1_delta_vs_statistical.csv
- block1_summary.json

## Current Limitations

- Stage 5A is not the final full 600-job Block 1.
- Stage 5A does not implement MICA.
- Stage 5A does not implement JGA-IMVC.
- Stage 5A does not implement FreeCSL.
- Mock LLM outputs are diagnostic only.
- No paper conclusion should be drawn from mock LLM results.

## Next Stage

Stage 5B should implement or wrap external baseline methods:

- MICA
- JGA-IMVC
- FreeCSL

Stage 5C should rebuild the full 600-job Block 1 manifest and launch the full missing-rate sweep.
