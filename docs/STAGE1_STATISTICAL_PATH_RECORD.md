# Stage 1 Statistical Path Record

## Status
- Stage 1 statistical Transformer y_stat branch extraction: DONE
- Current scope: available views -> Transformer cross-view generator -> y_stat
- Not included yet: LLM path, Prompt Adapter, Fusion Gate, DCP / Completer clustering head

## Current y_stat Protocol
- latent_fea: Tensor [N, V, D]
- inc_mask / available_mask: Tensor [N, V], 1 = observed view, 0 = missing view
- mask_fea: Tensor [B, V, D]
- in_mask: Tensor [B, V], views fed into Transformer
- out_mask: Tensor [B, V], originally available views
- y_stat: Tensor [N, V, D]

## Current Minimal Command
python run_statistical_path.py --dataset BDGP --missing-rate 0.5 --seed 0

## Output Files
outputs/statistical_path/{dataset}/missing_{rate_percent}/seed_{seed}/y_stat.pt
outputs/statistical_path/{dataset}/missing_{rate_percent}/seed_{seed}/model.pt
outputs/statistical_path/{dataset}/missing_{rate_percent}/seed_{seed}/run_summary.json

## Notes
- embedding_layers are frozen in Stage 1 by default.
- Stage 1 trains only the Transformer cross-view generator ETrans.
- loss_target can be:
  - visible: use out_mask, legacy behavior
  - heldout: use out_mask - in_mask, preferred for cross-view recovery
- If heldout target is empty for a batch, fallback to out_mask to avoid zero-loss batches.
