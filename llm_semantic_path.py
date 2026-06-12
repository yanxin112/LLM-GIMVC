import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from prompt_builder import summarize_latent_view


def format_missing_rate(missing_rate):
    rate = float(missing_rate)
    if rate <= 1.0:
        return str(int(round(rate * 100)))
    return str(int(round(rate)))


def prompt_hash(prompt):
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def response_hash(response):
    return hashlib.sha256(json.dumps(response, sort_keys=True, default=str).encode("utf-8")).hexdigest()


class LLMSemanticViewRecovery:
    def __init__(
        self,
        config,
        llm_provider,
        embedding_provider,
        prompt_builder,
        device,
    ):
        self.config = config
        self.llm_provider = llm_provider
        self.embedding_provider = embedding_provider
        self.prompt_builder = prompt_builder
        self.device = device
        self.dataset = config["Dataset"]["name"]
        self.missing_rate = config["Dataset"]["missing_rate"]
        self.seed = config["training"]["seed"]
        self.llm_cfg = config["LLM"]
        self.embedding_cfg = config.get("Embedding", {})
        self.query_observed_views = self.llm_cfg.get("query_observed_views", False)
        self.fail_fast = bool(self.llm_cfg.get("fail_fast", False))
        self.query_order = self.llm_cfg.get("query_order", "sequential")

    def _build_available_summaries(self, latent_fea, sample_index, observed_views):
        summaries = {}
        for view_idx in observed_views:
            summaries[int(view_idx)] = summarize_latent_view(latent_fea[sample_index, view_idx])
        return summaries

    def _normalize_confidence(self, value):
        confidence = float(value)
        if confidence > 1.0 and confidence <= 10.0:
            confidence = confidence / 10.0
        return float(max(0.0, min(1.0, confidence)))

    def _iter_query_items(self, latent_fea, available_mask, max_samples=None):
        n_samples, num_views, dim = latent_fea.shape
        max_index = n_samples if max_samples is None else min(int(max_samples), n_samples)
        items = []
        for sample_index in range(max_index):
            observed_views = torch.where(available_mask[sample_index] == 1)[0].tolist()
            missing_views = torch.where(available_mask[sample_index] == 0)[0].tolist()
            if len(observed_views) == 0:
                continue
            target_views = list(range(num_views)) if self.query_observed_views else missing_views
            for target_view in target_views:
                items.append((int(sample_index), int(target_view), [int(view_idx) for view_idx in observed_views]))
        if self.query_order == "random":
            generator = torch.Generator()
            generator.manual_seed(int(self.seed))
            order = torch.randperm(len(items), generator=generator).tolist()
            items = [items[idx] for idx in order]
        elif self.query_order != "sequential":
            raise ValueError(f"Unsupported query_order: {self.query_order}")
        return items

    def build_prompt_previews(self, latent_fea, available_mask, max_samples=None, limit=3):
        latent_fea = latent_fea.to(self.device)
        available_mask = available_mask.to(self.device)
        n_samples, num_views, dim = latent_fea.shape
        query_items = self._iter_query_items(latent_fea, available_mask, max_samples=max_samples)
        previews = []
        for sample_index, target_view, observed_views in query_items[:limit]:
            available_summaries = self._build_available_summaries(latent_fea, sample_index, observed_views)
            prompt = self.prompt_builder.build_prompt(
                sample_index=sample_index,
                target_view=target_view,
                available_view_indices=observed_views,
                available_view_summaries=available_summaries,
                dataset_metadata={"num_views": num_views, "latent_dim": dim},
            )
            previews.append(
                {
                    "sample_idx": sample_index,
                    "view_idx": target_view,
                    "prompt_hash": prompt_hash(prompt),
                    "prompt": prompt,
                }
            )
        return query_items, previews

    def recover(
        self,
        latent_fea,
        available_mask,
        max_samples=None,
        max_llm_queries=None,
    ):
        latent_fea = latent_fea.to(self.device)
        available_mask = available_mask.to(self.device)
        n_samples, num_views, dim = latent_fea.shape
        output_dim = int(self.llm_cfg.get("output_dim", dim))
        if output_dim != dim:
            raise ValueError(f"LLM output_dim {output_dim} must match latent_fea dim {dim} in Stage 2C")

        y_llm = torch.zeros((n_samples, num_views, dim), dtype=torch.float32, device=self.device)
        c_llm = torch.zeros((n_samples, num_views), dtype=torch.float32, device=self.device)
        s_cons = torch.zeros((n_samples, num_views), dtype=torch.float32, device=self.device)
        query_mask = torch.zeros((n_samples, num_views), dtype=torch.float32, device=self.device)
        records = []

        query_items = self._iter_query_items(latent_fea, available_mask, max_samples=max_samples)
        num_eligible_queries = len(query_items)
        if max_llm_queries is not None:
            query_items = query_items[: max(int(max_llm_queries), 0)]

        attempted = 0
        successful = 0
        failed = 0
        cached = 0
        embedding_cached = 0
        token_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "embedding_tokens": 0,
            "total_tokens": 0,
        }

        for sample_index, target_view, observed_views in query_items:
            available_mean = latent_fea[sample_index, observed_views, :].mean(dim=0)
            available_summaries = self._build_available_summaries(
                latent_fea,
                sample_index,
                observed_views,
            )
            prompt = self.prompt_builder.build_prompt(
                sample_index=sample_index,
                target_view=target_view,
                available_view_indices=observed_views,
                available_view_summaries=available_summaries,
                dataset_metadata={"num_views": num_views, "latent_dim": dim},
            )
            phash = prompt_hash(prompt)
            attempted += 1
            base_record = {
                "sample_idx": int(sample_index),
                "sample_index": int(sample_index),
                "view_idx": int(target_view),
                "target_view": int(target_view),
                "available_views": [int(view_idx) for view_idx in observed_views],
                "provider": self.llm_provider.name,
                "llm_model": self.llm_provider.model,
                "embedding_provider": self.embedding_provider.name,
                "embedding_model": self.embedding_provider.model,
                "embedding_dim": int(dim),
                "prompt_hash": phash,
            }
            try:
                response = self.llm_provider.recover_missing_view(
                    prompt,
                    metadata={"sample_idx": sample_index, "view_idx": target_view},
                )
                llm_from_cache = bool(getattr(self.llm_provider, "last_cache_hit", False))
                if llm_from_cache:
                    cached += 1
                recovered_text = response.get("recovered_text") or response.get("description")
                semantic_label = response.get("semantic_label", "")
                embedding_text = f"{semantic_label}\n{recovered_text}".strip()
                embedding = self.embedding_provider.embed_text(embedding_text)
                embedding_meta = getattr(self.embedding_provider, "last_metadata", {}) or {}
                if bool(embedding_meta.get("from_cache", False)):
                    embedding_cached += 1
                emb_tensor = torch.from_numpy(embedding).to(self.device).float()
                if emb_tensor.numel() != dim:
                    raise ValueError(f"Embedding dimension mismatch: got {emb_tensor.numel()}, expected {dim}")

                confidence = self._normalize_confidence(response.get("confidence", 0.0))
                y_llm[sample_index, target_view] = emb_tensor
                c_llm[sample_index, target_view] = confidence
                query_mask[sample_index, target_view] = 1

                cosine = F.cosine_similarity(emb_tensor.unsqueeze(0), available_mean.unsqueeze(0), dim=1)[0]
                consistency = float(((cosine + 1.0) / 2.0).clamp(0.0, 1.0).detach().cpu())
                s_cons[sample_index, target_view] = consistency

                usage = response.get("usage", {}) or {}
                emb_usage = embedding_meta.get("usage", {}) or {}
                token_usage["input_tokens"] += int(usage.get("input_tokens", 0) or 0)
                token_usage["output_tokens"] += int(usage.get("output_tokens", 0) or 0)
                token_usage["embedding_tokens"] += int(emb_usage.get("total_tokens", emb_usage.get("prompt_tokens", 0)) or 0)
                token_usage["total_tokens"] += int(usage.get("total_tokens", 0) or 0) + int(
                    emb_usage.get("total_tokens", 0) or 0
                )

                successful += 1
                records.append(
                    {
                        **base_record,
                        "status": "cached" if llm_from_cache else "success",
                        "response_hash": response_hash(response),
                        "recovered_text": recovered_text,
                        "description": recovered_text,
                        "semantic_label": semantic_label,
                        "confidence": float(confidence),
                        "s_cons": consistency,
                        "should_abstain": bool(response.get("should_abstain", False)),
                        "usage": usage,
                        "embedding_usage": emb_usage,
                        "embedding_from_cache": bool(embedding_meta.get("from_cache", False)),
                        "error_type": None,
                        "error_message": None,
                    }
                )
            except Exception as exc:
                failed += 1
                records.append(
                    {
                        **base_record,
                        "status": "failed",
                        "response_hash": None,
                        "recovered_text": None,
                        "semantic_label": None,
                        "confidence": 0.0,
                        "s_cons": 0.0,
                        "should_abstain": True,
                        "usage": {},
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
                if self.fail_fast:
                    raise

        is_partial = bool(
            (max_samples is not None and int(max_samples) < n_samples)
            or (max_llm_queries is not None and int(max_llm_queries) < num_eligible_queries)
            or (successful < num_eligible_queries)
        )
        return {
            "y_llm": y_llm,
            "c_llm": c_llm,
            "s_cons": s_cons,
            "query_mask": query_mask,
            "records": records,
            "num_cache_hits": cached,
            "num_cache_misses": max(0, attempted - cached),
            "num_cached_queries": cached,
            "num_embedding_cached_queries": embedding_cached,
            "num_eligible_queries": num_eligible_queries,
            "num_attempted_queries": attempted,
            "num_successful_queries": successful,
            "num_failed_queries": failed,
            "is_partial": is_partial,
            "token_usage": token_usage,
        }
