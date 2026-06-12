import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import requests
except ImportError:
    requests = None


def _safe_model_name(model):
    return str(model).replace("/", "_").replace("\\", "_").replace(":", "_")


def _sha256_text(text):
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


class BaseEmbeddingProvider:
    name = "base"
    model = "base"

    def embed_text(self, text):
        raise NotImplementedError


class MockEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, output_dim=512, model="mock-embedding"):
        self.name = "mock"
        self.model = model
        self.output_dim = output_dim
        self.last_metadata = {
            "provider": "mock",
            "model": model,
            "target_dim": output_dim,
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
            "from_cache": False,
        }

    def embed_text(self, text):
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        seed = int(digest[:16], 16) % (2 ** 32)
        rng = np.random.default_rng(seed)
        embedding = rng.normal(0.0, 1.0, size=self.output_dim).astype("float32")
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        self.last_metadata = {
            "provider": "mock",
            "model": self.model,
            "target_dim": self.output_dim,
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
            "from_cache": False,
        }
        return embedding.astype("float32")


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(
        self,
        model="text-embedding-3-small",
        api_key_env="OPENAI_API_KEY",
        target_dim=512,
        use_dimensions_parameter=True,
        normalize_embedding=True,
        timeout=60,
        max_retries=3,
        cache_root="outputs/embedding_cache",
        use_cache=True,
        force_refresh_cache=False,
        cache_only=False,
    ):
        self.name = "openai"
        self.model = model
        self.target_dim = int(target_dim)
        self.output_dim = int(target_dim)
        self.api_key_env = api_key_env
        self.use_dimensions_parameter = bool(use_dimensions_parameter)
        self.normalize_embedding = bool(normalize_embedding)
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache_root = Path(cache_root)
        self.use_cache = bool(use_cache)
        self.force_refresh_cache = bool(force_refresh_cache)
        self.cache_only = bool(cache_only)
        self.last_metadata = {}

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"{api_key_env} is required for embedding-provider=openai.")
        if OpenAI is None:
            raise ImportError("openai package is required for OpenAIEmbeddingProvider. Please install openai.")
        self.client = OpenAI(api_key=api_key, timeout=timeout, max_retries=max_retries)

    def _cache_path(self, text):
        digest = _sha256_text(
            json.dumps(
                {
                    "provider": "openai",
                    "model": self.model,
                    "target_dim": self.target_dim,
                    "text_hash": _sha256_text(text),
                    "normalize_embedding": self.normalize_embedding,
                },
                sort_keys=True,
            )
        )
        return (
            self.cache_root
            / "openai"
            / _safe_model_name(self.model)
            / f"dim_{self.target_dim}"
            / f"{digest}.json"
        )

    def _load_cache(self, text):
        path = self._cache_path(text)
        if self.use_cache and not self.force_refresh_cache and path.exists():
            with open(path, "r", encoding="utf-8-sig") as f:
                obj = json.load(f)
            embedding = np.asarray(obj["embedding"], dtype=np.float32)
            self.last_metadata = {
                "provider": "openai",
                "model": self.model,
                "target_dim": self.target_dim,
                "usage": obj.get("usage", {}),
                "from_cache": True,
                "cache_path": path.as_posix(),
            }
            return embedding
        if self.cache_only:
            raise RuntimeError(f"Embedding cache miss for text hash {_sha256_text(text)} and cache_only=True.")
        return None

    def _write_cache(self, text, embedding, usage):
        if not self.use_cache:
            return
        path = self._cache_path(text)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "provider": "openai",
                    "model": self.model,
                    "target_dim": self.target_dim,
                    "text_hash": _sha256_text(text),
                    "embedding": np.asarray(embedding, dtype=np.float32).tolist(),
                    "usage": usage,
                    "created_at": _utc_now(),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    def _usage_dict(self, usage):
        if usage is None:
            return {"prompt_tokens": 0, "total_tokens": 0}
        if isinstance(usage, dict):
            return {
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "total_tokens": int(usage.get("total_tokens") or 0),
            }
        return {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        }

    def embed_with_metadata(self, text):
        cached = self._load_cache(text)
        if cached is not None:
            return {"embedding": cached, **self.last_metadata}

        kwargs = {
            "model": self.model,
            "input": text,
        }
        if self.use_dimensions_parameter and self.model in ["text-embedding-3-small", "text-embedding-3-large"]:
            kwargs["dimensions"] = self.target_dim

        try:
            response = self.client.embeddings.create(**kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"OpenAI embedding query failed. If this model does not support dimensions, "
                f"disable use_dimensions_parameter explicitly. Error: {type(exc).__name__}: {exc}"
            ) from exc

        vector = np.asarray(response.data[0].embedding, dtype=np.float32)
        if vector.shape[0] != self.target_dim:
            raise RuntimeError(f"Embedding dim mismatch: got {vector.shape[0]}, expected {self.target_dim}.")
        if self.normalize_embedding:
            vector = vector / (np.linalg.norm(vector) + 1e-12)
        vector = vector.astype("float32")
        usage = self._usage_dict(getattr(response, "usage", None))
        self._write_cache(text, vector, usage)
        self.last_metadata = {
            "provider": "openai",
            "model": self.model,
            "target_dim": self.target_dim,
            "usage": usage,
            "from_cache": False,
        }
        return {"embedding": vector, **self.last_metadata}

    def embed_text(self, text):
        return self.embed_with_metadata(text)["embedding"]


class QwenEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(
        self,
        model="text-embedding-v4",
        api_key_env="DASHSCOPE_API_KEY",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        target_dim=512,
        use_dimensions_parameter=True,
        normalize_embedding=True,
        timeout=60,
        max_retries=3,
        cache_root="outputs/embedding_cache",
        use_cache=True,
        force_refresh_cache=False,
        cache_only=False,
    ):
        self.name = "qwen"
        self.model = model
        self.target_dim = int(target_dim)
        self.output_dim = int(target_dim)
        self.api_key_env = api_key_env
        self.base_url = base_url.rstrip("/")
        self.embedding_url = self.base_url + "/embeddings"
        self.use_dimensions_parameter = bool(use_dimensions_parameter)
        self.normalize_embedding = bool(normalize_embedding)
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache_root = Path(cache_root)
        self.use_cache = bool(use_cache)
        self.force_refresh_cache = bool(force_refresh_cache)
        self.cache_only = bool(cache_only)
        self.last_metadata = {}

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"{api_key_env} is required for embedding-provider=qwen.")
        if requests is None:
            raise ImportError("requests package is required for QwenEmbeddingProvider. Please install requests.")
        self.api_key = api_key

    def _cache_path(self, text):
        digest = _sha256_text(
            json.dumps(
                {
                    "provider": "qwen",
                    "model": self.model,
                    "target_dim": self.target_dim,
                    "text_hash": _sha256_text(text),
                    "normalize_embedding": self.normalize_embedding,
                },
                sort_keys=True,
            )
        )
        return (
            self.cache_root
            / "qwen"
            / _safe_model_name(self.model)
            / f"dim_{self.target_dim}"
            / f"{digest}.json"
        )

    def _load_cache(self, text):
        path = self._cache_path(text)
        if self.use_cache and not self.force_refresh_cache and path.exists():
            with open(path, "r", encoding="utf-8-sig") as f:
                obj = json.load(f)
            embedding = np.asarray(obj["embedding"], dtype=np.float32)
            self.last_metadata = {
                "provider": "qwen",
                "model": self.model,
                "target_dim": self.target_dim,
                "usage": obj.get("usage", {}),
                "from_cache": True,
                "cache_path": path.as_posix(),
            }
            return embedding
        if self.cache_only:
            raise RuntimeError(f"Qwen embedding cache miss for text hash {_sha256_text(text)} and cache_only=True.")
        return None

    def _write_cache(self, text, embedding, usage):
        if not self.use_cache:
            return
        path = self._cache_path(text)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "provider": "qwen",
                    "model": self.model,
                    "target_dim": self.target_dim,
                    "text_hash": _sha256_text(text),
                    "embedding": np.asarray(embedding, dtype=np.float32).tolist(),
                    "usage": usage,
                    "created_at": _utc_now(),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    def _usage_dict(self, usage):
        if usage is None:
            return {"prompt_tokens": 0, "total_tokens": 0}
        if isinstance(usage, dict):
            return {
                "prompt_tokens": int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
                "total_tokens": int(usage.get("total_tokens") or 0),
            }
        return {"prompt_tokens": 0, "total_tokens": 0}

    def _request_payload(self, text):
        payload = {
            "model": self.model,
            "input": text,
        }
        if self.use_dimensions_parameter:
            payload["dimensions"] = self.target_dim
        return payload

    def _post_json(self, payload):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error = None
        for attempt in range(int(self.max_retries) + 1):
            try:
                response = requests.post(
                    self.embedding_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                if response.status_code in (401, 403):
                    raise RuntimeError(
                        f"Qwen embedding authentication failed with HTTP {response.status_code}. "
                        f"Check {self.api_key_env}."
                    )
                if response.status_code == 429 or response.status_code >= 500:
                    body = response.text[:500]
                    last_error = RuntimeError(f"Qwen embedding HTTP {response.status_code}: {body}")
                    if attempt < int(self.max_retries):
                        time.sleep(min(2 ** attempt, 8))
                        continue
                    raise last_error
                if response.status_code >= 400:
                    raise RuntimeError(f"Qwen embedding HTTP {response.status_code}: {response.text[:500]}")
                return response.json()
            except requests.Timeout as exc:
                last_error = exc
                if attempt < int(self.max_retries):
                    time.sleep(min(2 ** attempt, 8))
                    continue
                raise RuntimeError(f"Qwen embedding request timed out after {self.timeout}s.") from exc
            except requests.RequestException as exc:
                last_error = exc
                if attempt < int(self.max_retries):
                    time.sleep(min(2 ** attempt, 8))
                    continue
                raise RuntimeError(f"Qwen embedding request failed: {type(exc).__name__}: {exc}") from exc
        raise RuntimeError(f"Qwen embedding request failed: {last_error}")

    def embed_with_metadata(self, text):
        cached = self._load_cache(text)
        if cached is not None:
            return {"embedding": cached, **self.last_metadata}

        try:
            response_json = self._post_json(self._request_payload(text))
            embedding = response_json["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Qwen embedding response did not contain data[0].embedding.") from exc
        except Exception:
            raise

        vector = np.asarray(embedding, dtype=np.float32)
        if vector.shape[0] != self.target_dim:
            raise ValueError(f"Qwen embedding dim mismatch: got {vector.shape[0]}, expected {self.target_dim}")
        if self.normalize_embedding:
            vector = vector / (np.linalg.norm(vector) + 1e-12)
        vector = vector.astype("float32")
        usage = self._usage_dict(response_json.get("usage", {}))
        self._write_cache(text, vector, usage)
        self.last_metadata = {
            "provider": "qwen",
            "model": self.model,
            "target_dim": self.target_dim,
            "usage": usage,
            "from_cache": False,
        }
        return {"embedding": vector, **self.last_metadata}

    def embed_text(self, text):
        return self.embed_with_metadata(text)["embedding"]


def build_embedding_provider(
    provider,
    model=None,
    output_dim=512,
    api_key_env="OPENAI_API_KEY",
    qwen_api_key_env="DASHSCOPE_API_KEY",
    qwen_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    qwen_timeout=60,
    qwen_max_retries=3,
    target_dim=None,
    use_dimensions_parameter=True,
    normalize_embedding=True,
    timeout=60,
    max_retries=3,
    cache_root="outputs/embedding_cache",
    use_cache=True,
    force_refresh_cache=False,
    cache_only=False,
):
    provider = str(provider).lower()
    target_dim = int(target_dim if target_dim is not None else output_dim)
    if provider == "mock":
        return MockEmbeddingProvider(output_dim=target_dim, model=model or "mock-embedding")
    if provider == "openai":
        if not model or model == "mock-embedding":
            model = "text-embedding-3-small"
        return OpenAIEmbeddingProvider(
            model=model,
            api_key_env=api_key_env,
            target_dim=target_dim,
            use_dimensions_parameter=use_dimensions_parameter,
            normalize_embedding=normalize_embedding,
            timeout=timeout,
            max_retries=max_retries,
            cache_root=cache_root,
            use_cache=use_cache,
            force_refresh_cache=force_refresh_cache,
            cache_only=cache_only,
        )
    if provider == "qwen":
        if not model or model == "mock-embedding":
            model = "text-embedding-v4"
        return QwenEmbeddingProvider(
            model=model,
            api_key_env=qwen_api_key_env,
            base_url=qwen_base_url,
            target_dim=target_dim,
            use_dimensions_parameter=use_dimensions_parameter,
            normalize_embedding=normalize_embedding,
            timeout=qwen_timeout,
            max_retries=qwen_max_retries,
            cache_root=cache_root,
            use_cache=use_cache,
            force_refresh_cache=force_refresh_cache,
            cache_only=cache_only,
        )
    raise ValueError(f"Unknown embedding provider: {provider}")


def get_embedding_provider(config):
    emb_cfg = config["Embedding"]
    provider = emb_cfg.get("provider", "mock")
    target_dim = emb_cfg.get("target_dim")
    if target_dim is None:
        target_dim = config["Module"].get("d_model", config["Module"].get("trans_dim", 512))
    if provider == "qwen":
        model = emb_cfg.get("model")
        if not model or model == "mock-embedding":
            model = emb_cfg.get("qwen_model", "text-embedding-v4")
    elif provider == "openai":
        model = emb_cfg.get("model")
        if not model or model == "mock-embedding":
            model = emb_cfg.get("openai_model", "text-embedding-3-small")
    else:
        model = emb_cfg.get("model") or emb_cfg.get("embedding_model", "mock-embedding")
    return build_embedding_provider(
        provider=provider,
        model=model,
        output_dim=target_dim,
        api_key_env=emb_cfg.get("openai_api_key_env", "OPENAI_API_KEY"),
        qwen_api_key_env=emb_cfg.get("qwen_api_key_env", "DASHSCOPE_API_KEY"),
        qwen_base_url=emb_cfg.get("qwen_base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        qwen_timeout=emb_cfg.get("qwen_timeout", emb_cfg.get("openai_timeout", 60)),
        qwen_max_retries=emb_cfg.get("qwen_max_retries", emb_cfg.get("openai_max_retries", 3)),
        target_dim=target_dim,
        use_dimensions_parameter=emb_cfg.get("use_dimensions_parameter", True),
        normalize_embedding=emb_cfg.get("normalize_embedding", True),
        timeout=emb_cfg.get("openai_timeout", 60),
        max_retries=emb_cfg.get("openai_max_retries", 3),
        cache_root=emb_cfg.get("cache_root", "outputs/embedding_cache"),
        use_cache=emb_cfg.get("use_cache", True),
        force_refresh_cache=emb_cfg.get("force_refresh_cache", False),
        cache_only=emb_cfg.get("cache_only", False),
    )
