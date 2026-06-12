import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path


try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import requests
except ImportError:
    requests = None


SCHEMA_VERSION = "semantic_recovery_v1"


SEMANTIC_RECOVERY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "recovered_text": {
            "type": "string",
            "description": "A concise semantic description of the missing view.",
        },
        "semantic_label": {
            "type": "string",
            "description": "A short semantic label or concept summary.",
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
        "rationale": {
            "type": "string",
        },
        "should_abstain": {
            "type": "boolean",
        },
    },
    "required": [
        "recovered_text",
        "semantic_label",
        "confidence",
        "rationale",
        "should_abstain",
    ],
}


def _safe_model_name(model):
    return str(model).replace("/", "_").replace("\\", "_").replace(":", "_")


def _sha256_text(text):
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _strip_json_fence(text):
    raw = str(text).strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    return raw


def _coerce_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value}")


class BaseLLMProvider:
    name = "base"
    model = "base"

    def recover_missing_view(self, prompt, cache_key=None, metadata=None):
        raise NotImplementedError

    def generate(self, prompt, cache_key=None, metadata=None):
        return self.recover_missing_view(prompt, cache_key=cache_key, metadata=metadata)


class MockLLMProvider(BaseLLMProvider):
    def __init__(self, model="mock-llm"):
        self.name = "mock"
        self.model = model

    def recover_missing_view(self, prompt, cache_key=None, metadata=None):
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        confidence = 5 + (int(digest[:8], 16) % 4)
        short_hash = digest[:12]
        description = (
            "Mock recovered semantic description for target view based on available "
            f"multi-view latent summaries. hash={short_hash}"
        )
        raw_response = json.dumps(
            {"description": description, "confidence": confidence},
            sort_keys=True,
        )
        return {
            "description": description,
            "recovered_text": description,
            "semantic_label": f"mock-{short_hash[:6]}",
            "confidence": confidence,
            "rationale": "Deterministic mock response generated from the prompt hash.",
            "should_abstain": False,
            "raw_response": raw_response,
            "provider": "mock",
            "model": self.model,
            "schema_version": SCHEMA_VERSION,
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
        }


class OpenAIChatLLMProvider(BaseLLMProvider):
    def __init__(
        self,
        model="gpt-4o-mini",
        api_key_env="OPENAI_API_KEY",
        timeout=60,
        max_retries=3,
        temperature=0.0,
        max_output_tokens=800,
        use_structured_outputs=True,
        schema_version=SCHEMA_VERSION,
        cache_root="outputs/llm_cache",
        use_cache=True,
        force_refresh_cache=False,
        cache_only=False,
    ):
        self.name = "openai"
        self.model = model
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.use_structured_outputs = use_structured_outputs
        self.schema_version = schema_version
        self.cache_root = Path(cache_root)
        self.use_cache = bool(use_cache)
        self.force_refresh_cache = bool(force_refresh_cache)
        self.cache_only = bool(cache_only)
        self.last_cache_hit = False

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"{api_key_env} is required for provider=openai.")
        if OpenAI is None:
            raise ImportError("openai package is required for OpenAIChatLLMProvider. Please install openai.")
        self.client = OpenAI(api_key=api_key, timeout=timeout, max_retries=max_retries)

    def _cache_path(self, prompt):
        prompt_hash = _sha256_text(prompt)
        return self.cache_root / "openai" / _safe_model_name(self.model) / f"{prompt_hash}.json"

    def _load_cache(self, prompt):
        path = self._cache_path(prompt)
        if self.use_cache and not self.force_refresh_cache and path.exists():
            with open(path, "r", encoding="utf-8-sig") as f:
                cached = json.load(f)
            response = cached["response"]
            response["cache_path"] = path.as_posix()
            self.last_cache_hit = True
            return response
        if self.cache_only:
            raise RuntimeError(f"LLM cache miss for prompt hash {_sha256_text(prompt)} and cache_only=True.")
        self.last_cache_hit = False
        return None

    def _write_cache(self, prompt, response):
        if not self.use_cache:
            return
        path = self._cache_path(prompt)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "provider": "openai",
                    "model": self.model,
                    "prompt_hash": _sha256_text(prompt),
                    "prompt": prompt,
                    "response": response,
                    "usage": response.get("usage", {}),
                    "created_at": _utc_now(),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    def _extract_response_text(self, response):
        if hasattr(response, "output_text") and response.output_text:
            return response.output_text
        if isinstance(response, dict) and response.get("output_text"):
            return response["output_text"]
        output = getattr(response, "output", None)
        if output is None and isinstance(response, dict):
            output = response.get("output")
        if output:
            for item in output:
                content = getattr(item, "content", None)
                if content is None and isinstance(item, dict):
                    content = item.get("content")
                if not content:
                    continue
                for part in content:
                    text = getattr(part, "text", None)
                    if text is None and isinstance(part, dict):
                        text = part.get("text")
                    if text:
                        return text
        raise ValueError("OpenAI response did not contain output text.")

    def _usage_dict(self, response):
        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")
        if usage is None:
            return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        if isinstance(usage, dict):
            return {
                "input_tokens": int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0),
                "output_tokens": int(usage.get("output_tokens") or usage.get("completion_tokens") or 0),
                "total_tokens": int(usage.get("total_tokens") or 0),
            }
        return {
            "input_tokens": int(getattr(usage, "input_tokens", getattr(usage, "prompt_tokens", 0)) or 0),
            "output_tokens": int(getattr(usage, "output_tokens", getattr(usage, "completion_tokens", 0)) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        }

    def _call_responses_api(self, prompt):
        return self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You recover concise semantic descriptions for missing views in incomplete "
                        "multi-view clustering. Return only fields required by the JSON schema."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            text={
                "format": {
                    "type": "json_schema",
                    "name": self.schema_version,
                    "schema": SEMANTIC_RECOVERY_SCHEMA,
                    "strict": True,
                }
            },
        )

    def _call_chat_completions_api(self, prompt):
        return self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return strict JSON with recovered_text, semantic_label, confidence, "
                        "rationale, and should_abstain."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": self.schema_version,
                    "schema": SEMANTIC_RECOVERY_SCHEMA,
                    "strict": True,
                },
            },
        )

    def _extract_chat_text(self, response):
        choice = response.choices[0]
        return choice.message.content

    def _validate_payload(self, payload):
        missing = [key for key in SEMANTIC_RECOVERY_SCHEMA["required"] if key not in payload]
        if missing:
            raise ValueError(f"OpenAI semantic response missing required keys: {missing}")
        confidence = float(payload["confidence"])
        if confidence < 0 or confidence > 1:
            raise ValueError(f"OpenAI semantic confidence must be in [0, 1], got {confidence}")
        return {
            "recovered_text": str(payload["recovered_text"]),
            "description": str(payload["recovered_text"]),
            "semantic_label": str(payload["semantic_label"]),
            "confidence": confidence,
            "rationale": str(payload["rationale"]),
            "should_abstain": bool(payload["should_abstain"]),
        }

    def recover_missing_view(self, prompt, cache_key=None, metadata=None):
        cached = self._load_cache(prompt)
        if cached is not None:
            return cached

        try:
            try:
                response = self._call_responses_api(prompt)
                raw_text = self._extract_response_text(response)
            except AttributeError:
                response = self._call_chat_completions_api(prompt)
                raw_text = self._extract_chat_text(response)
            payload = json.loads(raw_text)
            parsed = self._validate_payload(payload)
            usage = self._usage_dict(response)
            result = {
                **parsed,
                "raw_text": raw_text,
                "raw_response": {
                    "id": getattr(response, "id", None),
                    "model": getattr(response, "model", self.model),
                },
                "provider": "openai",
                "model": self.model,
                "schema_version": self.schema_version,
                "usage": usage,
            }
            self._write_cache(prompt, result)
            return result
        except Exception as exc:
            raise RuntimeError(f"OpenAI LLM query failed: {type(exc).__name__}: {exc}") from exc


class QwenChatLLMProvider(BaseLLMProvider):
    def __init__(
        self,
        model="qwen-plus",
        api_key_env="DASHSCOPE_API_KEY",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout=60,
        max_retries=3,
        temperature=0.0,
        max_output_tokens=800,
        use_structured_outputs=True,
        schema_version=SCHEMA_VERSION,
        cache_root="outputs/llm_cache",
        use_cache=True,
        force_refresh_cache=False,
        cache_only=False,
    ):
        self.name = "qwen"
        self.model = model
        self.api_key_env = api_key_env
        self.base_url = base_url.rstrip("/")
        self.chat_url = self.base_url + "/chat/completions"
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.use_structured_outputs = use_structured_outputs
        self.schema_version = schema_version
        self.cache_root = Path(cache_root)
        self.use_cache = bool(use_cache)
        self.force_refresh_cache = bool(force_refresh_cache)
        self.cache_only = bool(cache_only)
        self.last_cache_hit = False

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"{api_key_env} is required for provider=qwen.")
        if requests is None:
            raise ImportError("requests package is required for QwenChatLLMProvider. Please install requests.")
        self.api_key = api_key

    def _cache_path(self, prompt):
        prompt_hash = _sha256_text(prompt)
        return self.cache_root / "qwen" / _safe_model_name(self.model) / f"{prompt_hash}.json"

    def _load_cache(self, prompt):
        path = self._cache_path(prompt)
        if self.use_cache and not self.force_refresh_cache and path.exists():
            with open(path, "r", encoding="utf-8-sig") as f:
                cached = json.load(f)
            response = cached["response"]
            response["cache_path"] = path.as_posix()
            self.last_cache_hit = True
            return response
        if self.cache_only:
            raise RuntimeError(f"Qwen LLM cache miss for prompt hash {_sha256_text(prompt)} and cache_only=True.")
        self.last_cache_hit = False
        return None

    def _write_cache(self, prompt, response):
        if not self.use_cache:
            return
        path = self._cache_path(prompt)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "provider": "qwen",
                    "model": self.model,
                    "base_url": self.base_url,
                    "prompt_hash": _sha256_text(prompt),
                    "prompt": prompt,
                    "response": response,
                    "usage": response.get("usage", {}),
                    "created_at": _utc_now(),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    def _system_prompt(self):
        return (
            "You are a strict JSON generator for semantic view recovery. "
            "You must return valid JSON only. Do not return markdown. "
            "Do not wrap the answer in ```json. The JSON object must contain: "
            "recovered_text: string, semantic_label: string, confidence: number between 0 and 1, "
            "rationale: string, should_abstain: boolean."
        )

    def _request_payload(self, prompt):
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
        }

    def _post_json(self, payload):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error = None
        for attempt in range(int(self.max_retries) + 1):
            try:
                response = requests.post(
                    self.chat_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                if response.status_code in (401, 403):
                    raise RuntimeError(
                        f"Qwen authentication failed with HTTP {response.status_code}. Check {self.api_key_env}."
                    )
                if response.status_code == 429 or response.status_code >= 500:
                    body = response.text[:500]
                    last_error = RuntimeError(f"Qwen HTTP {response.status_code}: {body}")
                    if attempt < int(self.max_retries):
                        time.sleep(min(2 ** attempt, 8))
                        continue
                    raise last_error
                if response.status_code >= 400:
                    raise RuntimeError(f"Qwen HTTP {response.status_code}: {response.text[:500]}")
                return response.json()
            except requests.Timeout as exc:
                last_error = exc
                if attempt < int(self.max_retries):
                    time.sleep(min(2 ** attempt, 8))
                    continue
                raise RuntimeError(f"Qwen LLM request timed out after {self.timeout}s.") from exc
            except requests.RequestException as exc:
                last_error = exc
                if attempt < int(self.max_retries):
                    time.sleep(min(2 ** attempt, 8))
                    continue
                raise RuntimeError(f"Qwen LLM request failed: {type(exc).__name__}: {exc}") from exc
        raise RuntimeError(f"Qwen LLM request failed: {last_error}")

    def _extract_text(self, response_json):
        try:
            content = response_json["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("Qwen response did not contain choices[0].message.content.") from exc
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    parts.append(str(part.get("text") or part.get("content") or ""))
                else:
                    parts.append(str(part))
            content = "".join(parts)
        raw_text = _strip_json_fence(content)
        if not raw_text:
            raise ValueError("Qwen response content was empty.")
        return raw_text

    def _usage_dict(self, response_json):
        usage = response_json.get("usage", {}) if isinstance(response_json, dict) else {}
        return {
            "input_tokens": int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
        }

    def _validate_payload(self, payload):
        missing = [key for key in SEMANTIC_RECOVERY_SCHEMA["required"] if key not in payload]
        if missing:
            raise ValueError(f"Qwen semantic response missing required keys: {missing}")
        try:
            confidence = float(payload["confidence"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Qwen semantic confidence is not numeric: {payload.get('confidence')}") from exc
        confidence = float(max(0.0, min(1.0, confidence)))
        return {
            "recovered_text": str(payload["recovered_text"]),
            "description": str(payload["recovered_text"]),
            "semantic_label": str(payload["semantic_label"]),
            "confidence": confidence,
            "rationale": str(payload["rationale"]),
            "should_abstain": _coerce_bool(payload["should_abstain"]),
        }

    def recover_missing_view(self, prompt, cache_key=None, metadata=None):
        cached = self._load_cache(prompt)
        if cached is not None:
            return cached

        try:
            response_json = self._post_json(self._request_payload(prompt))
            raw_text = self._extract_text(response_json)
            try:
                payload = json.loads(raw_text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Qwen response was not valid JSON: {raw_text[:500]}") from exc
            parsed = self._validate_payload(payload)
            usage = self._usage_dict(response_json)
            result = {
                **parsed,
                "raw_text": raw_text,
                "raw_response": {
                    "id": response_json.get("id"),
                    "model": response_json.get("model", self.model),
                },
                "provider": "qwen",
                "model": self.model,
                "schema_version": self.schema_version,
                "usage": usage,
            }
            self._write_cache(prompt, result)
            return result
        except Exception as exc:
            raise RuntimeError(f"Qwen LLM query failed: {type(exc).__name__}: {exc}") from exc


def build_llm_provider(
    provider,
    model=None,
    temperature=0.0,
    api_key_env="OPENAI_API_KEY",
    qwen_api_key_env="DASHSCOPE_API_KEY",
    qwen_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    qwen_timeout=60,
    qwen_max_retries=3,
    qwen_temperature=0.0,
    qwen_max_output_tokens=800,
    timeout=60,
    max_retries=3,
    max_output_tokens=800,
    use_structured_outputs=True,
    schema_version=SCHEMA_VERSION,
    cache_root="outputs/llm_cache",
    use_cache=True,
    force_refresh_cache=False,
    cache_only=False,
):
    provider = str(provider).lower()
    if provider == "mock":
        return MockLLMProvider(model=model or "mock-llm")
    if provider == "openai":
        if not model or model == "mock-llm":
            model = "gpt-4o-mini"
        return OpenAIChatLLMProvider(
            model=model,
            api_key_env=api_key_env,
            timeout=timeout,
            max_retries=max_retries,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            use_structured_outputs=use_structured_outputs,
            schema_version=schema_version,
            cache_root=cache_root,
            use_cache=use_cache,
            force_refresh_cache=force_refresh_cache,
            cache_only=cache_only,
        )
    if provider == "qwen":
        if not model or model == "mock-llm":
            model = "qwen-plus"
        return QwenChatLLMProvider(
            model=model,
            api_key_env=qwen_api_key_env,
            base_url=qwen_base_url,
            timeout=qwen_timeout,
            max_retries=qwen_max_retries,
            temperature=qwen_temperature,
            max_output_tokens=qwen_max_output_tokens,
            use_structured_outputs=use_structured_outputs,
            schema_version=schema_version,
            cache_root=cache_root,
            use_cache=use_cache,
            force_refresh_cache=force_refresh_cache,
            cache_only=cache_only,
        )
    raise ValueError(f"Unknown LLM provider: {provider}")


def get_llm_provider(config):
    llm_cfg = config["LLM"]
    provider = llm_cfg.get("provider", "mock")
    if provider == "qwen":
        model = llm_cfg.get("model")
        if not model or model == "mock-llm":
            model = llm_cfg.get("qwen_model", "qwen-plus")
    elif provider == "openai":
        model = llm_cfg.get("model")
        if not model or model == "mock-llm":
            model = llm_cfg.get("openai_model", "gpt-4o-mini")
    else:
        model = llm_cfg.get("model") or llm_cfg.get("llm_model", "mock-llm")
    return build_llm_provider(
        provider=provider,
        model=model,
        temperature=llm_cfg.get("openai_temperature", llm_cfg.get("temperature", 0.0)),
        api_key_env=llm_cfg.get("openai_api_key_env", "OPENAI_API_KEY"),
        timeout=llm_cfg.get("openai_timeout", 60),
        max_retries=llm_cfg.get("openai_max_retries", 3),
        max_output_tokens=llm_cfg.get("openai_max_output_tokens", 800),
        qwen_api_key_env=llm_cfg.get("qwen_api_key_env", "DASHSCOPE_API_KEY"),
        qwen_base_url=llm_cfg.get("qwen_base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        qwen_timeout=llm_cfg.get("qwen_timeout", 60),
        qwen_max_retries=llm_cfg.get("qwen_max_retries", 3),
        qwen_temperature=llm_cfg.get("qwen_temperature", 0.0),
        qwen_max_output_tokens=llm_cfg.get("qwen_max_output_tokens", 800),
        use_structured_outputs=llm_cfg.get("use_structured_outputs", True),
        schema_version=llm_cfg.get("response_schema_version", SCHEMA_VERSION),
        cache_root=llm_cfg.get("cache_root", "outputs/llm_cache"),
        use_cache=llm_cfg.get("use_cache", True),
        force_refresh_cache=llm_cfg.get("force_refresh_cache", False),
        cache_only=llm_cfg.get("cache_only", False),
    )
