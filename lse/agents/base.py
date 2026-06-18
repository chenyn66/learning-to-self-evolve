from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from vllm import LLM
import torch
import gc
import os
import random
import time

from lse.model_merger import resolve_model_path
from lse.helpers import truncate_repetitive_suffix
import hashlib
import json
import threading
from datetime import datetime, timezone


class BaseAgent(ABC):
    """
    Abstract base class for self-evolving agents.

    The concrete "act" signature may vary by task (step-wise decisions vs.
    batch inferences). Implementations should accept flexible arguments.
    """

    @abstractmethod
    def reset(self, n_sims: int = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def _act_impl(self, *args, **kwargs):
        """Take an action or perform inference. Return task-specific output."""
        raise NotImplementedError

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        """Optional environment feedback hook (no-op for some tasks)."""
        raise NotImplementedError

    @abstractmethod
    def get_history(self) -> List[str]:
        """Return a list of stringified histories for logging/debugging."""
        raise NotImplementedError

    @abstractmethod
    def dummy_self_evolve(self, summary: List[Dict[str, Any]], **kwargs):
        """Return a chat-like history representing a dummy evolution round."""
        raise NotImplementedError

    @abstractmethod
    def _self_evolve_impl(self, summary: List[Dict[str, Any]], **kwargs):
        """Run the real self-evolution and return the chat history."""
        raise NotImplementedError

    @abstractmethod
    def format_history(self, history: List[Dict[str, str]] | List[str]) -> str:
        """Pretty-print a chat or textual history as a single string."""
        raise NotImplementedError

    # wrapper for act and self_evolve

    def act(self, *args, **kwargs):
        self.mode = "act"
        return self._act_impl(*args, **kwargs)

    def self_evolve(self, *args, **kwargs):
        self.mode = "self_evolve"
        return self._self_evolve_impl(*args, **kwargs)

    # -----------------------
    # Shared chat utilities
    # -----------------------
    @staticmethod
    def _build_openai_client_kwargs(base_url: str, api_key: str) -> Dict[str, Any]:
        """Construct OpenAI client kwargs with env-tunable timeout/retries.

        `LSE_OPENAI_TIMEOUT`:
            If set to a positive float, passed as OpenAI/httpx timeout.
        `LSE_OPENAI_CLIENT_MAX_RETRIES`:
            OpenAI SDK-level retries. Defaults to 0 because this class already
            applies explicit retry logic in `call_oai_rm_llm`.
        """
        kwargs: Dict[str, Any] = {
            "base_url": base_url,
            "api_key": api_key,
        }

        timeout_raw = (os.environ.get("LSE_OPENAI_TIMEOUT", "") or "").strip()
        if timeout_raw:
            try:
                timeout_s = float(timeout_raw)
                if timeout_s > 0:
                    kwargs["timeout"] = timeout_s
            except ValueError:
                print(
                    f"Warning: invalid LSE_OPENAI_TIMEOUT={timeout_raw!r}; "
                    "falling back to OpenAI default timeout."
                )

        max_retries_raw = (os.environ.get("LSE_OPENAI_CLIENT_MAX_RETRIES", "0") or "0").strip()
        try:
            kwargs["max_retries"] = max(int(max_retries_raw), 0)
        except ValueError:
            kwargs["max_retries"] = 0
            print(
                f"Warning: invalid LSE_OPENAI_CLIENT_MAX_RETRIES={max_retries_raw!r}; "
                "using 0."
            )

        return kwargs

    @staticmethod
    def _is_timeout_exception(exc: Exception) -> bool:
        msg = str(exc).lower()
        return "timed out" in msg or "timeout" in msg

    @staticmethod
    def _int_env(name: str, default: int) -> int:
        raw = (os.environ.get(name, str(default)) or str(default)).strip()
        try:
            return int(raw)
        except ValueError:
            return default

    @staticmethod
    def _float_env(name: str, default: float) -> float:
        raw = (os.environ.get(name, str(default)) or str(default)).strip()
        try:
            return float(raw)
        except ValueError:
            return default

    def _get_api_inflight_semaphore(self):
        """Optional per-process cap to smooth API request bursts.

        Set `LSE_API_INFLIGHT_LIMIT` to a positive integer to enable.
        If unset or <= 0, no additional cap is applied.
        """
        limit = self._int_env("LSE_API_INFLIGHT_LIMIT", 0)
        if limit <= 0:
            return None
        if (
            not hasattr(self, "_api_inflight_semaphore")
            or getattr(self, "_api_inflight_limit", None) != limit
        ):
            self._api_inflight_limit = limit
            self._api_inflight_semaphore = threading.BoundedSemaphore(value=limit)
        return self._api_inflight_semaphore

    def _get_timeout_log_state(self):
        """Thread-safe state for timeout prompt logging."""
        if not hasattr(self, "_timeout_log_lock"):
            self._timeout_log_lock = threading.Lock()
            self._timeout_prompt_seen = set()
            self._timeout_prompt_counts = {}
        return self._timeout_log_lock, self._timeout_prompt_seen, self._timeout_prompt_counts

    def _extract_timeout_prompt(self, msg: List[Dict[str, str]]) -> str:
        """Serialize full chat payload (all roles) for timeout diagnostics."""
        try:
            # Use deterministic ordering so the same full message payload hashes identically.
            return json.dumps(msg, ensure_ascii=False, sort_keys=True)
        except Exception:
            return str(msg)

    def _format_timeout_prompt_preview(self, msg: List[Dict[str, str]]) -> str:
        """Build a readable preview that preserves role + content for every message."""
        lines: List[str] = []
        for i, m in enumerate(msg):
            role = str(m.get("role", "unknown"))
            content = m.get("content", "")
            if not isinstance(content, str):
                try:
                    content = json.dumps(content, ensure_ascii=False, sort_keys=True)
                except Exception:
                    content = str(content)
            lines.append(f"[{i}:{role}] {content}")
        if lines:
            return "\n\n".join(lines)
        return self._extract_timeout_prompt(msg)

    def _log_timeout_prompt(
        self,
        msg: List[Dict[str, str]],
        exc: Exception,
        attempt: int,
        n_attempts: int,
        sleep_s: float,
        model_name: Optional[str] = None,
    ) -> str:
        """Log timeout-causing prompt context; return compact suffix for retry logs."""
        prompt_payload = self._extract_timeout_prompt(msg)
        prompt_hash = hashlib.sha256(prompt_payload.encode("utf-8", errors="ignore")).hexdigest()[:16]
        prompt_text = self._format_timeout_prompt_preview(msg)
        prompt_len = len(prompt_text)
        preview_chars = max(64, self._int_env("LSE_TIMEOUT_PROMPT_PREVIEW_CHARS", 512))
        preview = prompt_text[:preview_chars]
        if len(prompt_text) > preview_chars:
            preview += f"... [truncated {len(prompt_text) - preview_chars} chars]"

        lock, seen, counts = self._get_timeout_log_state()
        with lock:
            counts[prompt_hash] = counts.get(prompt_hash, 0) + 1
            seen_count = counts[prompt_hash]
            is_first_seen = prompt_hash not in seen
            if is_first_seen:
                seen.add(prompt_hash)

        # Always emit a compact line for each timeout event.
        print(
            "TimeoutPrompt "
            f"hash={prompt_hash} len={prompt_len} seen={seen_count} "
            f"attempt={attempt + 1}/{n_attempts} sleep_s={sleep_s:.2f} "
            f"model={model_name or self.model_path}"
        )
        # Emit prompt preview once per unique prompt by default.
        if is_first_seen or os.environ.get("LSE_TIMEOUT_LOG_EVERY_RETRY", "0") == "1":
            print(f"TimeoutPrompt preview: {preview}")

        # Optional JSONL sink for offline analysis.
        log_path = (os.environ.get("LSE_TIMEOUT_PROMPT_LOG_PATH", "") or "").strip()
        if log_path:
            try:
                json.dumps(msg, ensure_ascii=False)
                messages_value: Any = msg
            except Exception:
                messages_value = str(msg)

            record = {
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                "event": "timeout_retry",
                "prompt_hash": prompt_hash,
                "prompt_len": prompt_len,
                "seen_count": seen_count,
                "attempt": attempt + 1,
                "n_attempts": n_attempts,
                "sleep_s": sleep_s,
                "error": str(exc),
                "model": model_name or self.model_path,
                "is_first_seen_prompt": is_first_seen,
                "prompt_hash_basis": "full_messages_payload",
                "messages": messages_value,
                "messages_payload": prompt_payload,
            }
            if is_first_seen:
                # Keep a human-readable role-annotated rendering for quick inspection.
                record["prompt"] = prompt_text
            try:
                dir_name = os.path.dirname(log_path)
                if dir_name:
                    os.makedirs(dir_name, exist_ok=True)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as write_exc:
                print(f"Warning: failed to write timeout prompt log to {log_path}: {write_exc}")

        return f"[prompt_hash={prompt_hash} seen={seen_count}]"

    def call_oai_rm_llm(
        self,
        msg: List[Dict[str, str]],
        sampling_params,
        retry_count: int = 3,
        model_name: Optional[str] = None,
    ) -> List[str]:
        """Call an OpenAI-compatible chat.completions API with basic retry.

        Expects subclasses to set: self.llm (client), self.model_path, self.max_tokens.
        """
        result: List[str] = []
        retry_override = self._int_env("LSE_OAI_RM_RETRY_COUNT", 0)
        if retry_override > 0:
            n_attempts = retry_override
        else:
            n_attempts = max(int(retry_count), 1)
        backoff_base_s = max(self._float_env("LSE_OAI_RETRY_BACKOFF_BASE_S", 1.0), 0.0)
        backoff_cap_s = max(self._float_env("LSE_OAI_RETRY_BACKOFF_MAX_S", 30.0), backoff_base_s)
        retry_jitter_s = max(self._float_env("LSE_OAI_RETRY_JITTER_S", 0.25), 0.0)
        api_sem = self._get_api_inflight_semaphore()
        for attempt in range(n_attempts):
            try:
                # NOTE: vLLM's OpenAI-compatible /v1/chat/completions often rejects
                # logprobs/top_logprobs. Previously we (incorrectly) mapped sampling
                # top_k -> top_logprobs (these are different concepts), which can trigger 400s.
                # Disable logprobs by default for API mode; can be re-enabled via LSE_ENABLE_LOGPROBS=1.
                req_kwargs = {
                    "model": model_name or self.model_path,
                    "messages": msg,
                    "temperature": sampling_params.temperature,
                    "top_p": sampling_params.top_p,
                    "max_tokens": sampling_params.max_tokens,
                    "presence_penalty": sampling_params.presence_penalty,
                    "n": sampling_params.n,
                }

                # vLLM supports sampling params not in the OpenAI schema (e.g. top_k, min_p).
                # These can be passed via `extra_body` in the OpenAI client request.
                extra_body = {}
                top_k = getattr(sampling_params, "top_k", None)
                if top_k is not None:
                    extra_body["top_k"] = int(top_k)
                min_p = getattr(sampling_params, "min_p", None)
                if min_p is not None:
                    extra_body["min_p"] = float(min_p)
                if extra_body:
                    req_kwargs["extra_body"] = extra_body

                if os.environ.get("LSE_ENABLE_LOGPROBS", "0") == "1":
                    top_k = getattr(sampling_params, "top_k", None)
                    if top_k:
                        req_kwargs["logprobs"] = True
                        req_kwargs["top_logprobs"] = int(top_k)

                if api_sem is None:
                    response = self.llm.chat.completions.create(**req_kwargs)
                else:
                    with api_sem:
                        response = self.llm.chat.completions.create(**req_kwargs)
                result = [response.choices[i].message.content for i in range(sampling_params.n)]
                if len(result) != int(sampling_params.n):
                    raise RuntimeError(
                        f"Expected {sampling_params.n} choices, got {len(result)}"
                    )
                return result
            except Exception as exc:
                # Retry on transient API failures (rate limits, timeouts, 5xx, etc.).
                # Only raise after the final attempt.
                timeout_suffix = ""
                sleep_s = min(backoff_base_s * (2**attempt), backoff_cap_s) + random.random() * retry_jitter_s
                if self._is_timeout_exception(exc):
                    timeout_suffix = self._log_timeout_prompt(
                        msg=msg,
                        exc=exc,
                        attempt=attempt,
                        n_attempts=n_attempts,
                        sleep_s=sleep_s,
                        model_name=model_name,
                    )
                if attempt >= n_attempts - 1:
                    print("Exception: ", exc)
                    print("msg: ", msg)
                    raise
                print(
                    f"Exception (attempt {attempt + 1}/{n_attempts}): {exc}. "
                    f"Retrying in {sleep_s:.2f}s... {timeout_suffix}"
                )
                time.sleep(sleep_s)
        return result

    def _parallel_api_chat(
        self,
        batched_messages: List[List[Dict[str, str]]],
        sampling_params,
        model_name: Optional[str] = None,
    ) -> List[List[str]]:
        results = [None] * len(batched_messages)
        num_worker = int(os.environ.get("NUM_API_WORKERS", 32))
        with ThreadPoolExecutor(max_workers=num_worker) as executor:
            future_to_index = {
                executor.submit(
                    self.call_oai_rm_llm,
                    msg,
                    sampling_params,
                    model_name=model_name,
                ): idx
                for idx, msg in enumerate(batched_messages)
            }
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                result = future.result()
                results[idx] = result
        return results

    def _init_vllm_models(self):
        """Initialize vLLM models with sleep mode enabled."""
        from torch import cuda
        assert cuda.device_count() > 0, "No GPUs available"
        
        # If the user points at an FSDP-sharded checkpoint dir (e.g., a verl `actor/` dir),
        # transparently merge it to a HuggingFace model dir and use that for vLLM loading.
        orig_act = getattr(self, "act_model", None)
        orig_evolve = getattr(self, "evolve_model", None)
        self.act_model = resolve_model_path(orig_act, trust_remote_code=True)
        self.evolve_model = (
            None if orig_evolve is None else resolve_model_path(orig_evolve, trust_remote_code=True)
        )
        # Keep `model_path` consistent for local (non-API) runs.
        if getattr(self, "model_path", None) == orig_act:
            self.model_path = self.act_model

        self.llms = {}
        # Ensure unique models
        models = {self.act_model}
        if self.evolve_model:
            models.add(self.evolve_model)
        
        for m in models:
            if m is None: continue
            print(f"Initializing model {m} in sleep mode...")
            llm = LLM(
                model=m,
                gpu_memory_utilization=0.8,
                tensor_parallel_size=cuda.device_count(),
                enable_sleep_mode=True,
                trust_remote_code=True,
            )
            llm.sleep(level=2)
            self.llms[m] = llm
            
        self.active_model_name = None
        self.llm = None

    def _wake_model(self, model_name: str):
        if self.use_api:
            return

        if self.active_model_name == model_name:
            return

        if self.active_model_name:
            print(f"Putting {self.active_model_name} to sleep...")
            self.llms[self.active_model_name].sleep(level=2)
        
        print(f"Waking up {model_name}...")
        llm = self.llms[model_name]
        llm.wake_up(tags=["weights"])
        llm.collective_rpc("reload_weights")
        llm.wake_up(tags=["kv_cache"])
        
        self.active_model_name = model_name
        self.llm = llm

    def _batch_chat(
        self,
        batched_messages: List[List[Dict[str, str]]],
        params: Any,
        checker=None,
        max_tries: int = 3,
        no_think: bool = False,
        n_samples: int = 1,
    ) -> List[List[str]]:
        """Shared batch chat with retry and validation.

        If self.use_api is True, uses _parallel_api_chat; otherwise assumes a
        vLLM-style engine on self.llm with .chat(batched_messages, sampling_params).
        """

        # check if using the correct model
        if self.mode == "act":
            model_to_use = self.act_model
        elif self.mode == "self_evolve":
            model_to_use = self.evolve_model
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        params.n = n_samples

        if not self.use_api:
            self._wake_model(model_to_use)

        results = [None] * len(batched_messages)
        remaining_indices = list(range(len(batched_messages)))
        remaining_messages = batched_messages.copy()

        if no_think:
            for msg in remaining_messages:
                msg[-1]["content"] = msg[-1]["content"] + "\n/no_think"

        # Config-driven repetitive output truncation (defaults live in configs/base.yaml).
        trunc_cfg = getattr(getattr(self, "args", None), "truncate_repetitive", None)
        truncate_rep = bool(getattr(trunc_cfg, "enabled", False)) if trunc_cfg is not None else False
        min_run = int(getattr(trunc_cfg, "min_run_length", 200)) if trunc_cfg is not None else 200
        lookback = int(getattr(trunc_cfg, "lookback", 8192)) if trunc_cfg is not None else 8192
        add_marker = bool(getattr(trunc_cfg, "add_marker", False)) if trunc_cfg is not None else False

        for attempt in range(max_tries):
            if not remaining_indices:
                break

            # Generate responses for remaining prompts
            if self.use_api:
                responses = self._parallel_api_chat(
                    remaining_messages,
                    sampling_params=params,
                    model_name=model_to_use,
                )
            else:
                outputs = self.llm.chat(remaining_messages, sampling_params=params, use_tqdm=False)
                responses = [[] for _ in range(len(remaining_messages))]
                for idx, o in enumerate(outputs):
                    for i in range(n_samples):
                        responses[idx].append(o.outputs[i].text.strip())

            if truncate_rep:
                # Truncate pathological repetitive suffix loops before validation/logging.
                for idx in range(len(responses)):
                    cleaned = []
                    for r in responses[idx]:
                        r2, _info = truncate_repetitive_suffix(
                            r,
                            min_run_length=min_run,
                            lookback=lookback,
                            add_marker=add_marker,
                        )
                        cleaned.append(r2)
                    responses[idx] = cleaned
                        
            # Check each response
            new_remaining_indices = []
            new_remaining_messages = []

            for i, (orig_idx, response) in enumerate(zip(remaining_indices, responses)):
                if checker is None or any(checker(r) for r in response):
                    results[orig_idx] = response
                else:
                    results[orig_idx] = response
                    new_remaining_indices.append(orig_idx)
                    new_remaining_messages.append(remaining_messages[i])

            remaining_indices = new_remaining_indices
            remaining_messages = new_remaining_messages

            if remaining_indices:
                print(
                    f"Attempt {attempt + 1}: {len(remaining_indices)} responses failed validation, retrying..."
                )

        # Handle any remaining failures
        for idx in remaining_indices:
            print(
                f"Warning: Failed to get valid response for prompt {idx} after {max_tries} attempts"
            )
            # results[idx] = [''] * n_samples

        assert all(len(r) == n_samples for r in results), f"Expected {n_samples} responses per prompt, got {[len(r) for r in results]}"
  
        return results

__all__ = ["BaseAgent"]


