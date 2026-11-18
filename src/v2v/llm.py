"""Streaming LLM backends (Ollama by default, llama.cpp optional) + conversation memory."""

from __future__ import annotations

from collections import deque
import threading
from typing import Iterator, Protocol

from .config import LlmCfg


class LLMBackend(Protocol):
    def warmup(self) -> None: ...
    def stream(self, messages: list[dict], max_tokens: int | None = None, temperature: float | None = None) -> Iterator[str]: ...
    def complete(self, messages: list[dict], max_tokens: int | None = None, temperature: float | None = None) -> str: ...


class OllamaLLM:
    """Talks to the local Ollama server (it manages CUDA + quantized GGUF models for us)."""

    def __init__(self, cfg: LlmCfg):
        import ollama

        self.cfg = cfg
        self.client = ollama.Client(host=cfg.host)
        self.options = {
            "temperature": cfg.temperature,
            "num_predict": cfg.max_tokens,
            "num_ctx": cfg.num_ctx,
        }

    def warmup(self) -> None:
        try:
            self.client.chat(
                model=self.cfg.model,
                messages=[{"role": "user", "content": "hi"}],
                options={**self.options, "num_predict": 1},
                keep_alive=self.cfg.keep_alive,
            )
        except Exception as e:  # connection refused / model not pulled
            raise RuntimeError(
                f"Could not reach Ollama model '{self.cfg.model}' at {self.cfg.host}: {e}\n"
                f"  -> make sure Ollama is running and run:  ollama pull {self.cfg.model}"
            ) from e

    def stream(self, messages: list[dict], max_tokens: int | None = None, temperature: float | None = None) -> Iterator[str]:
        opts = dict(self.options)
        if max_tokens is not None:
            opts["num_predict"] = max_tokens
        if temperature is not None:
            opts["temperature"] = temperature
        for part in self.client.chat(
            model=self.cfg.model,
            messages=messages,
            stream=True,
            options=opts,
            keep_alive=self.cfg.keep_alive,
        ):
            tok = part["message"]["content"]
            if tok:
                yield tok

    def complete(self, messages: list[dict], max_tokens: int | None = None, temperature: float | None = None) -> str:
        opts = dict(self.options)
        if max_tokens is not None:
            opts["num_predict"] = max_tokens
        if temperature is not None:
            opts["temperature"] = temperature
        r = self.client.chat(
            model=self.cfg.model, messages=messages, options=opts, keep_alive=self.cfg.keep_alive
        )
        return r["message"]["content"]


class LlamaCppLLM:
    """In-process llama.cpp (pip install -e .[llamacpp] with a CUDA build)."""

    def __init__(self, cfg: LlmCfg):
        from .cuda_setup import add_cuda_dll_dirs

        add_cuda_dll_dirs()
        from llama_cpp import Llama

        self.cfg = cfg
        self._lock = threading.Lock()
        self.llm = Llama(
            model_path=cfg.llamacpp_model_path,
            n_gpu_layers=cfg.llamacpp_n_gpu_layers,
            n_ctx=cfg.num_ctx,
            verbose=False,
        )

    def warmup(self) -> None:
        self.complete([{"role": "user", "content": "hi"}], max_tokens=1)

    def stream(self, messages: list[dict], max_tokens: int | None = None, temperature: float | None = None) -> Iterator[str]:
        mt = max_tokens if max_tokens is not None else self.cfg.max_tokens
        temp = temperature if temperature is not None else self.cfg.temperature
        with self._lock:
            for chunk in self.llm.create_chat_completion(
                messages=messages, stream=True, max_tokens=mt, temperature=temp
            ):
                tok = chunk["choices"][0]["delta"].get("content")
                if tok:
                    yield tok

    def complete(self, messages: list[dict], max_tokens: int | None = None, temperature: float | None = None) -> str:
        mt = max_tokens if max_tokens is not None else self.cfg.max_tokens
        temp = temperature if temperature is not None else self.cfg.temperature
        with self._lock:
            r = self.llm.create_chat_completion(
                messages=messages, max_tokens=mt, temperature=temp
            )
        return r["choices"][0]["message"]["content"]


def make_llm(cfg: LlmCfg) -> LLMBackend:
    if cfg.backend == "ollama":
        return OllamaLLM(cfg)
    if cfg.backend == "llamacpp":
        return LlamaCppLLM(cfg)
    raise ValueError(f"Unknown llm.backend '{cfg.backend}' (use 'ollama' or 'llamacpp')")


class Conversation:
    def __init__(self, system_prompt: str, history_turns: int):
        self.system = {"role": "system", "content": system_prompt.strip()}
        self.history: deque[dict] = deque(maxlen=history_turns * 2)

    def messages_for(self, user_text: str) -> list[dict]:
        return [self.system, *self.history, {"role": "user", "content": user_text}]

    def add(self, user_text: str, reply: str) -> None:
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": reply})
