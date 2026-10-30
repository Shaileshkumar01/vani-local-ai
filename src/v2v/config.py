"""Typed configuration loaded from config.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AudioCfg:
    input_device: int | str | None = None
    output_device: int | str | None = None
    sample_rate: int = 16000
    frame_samples: int = 512


@dataclass
class VadCfg:
    threshold: float = 0.5
    min_speech_ms: int = 96
    min_silence_ms: int = 350         # Natural conversational pause threshold (avoids cutting user off)
    extended_silence_ms: int = 650    # Extended cutoff for trailing connectors (and, or, to...)
    filler_silence_ms: int = 900      # Silence needed before discarding fillers (mm, uh)
    pre_roll_ms: int = 300
    max_utterance_s: float = 20.0




@dataclass
class SttCfg:
    model: str = "distil-large-v3"
    device: str = "cuda"
    compute_type: str = "int8_float16"
    language: str | None = "en"
    beam_size: int = 1


@dataclass
class LlmCfg:
    backend: str = "ollama"
    model: str = "qwen2.5:1.5b"
    host: str = "http://127.0.0.1:11434"
    keep_alive: int | str = -1
    temperature: float = 0.7
    max_tokens: int = 160
    num_ctx: int = 2048
    history_turns: int = 24
    system_prompt: str = "You are a concise, friendly voice assistant."

    llamacpp_model_path: str = "models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
    llamacpp_n_gpu_layers: int = -1


@dataclass
class TtsCfg:
    lang_code: str = "a"
    voice: str = "af_heart"
    speed: float = 1.0
    device: str = "cuda"


@dataclass
class ChunkerCfg:
    first_min_chars: int = 12
    min_chars: int = 60


@dataclass
class PipelineCfg:
    barge_in: bool = False
    echo_tail_ms: int = 400
    log_csv: str = "logs/latency.csv"


@dataclass
class ServerCfg:
    host: str = "127.0.0.1"
    port: int = 8000


@dataclass
class Config:
    audio: AudioCfg = field(default_factory=AudioCfg)
    vad: VadCfg = field(default_factory=VadCfg)
    stt: SttCfg = field(default_factory=SttCfg)
    llm: LlmCfg = field(default_factory=LlmCfg)
    tts: TtsCfg = field(default_factory=TtsCfg)
    chunker: ChunkerCfg = field(default_factory=ChunkerCfg)
    pipeline: PipelineCfg = field(default_factory=PipelineCfg)
    server: ServerCfg = field(default_factory=ServerCfg)


def _build(cls: type, data: dict[str, Any] | None):
    obj = cls()
    for f in fields(cls):
        if not data or f.name not in data:
            continue
        current = getattr(obj, f.name)
        value = data[f.name]
        if is_dataclass(current) and isinstance(value, dict):
            value = _build(type(current), value)
        setattr(obj, f.name, value)
    return obj


def load_config(path: str | Path = "config.yaml") -> Config:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    return _build(Config, data or {})
