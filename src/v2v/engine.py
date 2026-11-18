"""Single shared ModelHub.

Loads Silero VAD, Whisper STT, LLM backend, and Kokoro TTS once on startup,
keeps them resident in memory/VRAM, and shares them across client sessions.
"""

from __future__ import annotations

import logging
import time

from .config import Config
from .llm import Conversation, LLMBackend, make_llm
from .metrics import MetricsLog
from .stt import WhisperSTT
from .tools import ToolRegistry
from .tools.memory_timer import load_memory_prompt_suffix
from .tts import KokoroTTS
from .vad import Endpointer, SileroVAD

logger = logging.getLogger("v2v.engine")


class Engine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.metrics = MetricsLog(cfg.pipeline.log_csv)
        self.tools = ToolRegistry()

        print("\033[36m[Engine]\033[0m Initializing shared models...", flush=True)
        t0 = time.perf_counter()

        # 1. Silero VAD (CPU)
        print("  -> Loading Silero VAD (CPU)...", flush=True)
        self.vad_model = SileroVAD(cfg.audio.sample_rate)

        # 2. Whisper STT (GPU / CTranslate2)
        print(f"  -> Loading faster-whisper '{cfg.stt.model}' on {cfg.stt.device} ({cfg.stt.compute_type})...", flush=True)
        self.stt = WhisperSTT(cfg.stt)

        # 3. LLM Backend (Ollama by default)
        print(f"  -> Connecting LLM backend '{cfg.llm.backend}' (model: {cfg.llm.model})...", flush=True)
        self.llm = make_llm(cfg.llm)
        self.tools.set_llm(self.llm)

        # 4. Kokoro TTS (GPU)
        print(f"  -> Loading Kokoro TTS on {cfg.tts.device}...", flush=True)
        self.tts = KokoroTTS(cfg.tts)

        # Warmup all models to avoid first-turn latency spikes
        print("  -> Warming up models (first CUDA kernel launches)...", flush=True)
        self.warmup()
        dt = time.perf_counter() - t0
        print(f"\033[32m[Engine]\033[0m All models warmed up and ready in {dt:.1f}s!\n", flush=True)

    def warmup(self) -> None:
        self.stt.warmup()
        try:
            self.llm.warmup()
        except Exception as e:
            logger.warning("LLM warmup failed (Ollama server might not be running yet): %s", e)
        self.tts.warmup()

    def create_endpointer(self) -> Endpointer:
        """Create a per-session endpointer wrapping the shared VAD model."""
        return Endpointer(self.vad_model, self.cfg.vad, self.cfg.audio)

    def create_conversation(self) -> Conversation:
        """Create a fresh conversation memory for a session."""
        prompt = self.cfg.llm.system_prompt + load_memory_prompt_suffix()
        return Conversation(prompt, self.cfg.llm.history_turns)
