"""Benchmark: Sequential vs Concurrent Streaming Pipeline.

Runs identical prompts through:
  Mode A: Traditional Sequential Pipeline (Wait for full STT -> Wait for full LLM -> Wait for TTS)
  Mode B: Our Concurrent Streaming Pipeline (STT -> Stream LLM + Phrase Chunker -> Stream Kokoro TTS)

Outputs a comparative table and saves logs/benchmark.csv for the major project report.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from v2v.chunker import PhraseChunker
from v2v.config import load_config
from v2v.cuda_setup import add_cuda_dll_dirs
from v2v.engine import Engine

add_cuda_dll_dirs()

TEST_PROMPTS = [
    "What is the capital of France and why is it famous?",
    "Explain what latency means in voice agents in one sentence.",
    "Can you give me three quick tips to improve my study habits?",
    "Tell me a short joke.",
    "Describe the difference between an edge device and cloud computing.",
]


def run_sequential(engine: Engine, prompt: str) -> dict[str, float]:
    conv = engine.create_conversation()
    messages = conv.messages_for(prompt)

    # 1. Full LLM completion
    t0 = time.perf_counter()
    full_text = engine.llm.complete(messages)
    t_llm_done = time.perf_counter()
    llm_ms = (t_llm_done - t0) * 1000

    # 2. TTS synthesis of the full text
    t_tts_start = time.perf_counter()
    first_audio_ms = None
    for chunk in engine.tts.synth(full_text):
        if first_audio_ms is None:
            first_audio_ms = (time.perf_counter() - t_tts_start) * 1000
            break

    total_time_to_first_audio = (time.perf_counter() - t0) * 1000
    return {
        "llm_ms": round(llm_ms, 1),
        "tts_first_ms": round(first_audio_ms or 0, 1),
        "time_to_first_audio_ms": round(total_time_to_first_audio, 1),
    }


def run_streaming(engine: Engine, prompt: str) -> dict[str, float]:
    conv = engine.create_conversation()
    messages = conv.messages_for(prompt)
    chunker = PhraseChunker(engine.cfg.chunker)

    t0 = time.perf_counter()
    t_first_tok = None
    first_phrase = None

    # Stream tokens until first phrase is cut
    for tok in engine.llm.stream(messages):
        if t_first_tok is None:
            t_first_tok = time.perf_counter()
        phrases = chunker.push(tok)
        if phrases:
            first_phrase = phrases[0]
            break

    if not first_phrase:
        first_phrase = chunker.flush() or "Hello"

    t_first_phrase = time.perf_counter()

    # Synthesize first phrase
    for _ in engine.tts.synth(first_phrase):
        break

    t_first_audio = time.perf_counter()

    return {
        "llm_ttft_ms": round((t_first_tok - t0) * 1000, 1) if t_first_tok else 0,
        "first_phrase_ms": round((t_first_phrase - t0) * 1000, 1),
        "time_to_first_audio_ms": round((t_first_audio - t0) * 1000, 1),
    }


def main():
    print("\n=======================================================")
    print("   VĀṆĪ BENCHMARK: SEQUENTIAL VS STREAMING PIPELINE    ")
    print("=======================================================\n")

    cfg = load_config()
    print("Initializing engine (loading models into VRAM)...")
    engine = Engine(cfg)

    results = []
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    csv_file = log_dir / "benchmark.csv"

    print("\nRunning comparative test turns...\n")
    print(f"{'PROMPT':<38} | {'SEQUENTIAL':<12} | {'STREAMING':<12} | {'SPEEDUP':<8}")
    print("-" * 78)

    for prompt in TEST_PROMPTS:
        seq = run_sequential(engine, prompt)
        strm = run_streaming(engine, prompt)

        t_seq = seq["time_to_first_audio_ms"]
        t_strm = strm["time_to_first_audio_ms"]
        speedup = t_seq / max(1.0, t_strm)

        short_p = (prompt[:35] + "...") if len(prompt) > 35 else prompt
        print(f"{short_p:<38} | {t_seq:>7.1f} ms    | {t_strm:>7.1f} ms    | \033[1;32m{speedup:>5.2f}x\033[0m")

        results.append({
            "prompt": prompt,
            "seq_total_ms": t_seq,
            "stream_total_ms": t_strm,
            "speedup": round(speedup, 2),
            "stream_ttft_ms": strm["llm_ttft_ms"],
        })

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["prompt", "seq_total_ms", "stream_total_ms", "speedup", "stream_ttft_ms"]
        )
        writer.writeheader()
        writer.writerows(results)

    avg_speedup = sum(r["speedup"] for r in results) / len(results)
    avg_stream = sum(r["stream_total_ms"] for r in results) / len(results)
    avg_seq = sum(r["seq_total_ms"] for r in results) / len(results)

    print("-" * 78)
    print(f"{'AVERAGE':<38} | {avg_seq:>7.1f} ms    | {avg_stream:>7.1f} ms    | \033[1;32m{avg_speedup:>5.2f}x\033[0m")
    print(f"\nAverage streaming response time: \033[1;32m{avg_stream:.1f} ms\033[0m (vs {avg_seq:.1f} ms sequential)")
    print(f"Results saved to: \033[36m{csv_file.resolve()}\033[0m\n")


if __name__ == "__main__":
    main()
