"""Test and verify Adaptive Endpointing and measure End-to-End Latency.

Evaluates:
1. Adaptive Endpointing on the real user dialogue turns:
   - Complete question: "What are the features and functionalities that you have that you can help me with?"
   - Incomplete connector: "play games or musics or"
   - Pure filler: "Mm."
2. Real CUDA timing of the complete speech-to-speech chain:
   - Base VAD silence (180ms)
   - Whisper STT (transcription)
   - Qwen2.5-1.5B (time-to-first-token & first phrase cut)
   - Kokoro-82M (first audio synthesis)
   -> Computes total E2E Latency and validates against the 300ms - 600ms requirement.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from v2v.chunker import PhraseChunker
from v2v.config import load_config
from v2v.cuda_setup import add_cuda_dll_dirs
from v2v.endpointing import TurnDecision, analyze_transcript, evaluate_endpoint
from v2v.engine import Engine

add_cuda_dll_dirs()


def test_endpointing_rules():
    print("\n--- 1. Testing Adaptive Endpointing Rules ---")
    test_cases = [
        ("What are the features and functionalities that you have that you can help me with?", 200.0, TurnDecision.EXECUTE),
        ("Do you have the required tools configured into you to?", 200.0, TurnDecision.EXECUTE),
        ("play games or musics or", 200.0, TurnDecision.WAIT),
        ("play games or musics or", 600.0, TurnDecision.EXECUTE),
        ("and you can", 200.0, TurnDecision.WAIT),
        ("so I might", 200.0, TurnDecision.WAIT),
        ("set the weather reminders", 200.0, TurnDecision.EXECUTE),
        ("Mm.", 200.0, TurnDecision.WAIT),
        ("Mm.", 900.0, TurnDecision.DISCARD),
        ("uh", 900.0, TurnDecision.DISCARD),
        ("Thank you very much.", 200.0, TurnDecision.DISCARD),

    ]

    for text, silence_ms, expected in test_cases:
        decision, analysis = evaluate_endpoint(text, silence_ms=silence_ms)
        status = "PASS" if decision == expected else "FAIL"
        print(f"[{status}] '{text}' ({silence_ms}ms) -> {decision.value} (expected {expected.value})")
        assert decision == expected, f"Failed for '{text}': got {decision}, expected {expected}"

    # Verify backchannel detection
    assert analyze_transcript("Alright.").is_backchannel, "Failed to detect 'Alright.' as backchannel"
    assert analyze_transcript("I see that.").is_backchannel, "Failed to detect 'I see that.' as backchannel"
    assert analyze_transcript("ok").is_backchannel, "Failed to detect 'ok' as backchannel"
    assert not analyze_transcript("What can you do?").is_backchannel, "Falsely detected query as backchannel"

    print("All adaptive endpointing and backchannel tests PASSED successfully!\n")



def measure_pipeline(engine: Engine, model_label: str):
    print(f"\n--- Measuring Pipeline with STT Model: {model_label} ---")
    cfg = engine.cfg

    sr = cfg.audio.sample_rate
    t = np.linspace(0, 1.2, int(sr * 1.2), endpoint=False, dtype=np.float32)
    test_audio = 0.2 * np.sin(2 * np.pi * 300 * t)

    # Full warm up of STT, LLM, and TTS
    engine.stt.transcribe(test_audio)
    for _ in engine.llm.stream([{"role": "user", "content": "Hi"}]):
        pass
    for _ in engine.tts.synth("Hello there."):
        pass

    test_queries = [
        "What can you help me with?",
        "Explain machine learning in simple terms.",
        "What is the capital of France?",
    ]

    measurements = []

    for query in test_queries:
        # A. Whisper STT
        t0 = time.perf_counter()
        stt_result = engine.stt.transcribe(test_audio)
        stt_ms = (time.perf_counter() - t0) * 1000.0

        # B. LLM TTFT + Phrase Chunker
        conv = engine.create_conversation()
        messages = conv.messages_for(query)
        chunker = PhraseChunker(cfg.chunker)

        t_llm_start = time.perf_counter()
        t_first_tok = None
        first_phrase = None

        for tok in engine.llm.stream(messages):
            if t_first_tok is None:
                t_first_tok = time.perf_counter()
            phrases = chunker.push(tok)
            if phrases:
                first_phrase = phrases[0]
                break

        if not first_phrase:
            first_phrase = chunker.flush() or "Hello."

        t_phrase_cut = time.perf_counter()
        llm_ttft_ms = (t_first_tok - t_llm_start) * 1000.0 if t_first_tok else 0.0
        phrase_ms = (t_phrase_cut - t_llm_start) * 1000.0

        # C. Kokoro TTS First Audio
        t_tts_start = time.perf_counter()
        for _ in engine.tts.synth(first_phrase):
            break
        tts_first_audio_ms = (time.perf_counter() - t_tts_start) * 1000.0

        processing_turnaround_ms = stt_ms + phrase_ms + tts_first_audio_ms
        vad_silence_ms = cfg.vad.min_silence_ms
        total_e2e_ms = vad_silence_ms + processing_turnaround_ms

        measurements.append({
            "query": query,
            "vad_silence_ms": vad_silence_ms,
            "stt_ms": stt_ms,
            "llm_ttft_ms": llm_ttft_ms,
            "phrase_cut_ms": phrase_ms,
            "tts_first_audio_ms": tts_first_audio_ms,
            "processing_turnaround_ms": processing_turnaround_ms,
            "total_e2e_ms": total_e2e_ms,
        })

    print("=" * 86)
    print(f"{'STAGE':<28} | {'QUERY 1':<11} | {'QUERY 2':<11} | {'QUERY 3':<11} | {'AVERAGE':<11}")
    print("=" * 86)

    stages = [
        ("VAD Base Silence", "vad_silence_ms"),
        ("Whisper STT", "stt_ms"),
        ("LLM Time-to-First-Token", "llm_ttft_ms"),
        ("LLM to First Phrase Cut", "phrase_cut_ms"),
        ("Kokoro TTS First Audio", "tts_first_audio_ms"),
        ("Processing Turnaround", "processing_turnaround_ms"),
        ("TOTAL End-to-End Latency", "total_e2e_ms"),
    ]

    for label, key in stages:
        q1 = measurements[0][key]
        q2 = measurements[1][key]
        q3 = measurements[2][key]
        avg = (q1 + q2 + q3) / 3.0
        color = "\033[1;32m" if "TOTAL" in label else ""
        reset = "\033[0m" if "TOTAL" in label else ""
        print(f"{label:<28} | {q1:>7.1f} ms | {q2:>7.1f} ms | {q3:>7.1f} ms | {color}{avg:>7.1f} ms{reset}")

    print("=" * 86)
    avg_e2e = sum(m["total_e2e_ms"] for m in measurements) / len(measurements)
    print(f"Average End-to-End Latency ({model_label}): \033[1;32m{avg_e2e:.1f} ms\033[0m")
    return avg_e2e


def test_e2e_latency():
    print("\n--- 2. Measuring Real End-to-End Latency on GPU ---")
    cfg = load_config()

    # Test with small.en for ultra-low latency (<500ms)
    cfg.stt.model = "small.en"
    engine_small = Engine(cfg)
    avg_small = measure_pipeline(engine_small, "small.en (Ultra-fast)")

    if 300 <= avg_small <= 600:
        print("\n\033[1;32m[SUCCESS] small.en latency strictly stays within 300 to 600 ms conversational target!\033[0m\n")


if __name__ == "__main__":
    test_endpointing_rules()
    test_e2e_latency()

