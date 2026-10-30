"""One-shot downloader to pre-cache all models locally for 100% offline usage."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from v2v.cuda_setup import add_cuda_dll_dirs

add_cuda_dll_dirs()

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def log_step(msg: str):
    print(f"\n===> {msg}", flush=True)


def main():
    print("=======================================================")
    print("      VANI VOICE AI - ONE-TIME MODEL DOWNLOADER        ")
    print("=======================================================")

    # 1. Ollama LLM
    log_step("1/4: Pulling Qwen2.5-1.5B-Instruct in Ollama (~1.0 GB)...")
    try:
        res = subprocess.run(["ollama", "pull", "qwen2.5:1.5b"], check=False)
        if res.returncode != 0:
            print("  Note: If ollama CLI wasn't found in this subshell, make sure Ollama is running and run: ollama pull qwen2.5:1.5b")
    except Exception as e:
        print(f"  Note: Could not run 'ollama pull': {e}")

    # 2. Silero VAD
    log_step("2/4: Caching Silero VAD v5 weights (~2 MB)...")
    try:
        from silero_vad import load_silero_vad

        load_silero_vad()
        print("  ✓ Silero VAD cached successfully.")
    except Exception as e:
        print(f"  ✗ Failed to load Silero VAD: {e}")

    # 3. faster-whisper distil-large-v3
    log_step("3/4: Downloading faster-whisper 'distil-large-v3' (~1.5 GB)...")
    try:
        from faster_whisper import WhisperModel

        # Loads on CPU just for downloading weights into Hugging Face cache
        WhisperModel("distil-large-v3", device="cpu", compute_type="int8")
        print("  ✓ faster-whisper distil-large-v3 cached successfully.")
    except Exception as e:
        print(f"  ✗ Failed to download Whisper: {e}")

    # 4. Kokoro-82M TTS
    log_step("4/4: Downloading Kokoro-82M weights & voices (~350 MB)...")
    try:
        from kokoro import KPipeline

        pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device="cpu")
        # Run a 2-word synthesis to download the default 'af_heart' voice tensor
        for _ in pipe("Hello", voice="af_heart"):
            pass
        print("  ✓ Kokoro-82M and default voice cached successfully.")
    except Exception as e:
        print(f"  ✗ Failed to load Kokoro: {e}")

    print("\n\033[1;32m✓ All models cached locally! The agent is now 100% offline capable.\033[0m\n")


if __name__ == "__main__":
    main()
