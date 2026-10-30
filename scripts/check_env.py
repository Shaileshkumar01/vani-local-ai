"""Pre-flight sanity check for GPU, CUDA, PyTorch, CTranslate2, Ollama, and Audio."""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to sys.path so scripts can import v2v modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from v2v.cuda_setup import add_cuda_dll_dirs

add_cuda_dll_dirs()


if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def check_mark(ok: bool) -> str:
    return "[PASS]" if ok else "[FAIL]"


def main():
    print("\n=======================================================")
    print("      VANI VOICE AI - SYSTEM ENVIRONMENT DIAGNOSTIC    ")
    print("=======================================================\n")

    # 1. Python version
    py_ok = sys.version_info >= (3, 11) and sys.version_info < (3, 13)
    print(f"{check_mark(py_ok)} Python version: {sys.version.split()[0]} (expected 3.11 or 3.12)")

    # 2. PyTorch & CUDA
    cuda_ok = False
    try:
        import torch

        cuda_ok = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "None detected"
        vram = (
            f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB"
            if cuda_ok
            else "N/A"
        )
        print(f"{check_mark(cuda_ok)} PyTorch CUDA: {torch.__version__} | GPU: {gpu_name} ({vram})")
    except Exception as e:
        print(f"{check_mark(False)} PyTorch CUDA check failed: {e}")

    # 3. faster-whisper & CTranslate2
    whisper_ok = False
    try:
        import ctranslate2

        supported_types = ctranslate2.get_supported_compute_types("cuda") if cuda_ok else []
        whisper_ok = "int8_float16" in supported_types or "float16" in supported_types
        print(
            f"{check_mark(whisper_ok)} CTranslate2 GPU compute types: {supported_types} (int8_float16 supported: {'int8_float16' in supported_types})"
        )
    except Exception as e:
        print(f"{check_mark(False)} faster-whisper / CTranslate2 check failed: {e}")

    # 4. Ollama LLM Server
    ollama_ok = False
    try:
        import httpx

        r = httpx.get("http://127.0.0.1:11434/api/tags", timeout=2.0)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            has_qwen = any("qwen2.5:1.5b" in m for m in models)
            ollama_ok = True
            qwen_mark = check_mark(has_qwen)
            print(f"{check_mark(True)} Ollama server running on http://127.0.0.1:11434")
            print(f"{qwen_mark}   Model 'qwen2.5:1.5b' installed: {has_qwen} (installed: {models})")
            if not has_qwen:
                print("       -> Run: ollama pull qwen2.5:1.5b")
        else:
            print(f"{check_mark(False)} Ollama responded with status {r.status_code}")
    except Exception:
        print(f"{check_mark(False)} Ollama server not reachable at http://127.0.0.1:11434")
        print("       -> Start Ollama from your Start menu or run: ollama serve")

    # 5. Kokoro TTS & Misaki
    kokoro_ok = False
    try:
        import kokoro
        import misaki

        kokoro_ok = True
        print(f"{check_mark(True)} Kokoro TTS (v{kokoro.__version__}) and Misaki G2P ready")
    except Exception as e:
        print(f"{check_mark(False)} Kokoro / Misaki import failed: {e}")

    # 6. Silero VAD
    vad_ok = False
    try:
        from silero_vad import load_silero_vad

        load_silero_vad()
        vad_ok = True
        print(f"{check_mark(True)} Silero VAD v5 ready (CPU)")
    except Exception as e:
        print(f"{check_mark(False)} Silero VAD check failed: {e}")

    # 7. Audio devices
    audio_ok = False
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        audio_ok = len(devices) > 0
        def_in, def_out = sd.default.device
        print(f"{check_mark(audio_ok)} Local Audio: {len(devices)} devices found (Default In: {def_in}, Out: {def_out})")
    except Exception as e:
        print(f"{check_mark(False)} sounddevice audio check failed: {e}")

    print("\n-------------------------------------------------------")
    all_pass = py_ok and cuda_ok and whisper_ok and ollama_ok and kokoro_ok and vad_ok
    if all_pass:
        print("\033[1;32mALL SYSTEMS OPERATIONAL! You are ready to start the voice agent.\033[0m")
        print("Run: .\\run.bat run v2v\n")
    else:
        print("\033[1;33mSOME COMPONENTS NEED ATTENTION (see details above).\033[0m\n")


if __name__ == "__main__":
    main()
