"""Windows helper: make CTranslate2 (faster-whisper) find the CUDA/cuDNN DLLs.

PyTorch's Windows CUDA wheels already bundle cuBLAS + cuDNN in ``torch/lib``.
Registering that folder lets faster-whisper reuse them, so no separate CUDA
Toolkit / cuDNN install is required.
"""

from __future__ import annotations

import os
import site
import sys
from pathlib import Path

_done = False


def add_cuda_dll_dirs() -> None:
    global _done
    # Disable Windows symlinks requirement in HuggingFace cache
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    if _done or sys.platform != "win32":
        return
    _done = True

    candidates: list[Path] = []
    try:
        import torch  # noqa: F401  (importing loads torch's CUDA DLLs into the process)

        candidates.append(Path(torch.__file__).parent / "lib")
    except ImportError:
        pass
    for sp in site.getsitepackages():
        nv = Path(sp) / "nvidia"
        if nv.is_dir():
            candidates.extend(nv.glob("*/bin"))

    for d in candidates:
        if d.is_dir():
            os.add_dll_directory(str(d))
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
