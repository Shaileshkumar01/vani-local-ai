"""CLI entry point:  run.bat run v2v [--config config.yaml] [--list-devices]"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    ap = argparse.ArgumentParser(description="Real-time local voice-to-voice agent")
    ap.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    ap.add_argument("--cli", action="store_true", help="Run in terminal mode using sounddevice (headphones recommended)")
    ap.add_argument("--list-devices", action="store_true", help="Show audio devices and exit")
    args = ap.parse_args()

    from .cuda_setup import add_cuda_dll_dirs

    add_cuda_dll_dirs()
    if sys.platform == "win32":
        os.system("")  # enable ANSI colours in the Windows console

    if args.list_devices:
        from .audio_io import list_devices

        list_devices()
        return

    from .config import load_config

    cfg = load_config(args.config)

    if args.cli:
        from .pipeline import VoicePipeline

        VoicePipeline(cfg).run()
    else:
        from .server import run_server

        run_server(cfg)


if __name__ == "__main__":
    main()
