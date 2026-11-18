"""Standalone Windows Desktop App launcher for Vāṇī (Edge WebView2 / Chromium App Mode + Global Hotkey)."""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import math
import os
import shutil
import struct
import subprocess
import threading
import time
import urllib.request
import zlib
from pathlib import Path

from .config import Config

MOD_ALT = 0x0001
MOD_NOREPEAT = 0x4000
VK_SPACE = 0x20
WM_HOTKEY = 0x0312
SW_RESTORE = 9
SW_SHOW = 5


def _is_server_ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/api/status", timeout=1.5) as resp:
            data = resp.read().decode("utf-8", errors="ignore")
            return '"ready"' in data
    except Exception:
        return False


def _kill_existing_server_on_port(port: int) -> None:
    """Terminate any stale process listening on the Vāṇī server port so fresh code always loads."""
    if os.name != "nt":
        return
    try:
        out = subprocess.check_output(
            ["netstat", "-ano", "-p", "TCP"],
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        my_pid = os.getpid()
        pids: set[int] = set()
        suffix = f":{port}"
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[1].endswith(suffix) and "LISTENING" in parts[3]:
                try:
                    pid = int(parts[4])
                    if pid > 0 and pid != my_pid:
                        pids.add(pid)
                except ValueError:
                    pass
        for pid in pids:
            print(f"[App] Stopping stale server process on port {port} (PID {pid})...", flush=True)
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True, check=False)
        if pids:
            time.sleep(0.6)
    except Exception:
        pass


def _ensure_vani_ico() -> Path:
    """Generate a native Windows .ico file (glowing cyan/violet Vāṇī orb) if not already present."""
    ico_path = Path(__file__).parent / "web" / "vani.ico"
    if ico_path.exists() and ico_path.stat().st_size > 1000:
        return ico_path

    w = h = 128
    raw_rows = bytearray()
    cx, cy = w / 2.0, h / 2.0
    for y in range(h):
        raw_rows.append(0)  # PNG filter type 0
        for x in range(w):
            dx, dy = x - cx, y - cy
            dist = math.hypot(dx, dy)
            # Rounded dark background
            if dist > 60:
                raw_rows.extend((0, 0, 0, 0))
                continue
            r, g, b, a = 9, 11, 18, 255
            # Outer ring glow
            if 44 <= dist <= 50:
                t = (x / w + y / h) * 0.5
                r = int(0 * (1 - t) + 124 * t)
                g = int(242 * (1 - t) + 58 * t)
                b = int(254 * (1 - t) + 237 * t)
            elif dist < 38:
                # Glowing core orb
                norm = dist / 38.0
                glow = max(0.0, 1.0 - norm * norm)
                t = norm
                cr = int(0 * (1 - t) + 124 * t)
                cg = int(242 * (1 - t) + 58 * t)
                cb = int(254 * (1 - t) + 237 * t)
                r = min(255, int(r * (1 - glow) + cr * glow + 80 * max(0.0, 1.0 - dist / 14.0)))
                g = min(255, int(g * (1 - glow) + cg * glow + 90 * max(0.0, 1.0 - dist / 14.0)))
                b = min(255, int(b * (1 - glow) + cb * glow + 90 * max(0.0, 1.0 - dist / 14.0)))
            raw_rows.extend((r, g, b, a))

    def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
    png_bytes = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(bytes(raw_rows), 9))
        + _png_chunk(b"IEND", b"")
    )
    # Wrap PNG in Windows ICO container
    ico_header = struct.pack("<HHH", 0, 1, 1)
    ico_entry = struct.pack("<BBBBHHII", w, h, 0, 0, 1, 32, len(png_bytes), 22)
    ico_path.parent.mkdir(parents=True, exist_ok=True)
    ico_path.write_bytes(ico_header + ico_entry + png_bytes)
    return ico_path


def install_desktop_shortcuts() -> list[Path]:
    """Create 'Vani Voice AI.lnk' shortcuts on the user's Windows Desktop and Start Menu."""
    if os.name != "nt":
        return []
    repo_root = Path(__file__).resolve().parents[2]
    bat_path = repo_root / "VaniApp.bat"
    ico_path = _ensure_vani_ico()

    # Create shortcuts in the project root, Desktop, OneDrive Desktop, and Start Menu
    candidate_dirs = [
        repo_root,
        Path(os.path.expandvars(r"%USERPROFILE%\OneDrive\Desktop")),
        Path(os.path.expandvars(r"%USERPROFILE%\Desktop")),
        Path(os.path.expandvars(r"%APPDATA%\Microsoft\Windows\Start Menu\Programs")),
    ]
    created: list[Path] = []
    for d in candidate_dirs:
        if d.exists():
            lnk = d / "Vani Voice AI.lnk"
            ps_script = (
                f"$ws = New-Object -ComObject WScript.Shell; "
                f"$s = $ws.CreateShortcut('{lnk}'); "
                f"$s.TargetPath = 'cmd.exe'; "
                f"$s.Arguments = '/c \"\"{bat_path}\"\"'; "
                f"$s.WorkingDirectory = '{repo_root}'; "
                f"$s.IconLocation = '{ico_path},0'; "
                f"$s.Description = 'Launch Vani Voice AI Desktop App'; "
                f"$s.WindowStyle = 7; "
                f"$s.Save()"
            )
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_script],
                capture_output=True,
                check=False,
            )
            if lnk.exists():
                created.append(lnk)
    return created


def _find_app_browser() -> str | None:
    """Locate Microsoft Edge, Chrome, or Brave for native --app window mode."""
    candidates = [
        shutil.which("msedge.exe"),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        shutil.which("chrome.exe"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        shutil.which("brave.exe"),
        r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\BraveSoftware\Brave-Browser\Application\brave.exe"),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


def _find_vani_window() -> int:
    """Find the native Vāṇī desktop app window handle by title."""
    if os.name != "nt":
        return 0
    user32 = ctypes.windll.user32
    user32.FindWindowExW.argtypes = [
        ctypes.wintypes.HWND,
        ctypes.wintypes.HWND,
        ctypes.wintypes.LPCWSTR,
        ctypes.wintypes.LPCWSTR,
    ]
    user32.FindWindowExW.restype = ctypes.wintypes.HWND
    user32.GetWindowTextW.argtypes = [ctypes.wintypes.HWND, ctypes.wintypes.LPWSTR, ctypes.c_int]
    user32.IsWindowVisible.argtypes = [ctypes.wintypes.HWND]

    curr = None
    while True:
        curr = user32.FindWindowExW(None, curr, "Chrome_WidgetWin_1", None)
        if not curr:
            break
        if not user32.IsWindowVisible(curr):
            continue
        buf = ctypes.create_unicode_buffer(512)
        user32.GetWindowTextW(curr, buf, 512)
        title = buf.value
        if "Vāṇī" in title or "Vani" in title or "127.0.0.1:8000" in title:
            return int(curr)
    return 0


def focus_vani_window() -> bool:
    """Bring the Vāṇī window to the foreground."""
    if os.name != "nt":
        return False
    user32 = ctypes.windll.user32
    hwnd = _find_vani_window()
    if hwnd:
        user32.ShowWindow(hwnd, SW_RESTORE)
        user32.ShowWindow(hwnd, SW_SHOW)
        user32.SetForegroundWindow(hwnd)
        return True
    return False


def _hotkey_listener_loop() -> None:
    """Register global Alt+Space hotkey on Windows to summon/focus the Vāṇī Desktop App."""
    if os.name != "nt":
        return
    user32 = ctypes.windll.user32
    hotkey_id = 1
    if not user32.RegisterHotKey(None, hotkey_id, MOD_ALT | MOD_NOREPEAT, VK_SPACE):
        return
    print("[App] Global hotkey registered: press Alt + Space anytime to summon Vāṇī.", flush=True)
    msg = ctypes.wintypes.MSG()
    try:
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) != 0:
            if msg.message == WM_HOTKEY and msg.wParam == hotkey_id:
                focus_vani_window()
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))
    finally:
        user32.UnregisterHotKey(None, hotkey_id)


def launch_app_window(url: str, mini: bool = False) -> subprocess.Popen | None:
    """Open the Vāṇī UI as a standalone chromeless desktop window."""
    _ensure_vani_ico()
    browser = _find_app_browser()
    target_url = f"{url}/?app=1" + ("&mini=1" if mini else "")

    if not browser:
        import webbrowser

        webbrowser.open(target_url)
        return None

    profile_dir = Path(os.path.expandvars(r"%LOCALAPPDATA%\VaniVoiceApp\profile"))
    profile_dir.mkdir(parents=True, exist_ok=True)
    win_size = "380,540" if mini else "1220,840"

    args = [
        browser,
        f"--app={target_url}",
        f"--user-data-dir={profile_dir}",
        f"--window-size={win_size}",
        "--autoplay-policy=no-user-gesture-required",
        "--disable-features=Translate,msEdgeSidebarV2",
        "--no-first-run",
        "--no-default-browser-check",
    ]
    print(f"[App] Launching Vāṇī Desktop Window ({Path(browser).name})...", flush=True)
    proc = subprocess.Popen(args)
    for _ in range(20):
        time.sleep(0.2)
        if focus_vani_window():
            break
    return proc


def run_desktop_app(cfg: Config, mini: bool = False) -> None:
    """Restart any stale server on our port, start the local GPU engine, and open the native Vāṇī desktop app."""
    install_desktop_shortcuts()
    _kill_existing_server_on_port(cfg.server.port)
    base_url = f"http://{cfg.server.host}:{cfg.server.port}"
    started_server = False

    if _is_server_ready(base_url):
        print(f"[App] Connected to existing Vāṇī engine at {base_url}", flush=True)
    else:
        from .server import run_server

        started_server = True
        server_thread = threading.Thread(target=run_server, args=(cfg,), name="vani-server", daemon=True)
        server_thread.start()
        print("[App] Waiting for Vāṇī GPU models to warm up...", flush=True)
        for _ in range(180):
            if _is_server_ready(base_url):
                break
            time.sleep(0.5)

    # Start global Alt+Space hotkey listener in background thread
    threading.Thread(target=_hotkey_listener_loop, name="vani-hotkey", daemon=True).start()

    t_launch = time.monotonic()
    proc = launch_app_window(base_url, mini=mini)
    if started_server:
        try:
            if proc is not None:
                proc.wait()
                # If Edge handed the window off to an existing profile process (< 5s exit),
                # keep the GPU server alive until closed/interrupted.
                if time.monotonic() - t_launch < 5.0:
                    while True:
                        time.sleep(1.0)
            else:
                while True:
                    time.sleep(1.0)
        except KeyboardInterrupt:
            pass
    elif proc is not None:
        try:
            proc.wait()
        except KeyboardInterrupt:
            pass
