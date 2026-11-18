"""Screen Share Vision (Shared Frame VLM + WinRT OCR) & Accurate Window/Task Inspector for Vāṇī.

Separates:
1. Visible On-Screen Windows (Z-ordered EnumWindows: Active Foreground, Visible, Minimized)
2. Background / Tray Processes (CreateToolhelp32Snapshot excluding windows already visible)
3. Live Shared Screen Analysis (from browser getDisplayMedia JPEG frame or Win32 GDI fallback)
   using local Ollama Vision-Language Models (if available) + Windows WinRT OCR.
"""

from __future__ import annotations

import base64
import ctypes
import ctypes.wintypes
import json
import logging
import os
import re
import struct
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from .base import BaseTool, ToolResult

logger = logging.getLogger("v2v.tools.screen_tasks")

_FRIENDLY_APP_NAMES: dict[str, str] = {
    "code.exe": "VS Code",
    "antigravity.exe": "Antigravity IDE",
    "antigravity ide.exe": "Antigravity IDE",
    "cursor.exe": "Cursor IDE",
    "msedge.exe": "Microsoft Edge",
    "chrome.exe": "Google Chrome",
    "brave.exe": "Brave Browser",
    "firefox.exe": "Firefox",
    "spotify.exe": "Spotify",
    "explorer.exe": "File Explorer",
    "windowsterminal.exe": "Windows Terminal",
    "powershell.exe": "PowerShell",
    "cmd.exe": "Command Prompt",
    "discord.exe": "Discord",
    "ms-teams.exe": "Microsoft Teams",
    "teams.exe": "Microsoft Teams",
    "msteams.exe": "Microsoft Teams",
    "zoom.exe": "Zoom",
    "whatsapp.exe": "WhatsApp",
    "whatsapp.root.exe": "WhatsApp",
    "telegram.exe": "Telegram",
    "slack.exe": "Slack",
    "notion.exe": "Notion",
    "notepad.exe": "Notepad",
    "taskmgr.exe": "Task Manager",
    "python.exe": "Python",
    "vlc.exe": "VLC Media Player",
    "obs64.exe": "OBS Studio",
    "steam.exe": "Steam",
    "acrobat.exe": "Adobe Acrobat",
    "winword.exe": "Microsoft Word",
    "excel.exe": "Microsoft Excel",
    "powerpnt.exe": "PowerPoint",
    "applicationframehost.exe": "Windows App",
}

_IGNORED_BG_EXES = {
    "system idle process",
    "system",
    "registry",
    "smss.exe",
    "csrss.exe",
    "wininit.exe",
    "services.exe",
    "lsass.exe",
    "svchost.exe",
    "fontdrvhost.exe",
    "dwm.exe",
    "memory compression",
    "conhost.exe",
    "runtimebroker.exe",
    "searchhost.exe",
    "startmenuexperiencehost.exe",
    "sihost.exe",
    "taskhostw.exe",
    "ctfmon.exe",
    "dllhost.exe",
    "audiodg.exe",
    "msedgewebview2.exe",
    "applicationframehost.exe",
    "shellexperiencehost.exe",
    "textinputhost.exe",
    "widgetservice.exe",
    "widgets.exe",
    "smartscreen.exe",
    "securityhealthservice.exe",
    "nissrv.exe",
    "msmpeng.exe",
    "spoolsv.exe",
    "wlanext.exe",
    "dashost.exe",
    "aggregatorhost.exe",
    "wudfhost.exe",
    "backgroundtaskhost.exe",
    "lockapp.exe",
    "searchindexer.exe",
    "codex-windows-sandbox-service.exe",
}

_IGNORED_WINDOW_TITLES = {
    "program manager",
    "windows input experience",
    "microsoft text input application",
    "nvidia geforce overlay",
    "default ime",
    "msctfime ui",
    "popuphost",
}

_VANI_UI_BOILERPLATE = (
    "LIVE PIPELINE LATENCY",
    "TARGET: < 300ms PROCESSING",
    "VAD Endpoint",
    "Distil-Large-v3",
    "1st Phrase Cut",
    "Chunker Engine",
    "Processing Turnaround",
    "CONVERSATION TRANSCRIPT",
    "Barge-in Active",
    "Major Project: Real-Time Edge Voice-to-Voice AI",
    "Zero Cloud",
    "Sub-300ms Local Voice Agent",
    "Start Voice Agent",
    "Stop Session",
    "Share Screen",
    "You're using an unsupported command-line flag",
    "unsupported command-line flag",
    "--use-fake-ui-for-media-stream",
    "-use-fake-ui-for-media-stream",
    "This poses stability and security risks.",
    "This poses stability",
    "Noise Cancel:",
)

_LAST_CPU_TIMES: tuple[int, int, int] | None = None

_VISION_MODEL_PREFIXES = (
    "moondream",
    "qwen2.5vl",
    "qwen2.5-vl",
    "qwen3-vl",
    "llama3.2-vision",
    "minicpm-v",
    "llava",
    "bakllava",
)


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.wintypes.DWORD),
        ("dwMemoryLoad", ctypes.wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_uint64),
        ("ullAvailPhys", ctypes.c_uint64),
        ("ullTotalPageFile", ctypes.c_uint64),
        ("ullAvailPageFile", ctypes.c_uint64),
        ("ullTotalVirtual", ctypes.c_uint64),
        ("ullAvailVirtual", ctypes.c_uint64),
        ("ullAvailExtendedVirtual", ctypes.c_uint64),
    ]


class _SYSTEM_POWER_STATUS(ctypes.Structure):
    _fields_ = [
        ("ACLineStatus", ctypes.c_byte),
        ("BatteryFlag", ctypes.c_byte),
        ("BatteryLifePercent", ctypes.c_byte),
        ("SystemStatusFlag", ctypes.c_byte),
        ("BatteryLifeTime", ctypes.wintypes.DWORD),
        ("BatteryFullLifeTime", ctypes.wintypes.DWORD),
    ]


class _FILETIME(ctypes.Structure):
    _fields_ = [
        ("dwLowDateTime", ctypes.wintypes.DWORD),
        ("dwHighDateTime", ctypes.wintypes.DWORD),
    ]


class _RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.wintypes.LONG),
        ("top", ctypes.wintypes.LONG),
        ("right", ctypes.wintypes.LONG),
        ("bottom", ctypes.wintypes.LONG),
    ]


class _PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", ctypes.wintypes.DWORD),
        ("cntUsage", ctypes.wintypes.DWORD),
        ("th32ProcessID", ctypes.wintypes.DWORD),
        ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
        ("th32ModuleID", ctypes.wintypes.DWORD),
        ("cntThreads", ctypes.wintypes.DWORD),
        ("th32ParentProcessID", ctypes.wintypes.DWORD),
        ("pcPriClassBase", ctypes.c_long),
        ("dwFlags", ctypes.wintypes.DWORD),
        ("szExeFile", ctypes.c_wchar * 260),
    ]


class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.wintypes.DWORD),
        ("PageFaultCount", ctypes.wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", ctypes.wintypes.DWORD),
        ("biWidth", ctypes.wintypes.LONG),
        ("biHeight", ctypes.wintypes.LONG),
        ("biPlanes", ctypes.wintypes.WORD),
        ("biBitCount", ctypes.wintypes.WORD),
        ("biCompression", ctypes.wintypes.DWORD),
        ("biSizeImage", ctypes.wintypes.DWORD),
        ("biXPelsPerMeter", ctypes.wintypes.LONG),
        ("biYPelsPerMeter", ctypes.wintypes.LONG),
        ("biClrUsed", ctypes.wintypes.DWORD),
        ("biClrImportant", ctypes.wintypes.DWORD),
    ]


def _friendly_app_name(exe_name: str) -> str:
    low = exe_name.lower().strip()
    if not low.endswith(".exe") and f"{low}.exe" in _FRIENDLY_APP_NAMES:
        return _FRIENDLY_APP_NAMES[f"{low}.exe"]
    if low in _FRIENDLY_APP_NAMES:
        return _FRIENDLY_APP_NAMES[low]
    stem = Path(exe_name).stem
    return stem.replace("_", " ").replace("-", " ").title()


def _get_process_exe_name(pid: int) -> str:
    if os.name != "nt" or pid <= 0:
        return ""
    kernel32 = ctypes.windll.kernel32
    kernel32.OpenProcess.restype = ctypes.wintypes.HANDLE
    kernel32.OpenProcess.argtypes = [ctypes.wintypes.DWORD, ctypes.wintypes.BOOL, ctypes.wintypes.DWORD]
    kernel32.CloseHandle.argtypes = [ctypes.wintypes.HANDLE]
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    h_proc = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h_proc:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(512)
        size = ctypes.wintypes.DWORD(512)
        if kernel32.QueryFullProcessImageNameW(h_proc, 0, buf, ctypes.byref(size)):
            return Path(buf.value).name
    finally:
        kernel32.CloseHandle(h_proc)
    return ""


def get_open_windows() -> list[dict[str, Any]]:
    """Enumerate visible and minimized desktop windows in Z-order (never guesses from OCR text)."""
    if os.name != "nt":
        return []

    user32 = ctypes.windll.user32
    dwmapi = ctypes.windll.dwmapi
    GWL_EXSTYLE = -20
    WS_EX_TOOLWINDOW = 0x00000080
    DWMWA_CLOAKED = 14

    WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
    user32.EnumWindows.argtypes = [WNDENUMPROC, ctypes.wintypes.LPARAM]
    user32.IsWindowVisible.argtypes = [ctypes.wintypes.HWND]
    user32.IsIconic.argtypes = [ctypes.wintypes.HWND]
    user32.GetWindowTextLengthW.argtypes = [ctypes.wintypes.HWND]
    user32.GetWindowTextW.argtypes = [ctypes.wintypes.HWND, ctypes.wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowLongW.argtypes = [ctypes.wintypes.HWND, ctypes.c_int]
    user32.GetWindowRect.argtypes = [ctypes.wintypes.HWND, ctypes.POINTER(_RECT)]
    user32.GetWindowThreadProcessId.argtypes = [ctypes.wintypes.HWND, ctypes.POINTER(ctypes.wintypes.DWORD)]
    user32.GetForegroundWindow.restype = ctypes.wintypes.HWND
    dwmapi.DwmGetWindowAttribute.argtypes = [
        ctypes.wintypes.HWND,
        ctypes.wintypes.DWORD,
        ctypes.c_void_p,
        ctypes.wintypes.DWORD,
    ]

    fg_hwnd = user32.GetForegroundWindow()
    results: list[dict[str, Any]] = []

    def _enum_cb(hwnd: int, _lparam: int) -> bool:
        try:
            if not user32.IsWindowVisible(hwnd):
                return True
            length = user32.GetWindowTextLengthW(hwnd)
            if length <= 0:
                return True

            ex_style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            if ex_style & WS_EX_TOOLWINDOW:
                return True

            cloaked = ctypes.c_int(0)
            dwmapi.DwmGetWindowAttribute(
                ctypes.wintypes.HWND(hwnd),
                DWMWA_CLOAKED,
                ctypes.byref(cloaked),
                ctypes.sizeof(cloaked),
            )
            if cloaked.value != 0:
                return True

            is_minimized = bool(user32.IsIconic(hwnd))
            if not is_minimized:
                rect = _RECT()
                if user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                    if (rect.right - rect.left) < 120 or (rect.bottom - rect.top) < 80:
                        return True

            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            title = buf.value.strip()
            if not title or title.lower() in _IGNORED_WINDOW_TITLES:
                return True

            pid = ctypes.wintypes.DWORD(0)
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            exe = _get_process_exe_name(pid.value)
            app_name = _friendly_app_name(exe) if exe else title.split(" - ")[-1].strip()
            if app_name == "Windows App":
                app_name = title.split(" - ")[0].strip()

            is_vani = "Vāṇī" in title or "Vani —" in title or "127.0.0.1:8000" in title
            results.append(
                {
                    "hwnd": int(hwnd),
                    "title": title,
                    "exe": exe.lower(),
                    "app": "Vāṇī Voice AI" if is_vani else app_name,
                    "is_active": bool(fg_hwnd and int(hwnd) == int(fg_hwnd)),
                    "is_minimized": is_minimized,
                    "is_vani": is_vani,
                }
            )
        except Exception:
            pass
        return True

    user32.EnumWindows(WNDENUMPROC(_enum_cb), 0)

    if not results:
        try:
            proc = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "Get-Process | Where-Object { $_.MainWindowTitle } | Select-Object ProcessName, MainWindowTitle | ConvertTo-Json -Compress",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                timeout=2.0,
                check=False,
            )
            out = (proc.stdout or "").strip()
            if out:
                items = json.loads(out)
                if isinstance(items, dict):
                    items = [items]
                for it in items:
                    title = str(it.get("MainWindowTitle") or "").strip()
                    pname = str(it.get("ProcessName") or "").strip()
                    if not title or title.lower() in _IGNORED_WINDOW_TITLES:
                        continue
                    is_vani = "Vāṇī" in title or "Vani —" in title or "127.0.0.1:8000" in title
                    app_name = _friendly_app_name(pname)
                    if app_name == "Windows App":
                        app_name = title
                    results.append(
                        {
                            "hwnd": 0,
                            "title": title,
                            "exe": f"{pname.lower()}.exe",
                            "app": "Vāṇī Voice AI" if is_vani else app_name,
                            "is_active": False,
                            "is_minimized": False,
                            "is_vani": is_vani,
                        }
                    )
        except Exception:
            pass

    return results


def get_system_and_tasks_summary(visible_apps: set[str] | None = None) -> dict[str, Any]:
    """Read live CPU %, RAM usage, Battery %, top memory tasks, and background apps (excluding visible windows)."""
    info: dict[str, Any] = {
        "cpu_percent": 0,
        "ram_percent": 0,
        "ram_used_gb": 0.0,
        "ram_total_gb": 0.0,
        "battery_percent": None,
        "is_charging": None,
        "top_tasks": [],
        "background_apps": [],
    }
    if os.name != "nt":
        return info

    visible_lower = {a.lower() for a in (visible_apps or set())}
    kernel32 = ctypes.windll.kernel32
    kernel32.OpenProcess.restype = ctypes.wintypes.HANDLE
    kernel32.OpenProcess.argtypes = [ctypes.wintypes.DWORD, ctypes.wintypes.BOOL, ctypes.wintypes.DWORD]
    kernel32.CloseHandle.argtypes = [ctypes.wintypes.HANDLE]

    mem = _MEMORYSTATUSEX()
    mem.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
    if kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)):
        info["ram_percent"] = int(mem.dwMemoryLoad)
        info["ram_total_gb"] = round(mem.ullTotalPhys / (1024**3), 1)
        info["ram_used_gb"] = round((mem.ullTotalPhys - mem.ullAvailPhys) / (1024**3), 1)

    def _read_cpu_times() -> tuple[int, int, int]:
        idle, kernel, user = _FILETIME(), _FILETIME(), _FILETIME()
        kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user))
        to_u64 = lambda ft: (ft.dwHighDateTime << 32) | ft.dwLowDateTime
        return to_u64(idle), to_u64(kernel), to_u64(user)

    global _LAST_CPU_TIMES
    try:
        curr = _read_cpu_times()
        if _LAST_CPU_TIMES is None:
            _LAST_CPU_TIMES = curr
        else:
            i1, k1, u1 = _LAST_CPU_TIMES
            i2, k2, u2 = curr
            _LAST_CPU_TIMES = curr
            idle_diff = i2 - i1
            total_diff = (k2 - k1) + (u2 - u1)
            if total_diff > 0:
                info["cpu_percent"] = max(0, min(100, round((1.0 - idle_diff / total_diff) * 100)))
    except Exception:
        pass

    pwr = _SYSTEM_POWER_STATUS()
    if kernel32.GetSystemPowerStatus(ctypes.byref(pwr)):
        if 0 <= pwr.BatteryLifePercent <= 100:
            info["battery_percent"] = int(pwr.BatteryLifePercent)
            info["is_charging"] = bool(pwr.ACLineStatus == 1)

    TH32CS_SNAPPROCESS = 0x00000002
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    psapi = ctypes.windll.psapi

    snap = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    app_mem_bytes: dict[str, int] = {}

    if snap and snap != -1:
        try:
            pe = _PROCESSENTRY32W()
            pe.dwSize = ctypes.sizeof(_PROCESSENTRY32W)
            has_proc = kernel32.Process32FirstW(snap, ctypes.byref(pe))
            while has_proc:
                exe = pe.szExeFile.strip()
                exe_low = exe.lower()
                if exe and exe_low not in _IGNORED_BG_EXES and pe.th32ProcessID > 4:
                    label = _friendly_app_name(exe)
                    h_p = kernel32.OpenProcess(
                        PROCESS_QUERY_LIMITED_INFORMATION,
                        False,
                        pe.th32ProcessID,
                    )
                    if h_p:
                        try:
                            pmc = _PROCESS_MEMORY_COUNTERS()
                            pmc.cb = ctypes.sizeof(_PROCESS_MEMORY_COUNTERS)
                            if psapi.GetProcessMemoryInfo(h_p, ctypes.byref(pmc), pmc.cb):
                                app_mem_bytes[label] = app_mem_bytes.get(label, 0) + int(pmc.WorkingSetSize)
                        finally:
                            kernel32.CloseHandle(h_p)
                has_proc = kernel32.Process32NextW(snap, ctypes.byref(pe))
        finally:
            kernel32.CloseHandle(snap)

    sorted_apps = sorted(app_mem_bytes.items(), key=lambda kv: kv[1], reverse=True)
    info["top_tasks"] = [
        {"name": name, "ram_mb": round(b / (1024**2))}
        for name, b in sorted_apps[:6]
        if b >= 20 * 1024 * 1024
    ]
    # Any non-system app using >= 30 MB that does not already have a visible window is a background app
    info["background_apps"] = [
        name for name, b in sorted_apps if b >= 30 * 1024 * 1024 and name.lower() not in visible_lower
    ][:8]
    return info


def capture_screen_rgb() -> np.ndarray | None:
    """Capture the primary desktop screen as a (H, W, 3) uint8 RGB numpy array via Win32 GDI."""
    if os.name != "nt":
        return None
    try:
        user32 = ctypes.windll.user32
        gdi32 = ctypes.windll.gdi32
        try:
            user32.SetProcessDPIAware()
        except Exception:
            pass

        w = user32.GetSystemMetrics(0)
        h = user32.GetSystemMetrics(1)
        if w <= 0 or h <= 0:
            return None

        hdc_s = user32.GetDC(0)
        hdc_m = gdi32.CreateCompatibleDC(hdc_s)
        hbmp = gdi32.CreateCompatibleBitmap(hdc_s, w, h)
        old_obj = gdi32.SelectObject(hdc_m, hbmp)
        gdi32.BitBlt(hdc_m, 0, 0, w, h, hdc_s, 0, 0, 0x00CC0020)
        gdi32.SelectObject(hdc_m, old_obj)

        bmi = _BITMAPINFOHEADER()
        bmi.biSize = ctypes.sizeof(_BITMAPINFOHEADER)
        bmi.biWidth = w
        bmi.biHeight = -h
        bmi.biPlanes = 1
        bmi.biBitCount = 32
        bmi.biCompression = 0
        buf_size = w * h * 4
        bmi.biSizeImage = buf_size
        buf = ctypes.create_string_buffer(buf_size)
        gdi32.GetDIBits(hdc_m, hbmp, 0, h, buf, ctypes.byref(bmi), 0)

        gdi32.DeleteObject(hbmp)
        gdi32.DeleteDC(hdc_m)
        user32.ReleaseDC(0, hdc_s)

        bgra = np.frombuffer(buf.raw, dtype=np.uint8).reshape((h, w, 4))
        rgb = bgra[:, :, [2, 1, 0]].copy()
        if float(rgb.mean()) < 0.5:
            return None
        return rgb
    except Exception as e:
        logger.warning("capture_screen_rgb failed: %s", e)
        return None


def _save_rgb_as_bmp(rgb: np.ndarray, path: Path) -> None:
    h, w, _ = rgb.shape
    bgr_bottom_up = rgb[::-1, :, ::-1]
    row_pad = (4 - (w * 3) % 4) % 4
    if row_pad:
        pad = np.zeros((h, row_pad), dtype=np.uint8)
        raw = np.hstack([bgr_bottom_up.reshape(h, w * 3), pad]).tobytes()
    else:
        raw = bgr_bottom_up.tobytes()

    bmi = _BITMAPINFOHEADER()
    bmi.biSize = ctypes.sizeof(_BITMAPINFOHEADER)
    bmi.biWidth = w
    bmi.biHeight = h
    bmi.biPlanes = 1
    bmi.biBitCount = 24
    bmi.biSizeImage = len(raw)
    bf_hdr = struct.pack("<2sIHHI", b"BM", 14 + 40 + len(raw), 0, 0, 54)
    path.write_bytes(bf_hdr + bytes(bmi) + raw)


def _run_winrt_ocr_on_file(img_path: Path, invert_dark: bool = True) -> str:
    """Run Windows WinRT OcrEngine on a BMP/JPEG/PNG file, with optional dark-mode inversion."""
    if os.name != "nt" or not img_path.exists():
        return ""

    ps_script = f"""
$ErrorActionPreference = 'SilentlyContinue'
Add-Type -AssemblyName System.Runtime.WindowsRuntime
Add-Type -AssemblyName System.Drawing
$null = [Windows.Storage.Streams.IRandomAccessStream, Windows.Storage.Streams, ContentType=WindowsRuntime]
$null = [Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics.Imaging, ContentType=WindowsRuntime]
$null = [Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType=WindowsRuntime]
$null = [Windows.Media.Ocr.OcrResult, Windows.Foundation, ContentType=WindowsRuntime]

$srcPath = '{img_path}'
$ocrPath = $srcPath

if ('{str(invert_dark).lower()}' -eq 'true' -and $srcPath.EndsWith('.jpg')) {{
    $bmp = [System.Drawing.Bitmap]::FromFile($srcPath)
    # Sample average luminance to detect dark-mode UIs
    $lumSum = 0.0; $cnt = 0
    for ($sy = 0; $sy -lt $bmp.Height; $sy += 24) {{
        for ($sx = 0; $sx -lt $bmp.Width; $sx += 24) {{
            $c = $bmp.GetPixel($sx, $sy)
            $lumSum += (0.299 * $c.R + 0.587 * $c.G + 0.114 * $c.B)
            $cnt++
        }}
    }}
    $isDark = ($cnt -gt 0 -and ($lumSum / $cnt) -lt 135)
    $w = [int]($bmp.Width * 2)
    $h = [int]($bmp.Height * 2)
    $up = New-Object System.Drawing.Bitmap($w, $h)
    $g = [System.Drawing.Graphics]::FromImage($up)
    $g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
    if ($isDark) {{
        $ia = New-Object System.Drawing.Imaging.ImageAttributes
        $cm = New-Object System.Drawing.Imaging.ColorMatrix
        $cm.Matrix00 = -1; $cm.Matrix11 = -1; $cm.Matrix22 = -1
        $cm.Matrix40 = 1; $cm.Matrix41 = 1; $cm.Matrix42 = 1
        $ia.SetColorMatrix($cm)
        $rect = New-Object System.Drawing.Rectangle(0, 0, $w, $h)
        $g.DrawImage($bmp, $rect, 0, 0, $bmp.Width, $bmp.Height, [System.Drawing.GraphicsUnit]::Pixel, $ia)
    }} else {{
        $g.DrawImage($bmp, 0, 0, $w, $h)
    }}
    $g.Dispose()
    $bmp.Dispose()
    $tmpOut = [System.IO.Path]::ChangeExtension($srcPath, '.ocr.png')
    $up.Save($tmpOut, [System.Drawing.Imaging.ImageFormat]::Png)
    $up.Dispose()
    $ocrPath = $tmpOut
}}

$asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {{
    $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'
}})[0]

function Await($WinRtTask, $ResultType) {{
    $asTask = $asTaskGeneric.MakeGenericMethod($ResultType)
    $netTask = $asTask.Invoke($null, @($WinRtTask))
    $netTask.Wait(-1) | Out-Null
    return $netTask.Result
}}

$fs = [System.IO.File]::OpenRead($ocrPath)
$stream = [System.IO.WindowsRuntimeStreamExtensions]::AsRandomAccessStream($fs)
$decoder = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
$bitmap = Await ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
if ($null -eq $engine) {{
    $lang = New-Object Windows.Globalization.Language('en-US')
    $engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage($lang)
}}
$ocrResult = Await ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
$stream.Dispose()
$fs.Dispose()
Write-Output $ocrResult.Text
"""
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=4.5,
            check=False,
        )
        raw = (proc.stdout or "").strip()
        for phrase in _VANI_UI_BOILERPLATE:
            raw = raw.replace(phrase, "")
        return re.sub(r"\s+", " ", raw).strip()[:1600]
    except Exception as e:
        logger.warning("Windows WinRT OCR failed: %s", e)
        return ""


def _detect_ollama_vision_model() -> str | None:
    """Return configured or auto-detected local Ollama vision model if Ollama is running."""
    env_model = os.environ.get("VANI_VISION_MODEL", "").strip()
    if env_model:
        return env_model
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=0.6) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
        for m in data.get("models", []):
            name = str(m.get("name", "")).lower()
            if any(name.startswith(pref) for pref in _VISION_MODEL_PREFIXES):
                return str(m.get("name"))
    except Exception:
        pass
    return None


def _analyze_jpeg_with_ollama_vlm(b64_jpeg: str, user_question: str) -> str:
    """Query a local Ollama VLM (moondream / qwen2.5vl / llava) on the shared screen JPEG."""
    model = _detect_ollama_vision_model()
    if not model or not b64_jpeg:
        return ""
    try:
        payload = json.dumps(
            {
                "model": model,
                "prompt": (
                    f"You are looking at the user's live shared computer screen. "
                    f"User question: {user_question}\n"
                    f"Describe clearly and concisely what apps, visuals, text, or errors are visible on the screen."
                ),
                "images": [b64_jpeg],
                "stream": False,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            "http://127.0.0.1:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=8.0) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
            return str(data.get("response") or "").strip()
    except Exception as e:
        logger.debug("Ollama VLM analysis skipped/failed: %s", e)
        return ""


def _encode_rgb_to_bmp_and_ocr(rgb: np.ndarray) -> str:
    """Upscale and dark-mode invert an RGB numpy array, then run WinRT OCR."""
    gray = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]).astype(np.uint8)
    mean_lum = float(gray[::8, ::8].mean())
    proc_gray = (255 - gray) if mean_lum < 140 else gray
    up = np.repeat(np.repeat(proc_gray, 2, axis=0), 2, axis=1)
    up_rgb = np.stack([up, up, up], axis=-1)
    bmp_path = Path(tempfile.gettempdir()) / "vani_screen_ocr.bmp"
    _save_rgb_as_bmp(up_rgb, bmp_path)
    return _run_winrt_ocr_on_file(bmp_path, invert_dark=False)


class ScreenTasksTool(BaseTool):
    """Tool that inspects visible windows, background tasks, and live shared screen content."""

    name = "screen_tasks"
    description = (
        "Inspect visible open windows, background running tasks & system resources (CPU, RAM, battery), "
        "and analyze the user's shared screen using local vision + OCR."
    )
    parameters = {
        "action": "One of: 'view_screen', 'list_windows', 'list_tasks', 'focus_window', 'open_task_manager'",
        "target": "Optional window/app name to focus when action='focus_window'",
    }

    def execute(self, **kwargs: Any) -> ToolResult:
        action = str(kwargs.get("action", "view_screen")).lower().strip()
        target = str(kwargs.get("target", "")).strip()
        screen_jpeg = kwargs.get("screen_jpeg")
        user_text = str(kwargs.get("user_text", "")).strip()

        if action in ("list_windows", "windows", "open_apps"):
            return self._list_windows()
        if action in ("list_tasks", "tasks", "system_status", "resources"):
            return self._list_tasks()
        if action in ("focus_window", "switch_window", "focus"):
            return self._focus_window(target)
        if action == "open_task_manager":
            return self._open_task_manager()
        return self._view_screen(screen_jpeg=screen_jpeg, user_text=user_text)

    def _gather_window_and_task_state(self) -> dict[str, Any]:
        wins = get_open_windows()
        other_wins = [w for w in wins if not w["is_vani"]]

        visible_wins = [w for w in other_wins if not w.get("is_minimized")]
        minimized_wins = [w for w in other_wins if w.get("is_minimized")]
        active_win = next((w for w in visible_wins if w["is_active"]), None)

        visible_app_names = {w["app"] for w in other_wins}
        stats = get_system_and_tasks_summary(visible_apps=visible_app_names)

        # Deduplicate while preserving Z-order (topmost first)
        visible_labels: list[str] = []
        seen_vis: set[str] = set()
        for w in visible_wins:
            if w["app"] not in seen_vis:
                seen_vis.add(w["app"])
                visible_labels.append(f"{w['app']} ({w['title'][:50]})")

        minimized_labels: list[str] = []
        seen_min: set[str] = set()
        for w in minimized_wins:
            if w["app"] not in seen_min and w["app"] not in seen_vis:
                seen_min.add(w["app"])
                minimized_labels.append(w["app"])

        return {
            "active_win": active_win,
            "visible_wins": visible_wins,
            "visible_labels": visible_labels,
            "minimized_labels": minimized_labels,
            "background_apps": stats.get("background_apps", []),
            "stats": stats,
        }

    def get_live_pc_context_snippet(self, screen_jpeg: str | None = None, user_text: str = "") -> str:
        """Build a fast live PC & screen context snippet for always-on follow-up awareness."""
        state = self._gather_window_and_task_state()
        vis = state["visible_labels"]
        mins = state["minimized_labels"]
        bg = state["background_apps"]
        active = state["active_win"]
        stats = state["stats"]

        vis_str = ", ".join(vis) if vis else "None (only Vāṇī Voice AI)"
        min_str = ", ".join(mins) if mins else "None"
        bg_str = ", ".join(bg) if bg else "None"
        fg_str = f"{active['app']} ({active['title'][:50]})" if active else "Vāṇī Voice AI"

        lines = [
            "[LIVE PC & SCREEN CONTEXT]",
            f"- Foreground Active Window: {fg_str}",
            f"- Visible Open Windows: {vis_str}",
            f"- Minimized Windows: {min_str}",
            f"- Background / Tray Apps: {bg_str}",
            f"- System: CPU {stats['cpu_percent']}%, RAM {stats['ram_percent']}% ({stats['ram_used_gb']} / {stats['ram_total_gb']} GB)",
        ]

        if screen_jpeg and any(
            k in user_text.lower()
            for k in ("screen", "see", "looking", "error", "read", "show", "window", "pop", "what")
        ):
            b64 = screen_jpeg.split(",", 1)[1] if "," in screen_jpeg else screen_jpeg
            vlm_desc = _analyze_jpeg_with_ollama_vlm(b64, user_text or "Describe what is on this screen.")
            if vlm_desc:
                lines.append(f"- Visual Scene Description (Shared Screen VLM): {vlm_desc}")
            else:
                tmp_jpg = Path(tempfile.gettempdir()) / "vani_shared_frame.jpg"
                try:
                    tmp_jpg.write_bytes(base64.b64decode(b64))
                    ocr_text = _run_winrt_ocr_on_file(tmp_jpg, invert_dark=True)
                    if ocr_text:
                        lines.append(f"- Visible Text on Shared Screen (OCR): {ocr_text[:600]}")
                except Exception:
                    pass

        lines.append(
            "Instruction: Answer accurately in 1 to 3 concise spoken sentences using ONLY these live facts. "
            "Do NOT claim you lack access to the screen or background apps."
        )
        return "\n".join(lines)

    def _list_windows(self) -> ToolResult:
        state = self._gather_window_and_task_state()
        vis = state["visible_labels"]
        mins = state["minimized_labels"]
        bg = state["background_apps"]
        active = state["active_win"]

        vis_str = ", ".join(vis) if vis else "None (only Vāṇī Voice AI is visible)"
        min_str = ", ".join(mins) if mins else "None"
        bg_str = ", ".join(bg) if bg else "None"
        fg_str = f"{active['app']} ({active['title'][:50]})" if active else "Vāṇī Voice AI"

        fallback = f"Open on screen: {vis_str}." + (f" Running in background: {bg_str}." if bg else "")
        llm_context = (
            "[LIVE PC WINDOWS & BACKGROUND APPS]\n"
            f"- Foreground Active Window: {fg_str}\n"
            f"- Visible Open Windows on Desktop: {vis_str}\n"
            f"- Minimized Windows: {min_str}\n"
            f"- Background / Tray Apps (NOT open on screen): {bg_str}\n"
            "Instruction: Answer which apps are open on the desktop in 1 to 2 short spoken sentences. "
            "ONLY list Visible Open Windows as open on screen. Mention Background Apps only if relevant, and explicitly note they are in the background."
        )
        return ToolResult(True, fallback, spoken_response=fallback, data=state, llm_context=llm_context)

    def _list_tasks(self) -> ToolResult:
        state = self._gather_window_and_task_state()
        stats = state["stats"]
        vis = [w["app"] for w in state["visible_wins"]]
        vis_unique = list(dict.fromkeys(vis))
        bg = state["background_apps"]
        top_tasks = stats["top_tasks"]

        top_str = ", ".join(f"{t['name']} ({t['ram_mb']} MB)" for t in top_tasks[:4]) or "None"
        vis_str = ", ".join(vis_unique) if vis_unique else "Vāṇī Voice AI"
        bg_str = ", ".join(bg) if bg else "None"

        batt_str = (
            f", Battery {stats['battery_percent']}% ({'charging' if stats['is_charging'] else 'on battery'})"
            if stats["battery_percent"] is not None
            else ""
        )

        fallback = (
            f"Background apps: {bg_str}. CPU is {stats['cpu_percent']}% and RAM is {stats['ram_percent']}%."
        )
        llm_context = (
            "[LIVE SYSTEM TASKS & RESOURCES]\n"
            f"- Visible Desktop Windows: {vis_str}\n"
            f"- Background User Apps: {bg_str}\n"
            f"- Heaviest Processes: {top_str}\n"
            f"- System Load: CPU {stats['cpu_percent']}%, RAM {stats['ram_percent']}% ({stats['ram_used_gb']} / {stats['ram_total_gb']} GB){batt_str}\n"
            "Instruction: Summarize the running background tasks and system usage in 1 to 2 natural spoken sentences."
        )
        return ToolResult(True, fallback, spoken_response=fallback, data=state, llm_context=llm_context)

    def _view_screen(self, screen_jpeg: str | None = None, user_text: str = "") -> ToolResult:
        state = self._gather_window_and_task_state()
        vis = state["visible_labels"]
        bg = state["background_apps"]
        active = state["active_win"]

        vlm_desc = ""
        ocr_text = ""
        source_mode = "none"

        if screen_jpeg:
            # Strip optional data URL prefix
            b64 = screen_jpeg.split(",", 1)[1] if "," in screen_jpeg else screen_jpeg
            source_mode = "live browser screen share"
            tmp_jpg = Path(tempfile.gettempdir()) / "vani_shared_frame.jpg"
            try:
                tmp_jpg.write_bytes(base64.b64decode(b64))
                vlm_desc = _analyze_jpeg_with_ollama_vlm(b64, user_text or "Describe what is on this screen.")
                if not vlm_desc:
                    ocr_text = _run_winrt_ocr_on_file(tmp_jpg, invert_dark=True)
            except Exception as e:
                logger.warning("Failed processing shared screen JPEG: %s", e)
        else:
            rgb = capture_screen_rgb()
            if rgb is not None:
                source_mode = "local desktop capture"
                ocr_text = _encode_rgb_to_bmp_and_ocr(rgb)

        vis_str = ", ".join(vis) if vis else "Vāṇī Voice AI"
        bg_str = ", ".join(bg) if bg else "None"
        fg_str = f"{active['app']} ({active['title'][:50]})" if active else "Vāṇī Voice AI"

        if source_mode == "none" and not vis:
            fallback = (
                "I cannot see your screen pixels yet. Click the Share Screen button in the app so I can analyze your screen."
            )
            return ToolResult(True, fallback, spoken_response=fallback, data=state)

        fallback = f"On your screen I see {vis_str}."
        llm_context = (
            "[LIVE SCREEN SHARE & WINDOW CONTEXT]\n"
            f"- Screen Source: {source_mode}\n"
            f"- Foreground Active Window: {fg_str}\n"
            f"- Visible Open Windows: {vis_str}\n"
            f"- Background Apps (hidden/tray): {bg_str}\n"
            f"- Visual Scene Description (VLM): {vlm_desc if vlm_desc else '(No local VLM installed; rely on window titles and OCR below)'}\n"
            f"- Visible Text on Shared Screen (OCR): {ocr_text if ocr_text else '(No text extracted)'}\n"
            "Instruction: Answer the user's question about what is on their screen in 1 to 2 natural spoken sentences using ONLY the facts above. "
            "Do NOT claim an app is open unless it appears in Visible Open Windows or Visual Scene Description."
        )
        return ToolResult(
            success=True,
            message=f"Analyzed screen via {source_mode}",
            spoken_response=fallback,
            data={"source": source_mode, "vlm": vlm_desc, "ocr": ocr_text[:300]},
            llm_context=llm_context,
        )

    def _focus_window(self, target: str) -> ToolResult:
        if not target:
            return ToolResult(False, "No window target specified.", spoken_response="Which window would you like me to switch to?")

        wins = get_open_windows()
        t_low = target.lower()
        match = None
        for w in wins:
            if t_low in w["app"].lower() or t_low in w["title"].lower() or t_low in w["exe"].lower():
                match = w
                break

        if not match or not match.get("hwnd"):
            spoken = f"I couldn't find an open window matching {target}."
            return ToolResult(False, spoken, spoken_response=spoken)

        if os.name == "nt":
            user32 = ctypes.windll.user32
            hwnd = match["hwnd"]
            user32.ShowWindow(hwnd, 9)
            user32.SetForegroundWindow(hwnd)

        spoken = f"Switched to {match['app']}."
        return ToolResult(True, spoken, spoken_response=spoken)

    def _open_task_manager(self) -> ToolResult:
        if os.name == "nt":
            subprocess.Popen(["taskmgr.exe"], shell=True)
        spoken = "Opening Windows Task Manager."
        return ToolResult(True, spoken, spoken_response=spoken)
