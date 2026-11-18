"""Autonomous File Finder & Safe Folder Organizer Tool for Vāṇī on Windows."""

from __future__ import annotations

import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any

from .base import BaseTool, ToolResult

logger = logging.getLogger("v2v.tools.file_manager")

_EXTENSION_CATEGORIES: dict[str, str] = {
    ".pdf": "Documents",
    ".doc": "Documents",
    ".docx": "Documents",
    ".ppt": "Documents",
    ".pptx": "Documents",
    ".xls": "Documents",
    ".xlsx": "Documents",
    ".txt": "Documents",
    ".md": "Documents",
    ".rtf": "Documents",
    ".png": "Images",
    ".jpg": "Images",
    ".jpeg": "Images",
    ".gif": "Images",
    ".webp": "Images",
    ".bmp": "Images",
    ".svg": "Images",
    ".py": "Code_and_Data",
    ".js": "Code_and_Data",
    ".ts": "Code_and_Data",
    ".html": "Code_and_Data",
    ".css": "Code_and_Data",
    ".json": "Code_and_Data",
    ".csv": "Code_and_Data",
    ".yaml": "Code_and_Data",
    ".yml": "Code_and_Data",
    ".sql": "Code_and_Data",
    ".mp3": "Audio_Video",
    ".wav": "Audio_Video",
    ".m4a": "Audio_Video",
    ".mp4": "Audio_Video",
    ".mkv": "Audio_Video",
    ".mov": "Audio_Video",
    ".zip": "Archives_and_Installers",
    ".rar": "Archives_and_Installers",
    ".7z": "Archives_and_Installers",
    ".tar": "Archives_and_Installers",
    ".gz": "Archives_and_Installers",
    ".exe": "Archives_and_Installers",
    ".msi": "Archives_and_Installers",
}

_SKIP_EXTENSIONS = {".lnk", ".bat", ".cmd", ".ini", ".url", ".tmp", ".crdownload", ".part"}


def _search_roots() -> list[Path]:
    candidates = [
        Path(os.path.expandvars(r"%USERPROFILE%\OneDrive\Desktop")),
        Path(os.path.expandvars(r"%USERPROFILE%\Desktop")),
        Path(os.path.expandvars(r"%USERPROFILE%\OneDrive\Documents")),
        Path(os.path.expandvars(r"%USERPROFILE%\Documents")),
        Path(os.path.expandvars(r"%USERPROFILE%\Downloads")),
        Path(__file__).resolve().parents[3] / "docs",
    ]
    roots: list[Path] = []
    seen: set[str] = set()
    for p in candidates:
        try:
            res = str(p.resolve()).lower()
            if p.exists() and p.is_dir() and res not in seen:
                seen.add(res)
                roots.append(p)
        except Exception:
            pass
    return roots


def _iter_files_shallow(root: Path, max_depth: int = 2) -> list[Path]:
    found: list[Path] = []
    try:
        for entry in root.iterdir():
            name_low = entry.name.lower()
            if name_low.startswith(".") or name_low in ("node_modules", "__pycache__", ".git", ".venv"):
                continue
            if entry.is_file():
                found.append(entry)
            elif entry.is_dir() and max_depth > 1:
                found.extend(_iter_files_shallow(entry, max_depth=max_depth - 1))
    except PermissionError:
        pass
    return found


class FileManagerTool(BaseTool):
    """Find/open user files across PC folders and safely organize messy folders by category."""

    name = "file_manager"
    description = (
        "Search Desktop, Documents, Downloads, and docs folders to find and open files by name, "
        "or safely organize loose files into categorized subfolders."
    )
    parameters = {
        "action": "One of: 'find_file', 'organize_folder'",
        "query": "Filename or keywords for find_file",
        "target": "Folder name for organize_folder (e.g., 'downloads', 'demo_folder')",
    }

    def execute(self, **kwargs: Any) -> ToolResult:
        action = str(kwargs.get("action", "find_file")).lower().strip()
        if action == "organize_folder":
            return self._organize_folder(
                str(kwargs.get("target", "downloads")),
                dry_run=bool(kwargs.get("dry_run", False)),
            )
        return self._find_file(
            str(kwargs.get("query", "")),
            open_match=bool(kwargs.get("open_match", True)),
        )

    def _find_file(self, query: str, open_match: bool = True) -> ToolResult:
        clean_q = re.sub(
            r"\b(?:my|the|a|file|files|document|named|called|please)\b",
            " ",
            query.lower(),
        )
        tokens = [t for t in re.findall(r"[a-z0-9]+", clean_q) if len(t) >= 2]
        if not tokens:
            return ToolResult(False, "No filename given.", spoken_response="Which file would you like me to find?")

        scored: list[tuple[float, Path]] = []
        for root in _search_roots():
            for f in _iter_files_shallow(root, max_depth=2):
                if f.suffix.lower() in (".ini", ".tmp"):
                    continue
                fname_low = f.name.lower()
                stem_low = f.stem.lower()
                hits = sum(1 for t in tokens if t in fname_low)
                if hits == 0:
                    continue
                score = hits * 10.0
                if " ".join(tokens) in stem_low:
                    score += 8.0
                if f.suffix.lower() in (".pptx", ".pdf", ".docx", ".md", ".txt", ".xlsx"):
                    score += 2.0
                scored.append((score, f))

        if not scored:
            q_label = " ".join(tokens)
            spoken = f"I couldn't find any file matching {q_label} in your Desktop, Documents, or Downloads."
            return ToolResult(False, spoken, spoken_response=spoken)

        scored.sort(key=lambda item: (item[0], item[1].stat().st_mtime), reverse=True)
        best = scored[0][1]

        if open_match and os.name == "nt":
            try:
                os.startfile(str(best))
            except Exception as e:
                logger.warning("Failed to open %s: %s", best, e)

        folder_label = best.parent.name
        spoken = f"Found {best.name} in {folder_label} and opened it for you." if open_match else f"Found {best.name} in {folder_label}."
        return ToolResult(
            True,
            spoken,
            spoken_response=spoken,
            data={"path": str(best), "matches": [str(p) for _, p in scored[:5]]},
        )

    def _resolve_target_folder(self, target: str) -> Path | None:
        t = target.lower().strip()
        if "demo" in t:
            for base in (
                Path(os.path.expandvars(r"%USERPROFILE%\OneDrive\Desktop")),
                Path(os.path.expandvars(r"%USERPROFILE%\Desktop")),
            ):
                if base.exists():
                    return base / "Demo_Folder"
        if "document" in t:
            for p in (
                Path(os.path.expandvars(r"%USERPROFILE%\OneDrive\Documents")),
                Path(os.path.expandvars(r"%USERPROFILE%\Documents")),
            ):
                if p.exists():
                    return p
        dl = Path(os.path.expandvars(r"%USERPROFILE%\Downloads"))
        return dl if dl.exists() else None

    def _organize_folder(self, target: str = "downloads", dry_run: bool = False) -> ToolResult:
        folder = self._resolve_target_folder(target)
        if not folder or not folder.exists():
            spoken = f"I couldn't find the {target} folder on your PC."
            return ToolResult(False, spoken, spoken_response=spoken)

        moved_count = 0
        used_categories: set[str] = set()

        for entry in list(folder.iterdir()):
            if not entry.is_file():
                continue
            ext = entry.suffix.lower()
            if ext in _SKIP_EXTENSIONS or entry.name.startswith("."):
                continue
            cat = _EXTENSION_CATEGORIES.get(ext)
            if not cat:
                continue

            dest_dir = folder / cat
            dest_path = dest_dir / entry.name
            if not dry_run:
                try:
                    dest_dir.mkdir(exist_ok=True)
                    if dest_path.exists():
                        dest_path = dest_dir / f"{entry.stem}_{int(entry.stat().st_mtime)}{entry.suffix}"
                    shutil.move(str(entry), str(dest_path))
                except Exception as e:
                    logger.debug("Skipped moving %s: %s", entry.name, e)
                    continue
            moved_count += 1
            used_categories.add(cat)

        if moved_count == 0:
            spoken = f"Your {folder.name} folder is already neat and organized."
            return ToolResult(True, spoken, spoken_response=spoken, data={"moved": 0, "folder": str(folder)})

        spoken = f"Organized {moved_count} file{'s' if moved_count != 1 else ''} in {folder.name} into {len(used_categories)} folders."
        if os.name == "nt" and not dry_run:
            try:
                os.startfile(str(folder))
            except Exception:
                pass
        return ToolResult(
            True,
            spoken,
            spoken_response=spoken,
            data={"moved": moved_count, "categories": sorted(used_categories), "folder": str(folder)},
        )
