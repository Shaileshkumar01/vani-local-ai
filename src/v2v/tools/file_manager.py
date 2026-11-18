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
        if action == "list_folder":
            return self._list_folder(
                str(kwargs.get("target", "downloads")),
                limit=int(kwargs.get("limit", 5) or 5),
            )
        if action == "reveal_file":
            return self._reveal_file(
                str(kwargs.get("path", "")),
                open_explorer=bool(kwargs.get("open_explorer", True)),
            )
        return self._find_file(
            str(kwargs.get("query", "")),
            open_match=bool(kwargs.get("open_match", True)),
        )

    # ------------------------------------------------------------ list folder
    @staticmethod
    def _resolve_listing_folder(target: str) -> Path | None:
        t = (target or "").lower()
        home = Path(os.path.expandvars("%USERPROFILE%"))
        project_root = Path(__file__).resolve().parents[3]
        if "workspace" in t:
            for p in (home / "OneDrive" / "Desktop" / "Vani_Workspace", home / "Desktop" / "Vani_Workspace", project_root / "Vani_Workspace"):
                if p.exists():
                    return p
            return None
        if t.strip() in ("docs", "docs folder", "project docs"):
            p = project_root / "docs"
            return p if p.exists() else None
        if "desktop" in t:
            candidates = [home / "OneDrive" / "Desktop", home / "Desktop"]
        elif "document" in t:
            candidates = [home / "OneDrive" / "Documents", home / "Documents"]
        elif "download" in t:
            candidates = [home / "Downloads"]
        elif "picture" in t or "photo" in t:
            candidates = [home / "OneDrive" / "Pictures", home / "Pictures"]
        else:
            return None
        for p in candidates:
            if p.exists() and p.is_dir():
                return p
        return None

    def _list_folder(self, target: str, limit: int = 5) -> ToolResult:
        folder = self._resolve_listing_folder(target)
        if folder is None:
            spoken = f"I couldn't find a {target} folder on your PC."
            return ToolResult(False, spoken, spoken_response=spoken)

        files: list[Path] = []
        sub_dirs = 0
        try:
            for entry in folder.iterdir():
                name_low = entry.name.lower()
                if name_low.startswith((".", "~$")) or name_low in ("desktop.ini", "thumbs.db"):
                    continue
                if entry.is_dir():
                    sub_dirs += 1
                elif entry.is_file() and entry.suffix.lower() not in _SKIP_EXTENSIONS:
                    files.append(entry)
        except PermissionError:
            spoken = f"I don't have permission to read your {folder.name} folder."
            return ToolResult(False, spoken, spoken_response=spoken)

        def _mtime(p: Path) -> float:
            try:
                return p.stat().st_mtime
            except OSError:
                return 0.0

        files.sort(key=_mtime, reverse=True)
        limit = max(1, min(limit, 10))
        top = files[:limit]
        label = folder.name
        if not top:
            spoken = f"Your {label} folder has no files right now" + (f", only {sub_dirs} subfolders." if sub_dirs else ".")
            return ToolResult(True, spoken, spoken_response=spoken, data={"folder": str(folder), "files": []})

        names = [p.name for p in top]
        if len(names) == 1:
            joined = names[0]
        else:
            joined = ", ".join(names[:-1]) + f", and {names[-1]}"
        spoken = (
            f"Your {label} folder has {len(files)} file{'s' if len(files) != 1 else ''}"
            + (f" and {sub_dirs} subfolder{'s' if sub_dirs != 1 else ''}" if sub_dirs else "")
            + f". The {len(top)} most recent {'are' if len(top) != 1 else 'is'}: {joined}."
        )
        return ToolResult(
            True,
            spoken,
            spoken_response=spoken,
            data={"folder": str(folder), "files": [str(p) for p in top], "total": len(files)},
        )

    # ------------------------------------------------------------ reveal file
    def _reveal_file(self, path: str, open_explorer: bool = True) -> ToolResult:
        if not path:
            spoken = "I haven't created or opened any file yet in this session, so there's nothing to locate."
            return ToolResult(False, spoken, spoken_response=spoken)
        p = Path(path)
        if not p.exists():
            spoken = f"The file {p.name} no longer exists at its saved location."
            return ToolResult(False, spoken, spoken_response=spoken, data={"path": path})
        if open_explorer and os.name == "nt":
            try:
                import subprocess

                subprocess.Popen(["explorer", "/select,", str(p)])
            except Exception as e:
                logger.warning("Could not reveal %s in Explorer: %s", p, e)
        spoken = (
            f"{p.name} is saved in the {p.parent.name} folder"
            + (", and I've opened that folder with the file highlighted." if open_explorer else ".")
        )
        return ToolResult(
            True,
            spoken,
            spoken_response=spoken,
            data={"path": str(p), "display_text": f"{spoken}\nFull path: {p}"},
        )

    def _find_file(self, query: str, open_match: bool = True) -> ToolResult:
        q_norm = (
            query.lower()
            .replace("shellish", "shailesh")
            .replace("shellfish", "shailesh")
            .replace("chalice", "shailesh")
        )
        clean_q = re.sub(
            r"\b(?:i|need|want|you|to|find|locate|open|search|for|my|the|a|an|file|files|document|named|called|please|on|pc|desktop)\b",
            " ",
            q_norm,
        )
        tokens = [t for t in re.findall(r"[a-z0-9]+", clean_q) if len(t) >= 2]
        if not tokens:
            return ToolResult(False, "No filename given.", spoken_response="Which file would you like me to find?")

        scored: list[tuple[float, Path]] = []
        for root in _search_roots():
            for f in _iter_files_shallow(root, max_depth=3):
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
