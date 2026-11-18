"""Local Document / Knowledge Base RAG Tool for Vāṇī.

Scans the project's `docs/` directory for `.md`, `.txt`, `.csv`, and `.pdf` files,
chunks content by section/paragraph, scores relevance using TF/keyword overlap,
and injects the top passages into LLM context for 100% offline spoken Q&A.
"""

from __future__ import annotations

import logging
import math
import re
import zlib
from collections import Counter
from pathlib import Path
from typing import Any

from .base import BaseTool, ToolResult

logger = logging.getLogger("v2v.tools.doc_rag")

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "in", "on", "at", "to", "for", "of",
    "and", "or", "my", "our", "your", "what", "which", "who", "how", "does", "do", "did",
    "say", "says", "about", "according", "from", "document", "documents", "doc", "docs",
    "report", "notes", "file", "pdf", "summarize", "summary", "tell", "me", "explain",
    "can", "you", "please", "project",
}


def _default_docs_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "docs"


def _extract_pdf_text_fallback(pdf_path: Path) -> str:
    """Extract readable ASCII/UTF-8 text from PDF using pypdf/pymupdf if present, else pure-Python zlib stream scan."""
    try:
        import pypdf  # type: ignore

        reader = pypdf.PdfReader(str(pdf_path))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception:
        pass

    try:
        import fitz  # type: ignore

        doc = fitz.open(str(pdf_path))
        return "\n\n".join(page.get_text() for page in doc)
    except Exception:
        pass

    # Pure-Python fallback: decompress FlateDecode PDF content streams and extract Tj/TJ string literals
    try:
        data = pdf_path.read_bytes()
        texts: list[str] = []
        for match in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", data, re.DOTALL):
            raw_stream = match.group(1)
            try:
                decompressed = zlib.decompress(raw_stream).decode("latin-1", errors="ignore")
            except Exception:
                continue
            # Grab parenthesized PDF text strings inside BT..ET blocks
            for s_match in re.finditer(r"\(([^()\\]*(?:\\.[^()\\]*)*)\)", decompressed):
                s = s_match.group(1).replace(r"\(", "(").replace(r"\)", ")").replace(r"\n", " ")
                cleaned = re.sub(r"[^ -~]+", " ", s).strip()
                if len(cleaned) >= 2:
                    texts.append(cleaned)
        return " ".join(texts)
    except Exception as e:
        logger.debug("Fallback PDF extraction failed for %s: %s", pdf_path.name, e)
        return ""


def _read_document_text(path: Path) -> str:
    ext = path.suffix.lower()
    try:
        if ext in (".md", ".txt", ".csv", ".json", ".py"):
            return path.read_text(encoding="utf-8", errors="ignore")
        if ext == ".pdf":
            return _extract_pdf_text_fallback(path)
    except Exception as e:
        logger.warning("Failed reading %s: %s", path, e)
    return ""


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]{2,}", text.lower()) if t not in _STOPWORDS]


def _chunk_text(text: str, source_name: str) -> list[dict[str, str]]:
    """Split markdown or plain text into clean paragraph/section chunks."""
    # Split on markdown headings or double newlines
    raw_blocks = re.split(r"\n(?=#{1,4}\s)|\n\s*\n", text)
    chunks: list[dict[str, str]] = []
    current_heading = ""
    buf = ""

    for blk in raw_blocks:
        blk = blk.strip()
        if not blk:
            continue
        if blk.startswith("#"):
            lines = blk.splitlines()
            current_heading = lines[0].lstrip("# ").strip()
            body = "\n".join(lines[1:]).strip()
            if body:
                text_out = f"[{current_heading}] {body}"
                chunks.append({"source": source_name, "text": text_out})
            elif current_heading:
                buf = f"[{current_heading}] "
        else:
            combined = (buf + blk).strip() if buf else blk
            buf = ""
            if len(combined) > 700:
                # Split oversized paragraphs
                for i in range(0, len(combined), 550):
                    part = combined[i : i + 600].strip()
                    if part:
                        chunks.append({"source": source_name, "text": part})
            else:
                chunks.append({"source": source_name, "text": combined})
    return chunks


class DocRAGTool(BaseTool):
    """Offline Local Document / PDF RAG tool over `docs/`."""

    name = "doc_rag"
    description = (
        "Search local documents, project reports, PDFs, and notes in the `docs/` folder "
        "and answer questions using grounded retrieved passages."
    )
    parameters = {
        "action": "One of: 'query', 'summarize', 'list_docs'",
        "query": "User question or search topic",
    }

    def __init__(self, docs_dir: Path | None = None) -> None:
        self.docs_dir = docs_dir or _default_docs_dir()

    def _load_chunks(self) -> tuple[list[Path], list[dict[str, str]]]:
        if not self.docs_dir.exists():
            return [], []
        files = sorted(
            p for p in self.docs_dir.iterdir()
            if p.is_file() and p.suffix.lower() in (".md", ".txt", ".pdf", ".csv")
        )
        all_chunks: list[dict[str, str]] = []
        for f in files:
            content = _read_document_text(f)
            if content.strip():
                all_chunks.extend(_chunk_text(content, f.name))
        return files, all_chunks

    def execute(self, **kwargs: Any) -> ToolResult:
        action = str(kwargs.get("action", "query")).lower().strip()
        query = str(kwargs.get("query", "")).strip()

        files, chunks = self._load_chunks()
        if not files or not chunks:
            spoken = "Your docs folder doesn't have any readable documents yet. Drop a PDF, Markdown, or text file into the docs folder."
            return ToolResult(False, spoken, spoken_response=spoken)

        if action == "list_docs":
            names = ", ".join(f.name for f in files)
            spoken = f"I found {len(files)} local document{'s' if len(files) != 1 else ''} in your docs folder: {names}."
            return ToolResult(True, spoken, spoken_response=spoken, data={"files": [f.name for f in files]})

        q_tokens = _tokenize(query)
        if action == "summarize" or not q_tokens:
            # Take overview + top structural chunks across docs
            top_chunks = chunks[:4]
        else:
            q_counts = Counter(q_tokens)
            scored: list[tuple[float, dict[str, str]]] = []
            for ch in chunks:
                c_tokens = _tokenize(ch["text"])
                if not c_tokens:
                    continue
                c_counts = Counter(c_tokens)
                overlap = sum(min(q_counts[t], c_counts[t]) * (1.5 if len(t) > 5 else 1.0) for t in q_counts)
                # Boost exact phrase/substring matches
                q_clean = re.sub(r"\s+", " ", " ".join(q_tokens))
                if q_clean and q_clean in ch["text"].lower():
                    overlap += 3.0
                score = overlap / (math.sqrt(len(c_tokens)) + 1.0)
                if score > 0:
                    scored.append((score, ch))
            scored.sort(key=lambda item: item[0], reverse=True)
            top_chunks = [item[1] for item in scored[:3]] if scored else chunks[:3]

        passages = "\n\n".join(f"From {c['source']}:\n{c['text']}" for c in top_chunks)
        llm_context = (
            "[LOCAL DOCUMENT RAG CONTEXT]\n"
            f"{passages[:1800]}\n"
            "Instruction: Answer the user's question accurately in 1 to 3 clear spoken sentences using ONLY the retrieved document passages above."
        )
        fallback = top_chunks[0]["text"][:220]
        return ToolResult(
            success=True,
            message=f"Retrieved {len(top_chunks)} relevant passage(s) from {len(files)} doc(s).",
            spoken_response=fallback,
            data={"files": [f.name for f in files], "passages_count": len(top_chunks)},
            llm_context=llm_context,
        )
