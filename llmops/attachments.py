# llmops/attachments.py
"""Extract incident context from user-uploaded files (Attach button).

Supports the file types an IT user realistically attaches to a ticket:
plain text / logs / markdown / csv, and PDF (SOP or exported ticket).
Images/screenshots are acknowledged by name (vision OCR is out of scope
for the current model set). Pure-Python; PDF path lazily imports pypdf.

Returned text is appended to the incident so retrieval + resolution are
grounded on the attachment too.
"""
from __future__ import annotations

import io
import os
from typing import Any

TEXT_EXTS = {".txt", ".log", ".md", ".markdown", ".csv", ".json", ".yaml", ".yml"}
PDF_EXTS = {".pdf"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
MAX_CHARS = 4000  # cap appended context so prompts stay bounded


def _ext(name: str) -> str:
    return os.path.splitext(name.lower())[1]


def extract_text(filename: str, data: bytes) -> dict[str, Any]:
    """Return {"kind", "chars", "text", "note"} for an uploaded file.

    Never raises on bad input: unreadable files come back with kind
    'error' and an explanatory note so the UI can show a small caption.
    """
    ext = _ext(filename)
    try:
        if ext in TEXT_EXTS:
            text = data.decode("utf-8", errors="replace").strip()
            return {"kind": "text", "chars": len(text),
                    "text": text[:MAX_CHARS], "note": ""}
        if ext in PDF_EXTS:
            return _extract_pdf(data)
        if ext in IMAGE_EXTS:
            return {"kind": "image", "chars": 0, "text": "",
                    "note": f"image '{filename}' attached (screenshot noted; "
                            "text not extracted)"}
        # unknown -> best-effort utf-8
        text = data.decode("utf-8", errors="replace").strip()
        if text and text.isprintable():
            return {"kind": "text", "chars": len(text),
                    "text": text[:MAX_CHARS], "note": f"read '{filename}' as text"}
        return {"kind": "error", "chars": 0, "text": "",
                "note": f"unsupported file type '{ext}'"}
    except Exception as exc:  # noqa: BLE001
        return {"kind": "error", "chars": 0, "text": "",
                "note": f"could not read '{filename}': {type(exc).__name__}"}


def _extract_pdf(data: bytes) -> dict[str, Any]:
    try:
        from pypdf import PdfReader  # lazy optional dependency
    except ImportError:
        return {"kind": "error", "chars": 0, "text": "",
                "note": "PDF attached but pypdf not installed "
                        "(pip install pypdf)"}
    try:
        reader = PdfReader(io.BytesIO(data))
        parts = []
        for page in reader.pages[:20]:  # cap pages
            parts.append(page.extract_text() or "")
        text = "\n".join(parts).strip()
        return {"kind": "pdf", "chars": len(text), "text": text[:MAX_CHARS],
                "note": f"extracted {len(reader.pages)} page(s) of PDF text"}
    except Exception as exc:  # noqa: BLE001
        return {"kind": "error", "chars": 0, "text": "",
                "note": f"PDF unreadable: {type(exc).__name__}"}


def merge_incident(incident: str, attachment: dict[str, Any]) -> str:
    """Append extracted attachment text to the incident description."""
    text = (attachment or {}).get("text", "").strip()
    if not text:
        return incident
    return (
        f"{incident.strip()}\n\n"
        f"--- Attached context ({attachment.get('kind', 'file')}) ---\n{text}"
    )
