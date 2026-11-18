"""Hands-Free WhatsApp & Email Dispatcher Tool for Vāṇī on Windows."""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.parse
import webbrowser
from pathlib import Path
from typing import Any

from .base import BaseTool, ToolResult

logger = logging.getLogger("v2v.tools.comms")


def _contacts_path() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "contacts.json"


def _load_contacts() -> dict[str, dict[str, str]]:
    path = _contacts_path()
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return {str(k).lower(): v for k, v in raw.items() if isinstance(v, dict)}
    except Exception as e:
        logger.warning("Failed reading contacts.json: %s", e)
    return {}


class CommsTool(BaseTool):
    """Compose and launch WhatsApp messages and Email drafts via native OS handlers."""

    name = "comms"
    description = (
        "Send or prepare hands-free WhatsApp messages via whatsapp:// URI, "
        "or compose Email drafts via mailto: URI using optional data/contacts.json."
    )
    parameters = {
        "action": "One of: 'whatsapp', 'email'",
        "recipient": "Contact name, phone number, or email address",
        "message": "Message or email body text",
        "subject": "Email subject line",
    }

    def execute(self, **kwargs: Any) -> ToolResult:
        action = str(kwargs.get("action", "whatsapp")).lower().strip()
        recipient = str(kwargs.get("recipient", "")).strip()
        message = str(kwargs.get("message", "")).strip()
        subject = str(kwargs.get("subject", "")).strip()
        dry_run = bool(kwargs.get("dry_run", False))

        if action == "email":
            return self._draft_email(recipient=recipient, subject=subject, message=message, dry_run=dry_run)
        return self._send_whatsapp(recipient=recipient, message=message, dry_run=dry_run)

    def _resolve_contact(self, recipient: str) -> tuple[str, str]:
        """Return (phone, email) resolved from contacts.json or raw string."""
        if not recipient:
            return "", ""
        r_low = recipient.lower().strip()
        contacts = _load_contacts()
        info = contacts.get(r_low, {})
        phone = str(info.get("phone", "")).strip()
        email = str(info.get("email", "")).strip()

        if not phone and re.fullmatch(r"\+?[0-9\-\s]{7,18}", recipient):
            phone = re.sub(r"[^0-9+]", "", recipient)
        if not email and "@" in recipient:
            email = recipient.strip()
        return phone, email

    def _send_whatsapp(self, recipient: str, message: str, dry_run: bool = False) -> ToolResult:
        if not message:
            return ToolResult(False, "No WhatsApp message specified.", spoken_response="What message would you like to send on WhatsApp?")

        phone, _ = self._resolve_contact(recipient)
        encoded_text = urllib.parse.quote(message)
        if phone:
            clean_phone = re.sub(r"[^0-9]", "", phone)
            uri = f"whatsapp://send?phone={clean_phone}&text={encoded_text}"
        else:
            uri = f"whatsapp://send?text={encoded_text}"

        if not dry_run and os.name == "nt":
            try:
                os.startfile(uri)
            except Exception as e:
                logger.warning("whatsapp:// launch failed, opening web fallback: %s", e)
                webbrowser.open(f"https://wa.me/{phone or ''}?text={encoded_text}")

        to_label = f" to {recipient}" if recipient else ""
        spoken = f"Opening WhatsApp with your message{to_label}: {message}."
        return ToolResult(True, spoken, spoken_response=spoken, data={"uri": uri, "recipient": recipient, "phone": phone})

    def _draft_email(self, recipient: str, subject: str, message: str, dry_run: bool = False) -> ToolResult:
        _, email_addr = self._resolve_contact(recipient)
        subj = subject or "Quick Update"
        body = message or f"Hi {recipient or 'there'},\n\nRegarding {subj}.\n\nBest regards,"

        params = urllib.parse.urlencode({"subject": subj, "body": body}, quote_via=urllib.parse.quote)
        uri = f"mailto:{email_addr}?{params}"

        if not dry_run:
            webbrowser.open(uri)

        to_label = f" to {recipient}" if recipient else ""
        spoken = f"Opening a draft email{to_label} about {subj}."
        return ToolResult(True, spoken, spoken_response=spoken, data={"uri": uri, "email": email_addr, "subject": subj})
