"""Module 1: Normalizer — whitespace cleaning, JSON canonicalization, type marking.

This is the first stage of the optimization pipeline. It takes raw, user-written
messages and produces a clean, deterministic representation.
"""

import json
import re
from typing import Any


# ── Whitespace Sanitizer ───────────────────────────────────────────────


class WhitespaceSanitizer:
    """Normalize whitespace in message content without changing semantics.

    Operations:
        - \\r\\n → \\n  (Windows to Unix line endings)
        - Strip BOM if present
        - Strip trailing whitespace from each line
        - Collapse multiple blank lines into one
        - Remove trailing newlines at end of content
    """

    BOM = "﻿"

    @classmethod
    def sanitize(cls, content: str) -> str:
        if not isinstance(content, str):
            return content

        text = content
        # Strip BOM
        if text.startswith(cls.BOM):
            text = text[len(cls.BOM):]

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Strip trailing whitespace per line
        lines = text.split("\n")
        lines = [line.rstrip() for line in lines]

        # Collapse 3+ consecutive blank lines into 1
        result = []
        blank_count = 0
        for line in lines:
            if line.strip() == "":
                blank_count += 1
                if blank_count <= 1:
                    result.append("")
            else:
                blank_count = 0
                result.append(line)

        # Remove trailing blank lines
        while result and result[-1].strip() == "":
            result.pop()

        return "\n".join(result)

    @classmethod
    def sanitize_messages(cls, messages: list[dict]) -> list[dict]:
        return [
            {k: cls.sanitize(v) if isinstance(v, str) else v
             for k, v in msg.items()}
            for msg in messages
        ]


# ── JSON Canonicalizer ──────────────────────────────────────────────────


class JsonCanonicalizer:
    """Recursively reorder JSON object keys for deterministic serialization.

    The key insight: same JSON data → different string representation
    → different tokens → cache miss. By always sorting keys, we ensure
    identical data produces identical strings.
    """

    @classmethod
    def _canonicalize_value(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {k: cls._canonicalize_value(v) for k, v in sorted(value.items())}
        if isinstance(value, list):
            return [cls._canonicalize_value(item) for item in value]
        return value

    @classmethod
    def canonicalize(cls, data: Any) -> str:
        """Return a deterministic JSON string for the given data.

        Args:
            data: JSON-serializable object (dict, list, etc.)

        Returns:
            A JSON string with sorted keys, no extra whitespace.
        """
        canonicalized = cls._canonicalize_value(data)
        return json.dumps(canonicalized, ensure_ascii=False,
                          sort_keys=True, separators=(",", ":"))

    @classmethod
    def try_canonicalize_content(cls, content: str) -> str:
        """If content is parseable JSON, return canonicalized form.

        If parsing fails (not valid JSON), return content unchanged.
        """
        if not isinstance(content, str):
            return content

        stripped = content.strip()
        if not (stripped.startswith("{") or stripped.startswith("[")):
            return content

        try:
            parsed = json.loads(stripped)
            return cls.canonicalize(parsed)
        except (json.JSONDecodeError, ValueError):
            return content

    @classmethod
    def canonicalize_messages(cls, messages: list[dict]) -> list[dict]:
        result = []
        for msg in messages:
            new_msg = {}
            for k, v in msg.items():
                if isinstance(v, str):
                    new_msg[k] = cls.try_canonicalize_content(v)
                elif isinstance(v, (dict, list)):
                    new_msg[k] = cls._canonicalize_value(v)
                else:
                    new_msg[k] = v
            result.append(new_msg)
        return result


# ── Type Marker ─────────────────────────────────────────────────────────


class TypeMarker:
    """Tag each message with its type for downstream modules.

    Adds a '_role_type' marker based on the message's 'role' field.
    This is informational only — the original role field is preserved.
    """

    VALID_ROLES = {"system", "user", "assistant", "tool"}

    @classmethod
    def mark(cls, messages: list[dict]) -> list[dict]:
        result = []
        for msg in messages:
            role = msg.get("role", "")
            # Determine role type
            role_type = role if role in cls.VALID_ROLES else "other"
            # Add _role_type marker (for internal use, not serialized to API)
            result.append({**msg, "_role_type": role_type})
        return result


# ── Normalizer Orchestrator ────────────────────────────────────────────


class Normalizer:
    """First stage of the optimization pipeline.

    Applies: WhitespaceSanitizer → JsonCanonicalizer → TypeMarker
    """

    @classmethod
    def run(cls, messages: list[dict]) -> list[dict]:
        """Normalize a list of messages.

        Args:
            messages: Raw input messages list.

        Returns:
            Cleaned, canonicalized, and tagged messages.
        """
        msgs = WhitespaceSanitizer.sanitize_messages(messages)
        msgs = JsonCanonicalizer.canonicalize_messages(msgs)
        msgs = TypeMarker.mark(msgs)
        return msgs
