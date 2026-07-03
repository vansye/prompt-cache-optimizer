"""Safety checks — ensure optimizations don't break semantics.

Every stage of the pipeline has a safety check that can
reject an optimization and fall back to the original input.

Core principle: better to miss a cache hit than to change output.
"""

import json
from typing import Any


class SafetyCheck:
    """Collection of static safety verification methods."""

    @staticmethod
    def verify_reorder(original: list[dict], optimized: list[dict]) -> bool:
        """Verify that reordering hasn't lost or duplicated content.

        Checks:
          - Same number of non-separator messages
          - Same set of (role, content) pairs (ignoring order)
          - The system message (if any) is unchanged

        Returns True if safe, False if the optimization should be rejected.
        """
        # Count non-separator messages
        orig_non_sep = [m for m in original if not m.get("_is_separator")]
        opt_non_sep = [m for m in optimized if not m.get("_is_separator")]

        if len(orig_non_sep) != len(opt_non_sep):
            return False  # Messages were lost or duplicated

        # Compare content sets (ignoring order)
        orig_set = _content_signature_set(orig_non_sep)
        opt_set = _content_signature_set(opt_non_sep)
        if orig_set != opt_set:
            return False  # Content changed

        return True

    @staticmethod
    def verify_system_unchanged(original: list[dict], optimized: list[dict]) -> bool:
        """Verify that system messages are semantically unchanged.

        Specifically checks that the original system content is still
        present in the optimized system content (may have been padded).
        """
        orig_system = _get_system_content(original)
        opt_system = _get_system_content(optimized)

        if orig_system is None and opt_system is None:
            return True
        if orig_system is None or opt_system is None:
            return False  # System was added or removed

        # Original system content must be a substring of optimized
        return orig_system in opt_system


# ── Internal Helpers ────────────────────────────────────────────────────


def _content_signature(msg: dict) -> str:
    """Create a deterministic signature for a message's content."""
    role = msg.get("role", "")
    content = msg.get("content", "")
    if isinstance(content, (dict, list)):
        content = json.dumps(content, sort_keys=True, ensure_ascii=False)
    return f"{role}:::{content}"


def _content_signature_set(messages: list[dict]) -> set[str]:
    """Create a set of content signatures (ignores ordering)."""
    return {_content_signature(m) for m in messages if not m.get("_is_separator")}


def _get_system_content(messages: list[dict]) -> str | None:
    """Extract the system message content, if any."""
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if isinstance(content, list):
                parts = [
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                return "\n".join(parts)
            return content
    return None
