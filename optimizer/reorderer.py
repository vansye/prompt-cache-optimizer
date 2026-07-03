"""Module 2: Reorderer — deterministic ordering of messages.

The goal: ensure the same set of messages always produces the same order,
so the prefix remains stable across requests.

Rules:
  1. System messages always come first (preserving original system order)
  2. Within same-role groups, order is deterministic by content hash
  3. Fixed separators are inserted between role groups
"""

import hashlib
from typing import Optional

from .config import get_config, ProviderConfig
from .safety import SafetyCheck


# ── Content Hash ────────────────────────────────────────────────────────


def _content_hash(msg: dict) -> str:
    """Stable hash of message content for deterministic ordering.

    Uses only the 'content' field so that the same content always maps
    to the same position, regardless of other metadata.
    """
    content = msg.get("content", "")
    if isinstance(content, list):
        # For multi-part content (e.g. Anthropic content blocks),
        # hash the JSON representation
        import json
        content = json.dumps(content, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(str(content).encode("utf-8")).hexdigest()


# ── Ordering Strategy ──────────────────────────────────────────────────


class Reorderer:
    """Deterministic message reordering.

    The reordering is designed to be:
    - Stable: same input → same output every time
    - Safe: preserves the semantic structure of the conversation
    - Minimal: only moves messages when it improves prefix stability
    """

    def __init__(self, config: Optional[ProviderConfig] = None):
        self.config = config

    def run(self, messages: list[dict]) -> list[dict]:
        """Reorder messages deterministically.

        Args:
            messages: Normalized messages (with '_role_type' tags).

        Returns:
            Reordered messages with separators inserted.
        """
        if not messages:
            return messages

        ordered = self._reorder(messages)
        ordered = self._insert_separators(ordered)

        # Safety check: verify no semantic loss
        if not SafetyCheck.verify_reorder(messages, ordered):
            return messages  # Fall back to original

        return ordered

    def _has_tool_conversation(self, messages: list[dict]) -> bool:
        """Detect if the conversation contains tool calls.

        Tool conversations have a specific interleaving pattern that
        must be preserved: assistant(tool_use) → tool → assistant → ...
        """
        for msg in messages:
            if msg.get("_role_type") == "tool":
                return True
            # Check for tool_use in assistant content blocks
            content = msg.get("content", "")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        return True
            # Check for tool_use_id in tool results
            if msg.get("tool_use_id"):
                return True
        return False

    def _reorder(self, messages: list[dict]) -> list[dict]:
        """Core reordering logic.

        Two strategies, chosen automatically:

        **Simple conversations** (no tool calls):
          1. System messages to front.
          2. Group remaining by role.
          3. Sort within each group by content hash.
          4. Concatenate: system + user + assistant + tool.

        **Tool conversations** (has tool_use/tool messages):
          1. System messages to front.
          2. Preserve original relative order of everything else.
          3. Tool call chains are fragile — reordering breaks them.
        """
        # Separate system messages (preserve original order)
        system_msgs = [m for m in messages if m.get("_role_type") == "system"]
        other_msgs = [m for m in messages if m.get("_role_type") != "system"]

        is_tool_conversation = self._has_tool_conversation(messages)

        if is_tool_conversation:
            # Conservative: preserve original order, just move system to front
            result = list(system_msgs) + other_msgs
        else:
            # Safe to regroup: no fragile tool call chains
            from collections import OrderedDict
            groups: dict[str, list[dict]] = OrderedDict()
            for msg in other_msgs:
                role = msg.get("_role_type", "other")
                if role not in groups:
                    groups[role] = []
                groups[role].append(msg)

            # Sort within each group by content hash
            for role in groups:
                groups[role].sort(key=_content_hash)

            # Rebuild: system first, then groups in priority order
            result = list(system_msgs)
            for role in ["user", "assistant", "tool", "other"]:
                if role in groups:
                    result.extend(groups[role])

        return result

    def _insert_separators(self, messages: list[dict]) -> list[dict]:
        """Insert fixed separators between role-group boundaries.

        This ensures that the token pattern at group boundaries is always
        the same, which helps prefix matching.
        """
        if not messages:
            return messages

        sep = (self.config.separator if self.config
               else "\n\n---\n\n")

        result = []
        prev_role = None
        for msg in messages:
            current_role = msg.get("_role_type", "other")
            if prev_role is not None and current_role != prev_role:
                # Insert separator between role groups
                result.append({
                    "role": "assistant",
                    "content": sep,
                    "_is_separator": True,
                    "_role_type": "separator",
                })
            result.append(msg)
            prev_role = current_role

        return result


# ── Convenience ─────────────────────────────────────────────────────────


def reorder(messages: list[dict], config: Optional[ProviderConfig] = None) -> list[dict]:
    """Shortcut: create a Reorderer and run it."""
    return Reorderer(config=config).run(messages)
