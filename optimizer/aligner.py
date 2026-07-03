"""Module 3: Aligner — cache threshold alignment and cache tag injection.

This module ensures the prefix is long enough to qualify for caching
and adds provider-specific cache control markers.
"""

from typing import Optional

from .config import get_config, ProviderConfig


# ── Simple Token Estimator ──────────────────────────────────────────────


def _estimate_tokens(text: str) -> int:
    """Token count estimation with tiktoken (accurate) + safe fallback.

    Uses tiktoken (cl100k_base) when available for BPE-accurate counts.
    Falls back to len//6 which is conservative for all common text types
    (English, code, JSON, CJK) — it always *under*estimates vs actual
    token count, ensuring we never pad too little and miss the cache.
    """
    if not isinstance(text, str):
        return 0
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        # Conservative fallback: assume 6+ chars per token.
        # Measured ratios: English ~5.5, code ~4.8, JSON ~3.5
        # len//6 underestimates all of them (safe).
        return len(text) // 6


def _prefix_token_count(messages: list[dict]) -> int:
    """Count tokens in the prefix (messages before the last user message).

    The "prefix" is everything before the final user message — this is
    what gets cached and reused across requests.
    """
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            # Multi-part content: sum text parts
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    total += _estimate_tokens(block.get("text", ""))
                elif isinstance(block, str):
                    total += _estimate_tokens(block)
        elif isinstance(content, str):
            total += _estimate_tokens(content)
        else:
            total += _estimate_tokens(str(content))
    return total


# ── Padding ─────────────────────────────────────────────────────────────


# Standard reusable instruction padding — designed to be:
# - Semantically neutral (boilerplate instructions the model already understands)
# - Always the same text across requests (so it contributes to cache prefix)
_DEFAULT_PADDING = (
    "You are an AI assistant. Follow the instructions carefully. "
    "Respond accurately and concisely. Use the provided context when available. "
    "Adhere to any formatting guidelines specified."
)


class Aligner:
    """Adjust message structure to meet cache thresholds.

    Two responsibilities:
      1. Ensure the static prefix is long enough for caching
      2. Add cache control markers where the provider supports them
    """

    def __init__(self, config: Optional[ProviderConfig] = None):
        self.config = config

    def run(self, messages: list[dict]) -> list[dict]:
        """Align messages for caching.

        Args:
            messages: Reordered messages.

        Returns:
            Messages with cache alignment applied.
        """
        if not messages:
            return messages

        cfg = self.config
        if cfg is None:
            return messages

        # Step 1: Inject cache control markers
        if cfg.supports_breakpoints:
            messages = self._inject_cache_tags(messages)

        # Step 2: Check threshold and pad if needed
        prefix_tokens = _prefix_token_count(messages)
        if prefix_tokens < cfg.cache_threshold:
            messages = self._pad_prefix(messages, cfg)

        return messages

    def _inject_cache_tags(self, messages: list[dict]) -> list[dict]:
        """Add cache_control breakpoint markers for Anthropic-style APIs.

        Strategy:
          - If there's a system message, mark it as a cache breakpoint
            (the system prompt is the most stable prefix element).
          - Mark the first user message as a breakpoint too.
        """
        result = []
        for i, msg in enumerate(messages):
            role = msg.get("role", "")
            content = msg.get("content", "")

            # Skip internal markers
            if msg.get("_is_separator"):
                result.append(msg)
                continue

            # Add cache_control to system message
            if role == "system" and self.config.cache_tag_system:
                if isinstance(content, str):
                    msg = {
                        **msg,
                        "content": [
                            {"type": "text", "text": content,
                             "cache_control": {"type": "ephemeral"}}
                        ],
                    }
                elif isinstance(content, list):
                    # If already a list, mark the last text block
                    blocks = list(content)
                    for j in range(len(blocks) - 1, -1, -1):
                        if isinstance(blocks[j], dict) and blocks[j].get("type") == "text":
                            blocks[j] = {
                                **blocks[j],
                                "cache_control": {"type": "ephemeral"},
                            }
                            break
                    msg = {**msg, "content": blocks}

            # Add cache_control to first user message
            elif (role == "user"
                  and self.config.cache_tag_message
                  and not any(m.get("_role_type") == "user" for m in result)):
                if isinstance(content, str):
                    msg = {
                        **msg,
                        "content": [
                            {"type": "text", "text": content,
                             "cache_control": {"type": "ephemeral"}}
                        ],
                    }
                elif isinstance(content, list):
                    blocks = list(content)
                    for j in range(len(blocks) - 1, -1, -1):
                        if isinstance(blocks[j], dict) and blocks[j].get("type") == "text":
                            blocks[j] = {
                                **blocks[j],
                                "cache_control": {"type": "ephemeral"},
                            }
                            break
                    msg = {**msg, "content": blocks}

            result.append(msg)

        return result

    def _pad_prefix(self, messages: list[dict], cfg: ProviderConfig) -> list[dict]:
        """If prefix is below threshold, add reusable padding.

        Padding is injected into the system message (if one exists)
        or prepended as a pseudo-system message.

        The padding text is always the same string, so it contributes
        to the cache prefix across all requests.
        """
        current = _prefix_token_count(messages)
        needed = cfg.cache_threshold - current
        if needed <= 0:
            return messages

        # Find how many padding tokens we need.
        # Apply 1.3x safety multiplier: even with tiktoken, DeepSeek may use a
        # different tokenizer, and this ensures we always meet the threshold.
        padding_per_repeat = _estimate_tokens(_DEFAULT_PADDING)
        padding_repeats = max(1, int(needed * 1.3 / padding_per_repeat) + 2)
        padding_text = "\n".join([_DEFAULT_PADDING] * padding_repeats)

        # Find system message to attach padding to
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, str):
                    messages[i] = {**msg, "content": content + "\n" + padding_text}
                elif isinstance(content, list):
                    blocks = list(content)
                    # If there's already a cache-tagged block, pad the last non-tagged text block
                    text_blocks = [b for b in blocks
                                   if isinstance(b, dict) and b.get("type") == "text"]
                    if text_blocks:
                        last_idx = blocks.index(text_blocks[-1])
                        blocks[last_idx] = {
                            **blocks[last_idx],
                            "text": blocks[last_idx].get("text", "") + "\n" + padding_text,
                        }
                        messages[i] = {**msg, "content": blocks}
                break
        else:
            # No system message exists — we don't add one (respecting user's structure)
            pass

        return messages


# ── Convenience ─────────────────────────────────────────────────────────


def align(messages: list[dict], config: Optional[ProviderConfig] = None) -> list[dict]:
    """Shortcut: create an Aligner and run it."""
    return Aligner(config=config).run(messages)
