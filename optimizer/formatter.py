"""Module 4: Formatter — provider-specific output formatting.

Translates the internal message format into the exact structure
expected by each API provider, including cache control markers.
"""

from typing import Optional

from .config import get_config, ProviderConfig


class Formatter:
    """Format optimized messages for a specific provider API.

    The formatter is the final stage before the request is sent.
    It handles:
      - Stripping internal markers (``_role_type``, ``_is_separator``)
      - Structuring content blocks per provider conventions
      - Ensuring cache control parameters are in the right place
    """

    def __init__(self, config: Optional[ProviderConfig] = None):
        self.config = config

    def format(self, messages: list[dict], provider: str = "") -> list[dict]:
        """Format messages for API consumption.

        Args:
            messages: Optimized messages (may contain internal markers).
            provider: Target provider name. Overrides self.config if set.

        Returns:
            Messages ready to be sent to the API (no internal markers).
        """
        cfg = self.config or get_config(provider)

        # Sanitize: remove internal markers
        cleaned = self._strip_internal(messages)

        # Provider-specific formatting
        if cfg.name == "anthropic":
            return self._format_anthropic(cleaned)
        elif cfg.name == "openai":
            return self._format_openai(cleaned)
        else:
            # Unknown provider: just strip internal markers
            return cleaned

    @staticmethod
    def _strip_internal(messages: list[dict]) -> list[dict]:
        """Remove internal markers and separator messages."""
        result = []
        for msg in messages:
            # Skip separator messages entirely
            if msg.get("_is_separator"):
                continue
            # Remove internal tags
            result.append({
                k: v for k, v in msg.items()
                if not k.startswith("_")
            })
        return result

    @staticmethod
    def _format_anthropic(messages: list[dict]) -> list[dict]:
        """Anthropic-specific formatting.

        Anthropic expects:
          - System message as a separate ``system`` parameter or in messages
          - Content can be string or list of content blocks
          - ``cache_control`` is a property on content blocks
        """
        # Already in the right format from Aligner
        return messages

    @staticmethod
    def _format_openai(messages: list[dict]) -> list[dict]:
        """OpenAI-specific formatting.

        OpenAI expects:
          - Messages with 'role' and 'content' (content is always string)
          - cache_control is not per-block; threshold is automatic
        """
        result = []
        for msg in messages:
            content = msg.get("content", "")

            # OpenAI uses string content (not content blocks)
            if isinstance(content, list):
                # Flatten content blocks to string
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                content = "\n".join(text_parts)

            result.append({
                "role": msg["role"],
                "content": content,
            })

        return result


# ── Convenience ─────────────────────────────────────────────────────────


def format_messages(
    messages: list[dict],
    provider: str = "",
    config: Optional[ProviderConfig] = None,
) -> list[dict]:
    """Shortcut: create a Formatter and run it."""
    return Formatter(config=config).format(messages, provider=provider)
