"""Provider configurations for prompt caching.

Each provider has different:
- Cache threshold (minimum prefix tokens to enable caching)
- Cache tag format (how to mark cache breakpoints)
- Message format (roles, content structure)

Adding a new provider = adding an entry here.
No core code changes needed.
"""

from dataclasses import dataclass, field
from typing import Optional

# ── Provider Configuration ──────────────────────────────────────────────


@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider's caching behavior.

    Attributes:
        name: Provider identifier (e.g. "anthropic", "openai").
        cache_threshold: Minimum tokens in the prefix for caching to activate.
        supports_breakpoints: Whether the provider allows manual cache_control markers.
        default_model: Fallback model name for token estimation.
        cache_tag_system: Whether to add cache tag to system message.
        cache_tag_message: Whether the provider adds cache tags to non-system messages.
        separator: Default segment separator.
    """
    name: str
    cache_threshold: int = 1024
    supports_breakpoints: bool = False
    default_model: str = ""
    cache_tag_system: bool = False
    cache_tag_message: bool = False
    separator: str = "\n\n---\n\n"


# ── Built-in Providers ─────────────────────────────────────────────────

PROVIDER_CONFIGS: dict[str, ProviderConfig] = {
    "anthropic": ProviderConfig(
        name="anthropic",
        cache_threshold=1024,
        supports_breakpoints=True,
        default_model="claude-sonnet-5-20250601",
        cache_tag_system=True,
        cache_tag_message=True,
        separator="\n\n---\n\n",
    ),
    "openai": ProviderConfig(
        name="openai",
        cache_threshold=1025,
        supports_breakpoints=False,
        default_model="gpt-4o",
        cache_tag_system=False,
        cache_tag_message=False,
        separator="\n\n---\n\n",
    ),
    "deepseek": ProviderConfig(
        name="deepseek",
        cache_threshold=128,
        supports_breakpoints=False,
        default_model="deepseek-v4-flash",
        cache_tag_system=False,
        cache_tag_message=False,
        separator="\n\n---\n\n",
    ),
}

# ── Message Type Helpers ───────────────────────────────────────────────

# The standard role values we recognize
ROLES = {"system", "user", "assistant", "tool"}

# Messages that should come first (in order)
PRIORITY_ROLES = ["system", "user", "assistant", "tool"]


def get_config(provider: str) -> ProviderConfig:
    """Look up a provider config by name.

    Falls back to a sensible default if the provider is not known.
    """
    if provider in PROVIDER_CONFIGS:
        return PROVIDER_CONFIGS[provider]
    # Unknown provider: use sensible defaults
    return ProviderConfig(
        name=provider,
        cache_threshold=1024,
        default_model="unknown",
    )


def register_provider(name: str, **overrides) -> ProviderConfig:
    """Register a custom provider at runtime.

    Args:
        name: Provider identifier.
        **overrides: Any ProviderConfig field to override.

    Returns:
        The newly created ProviderConfig.
    """
    cfg = ProviderConfig(name=name, **overrides)
    PROVIDER_CONFIGS[name] = cfg
    return cfg
