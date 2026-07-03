"""Provider configurations for prompt caching.

Each provider has different:
- Cache threshold (minimum prefix tokens to enable caching)
- Cache tag format (how to mark cache breakpoints)
- Message format (roles, content structure)
- API format (openai vs anthropic)

Providers are loaded from ``providers.json`` at import time.
Additional providers can be registered at runtime via ``register_provider()``.
"""

import json
import os
from dataclasses import dataclass, field
from fnmatch import fnmatch
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
        api_format: API protocol format ("openai" or "anthropic").
    """
    name: str
    cache_threshold: int = 1024
    supports_breakpoints: bool = False
    default_model: str = ""
    cache_tag_system: bool = False
    cache_tag_message: bool = False
    separator: str = "\n\n---\n\n"
    api_format: str = "openai"


@dataclass
class ModelInfo:
    """Resolved model information from the registry.

    Attributes:
        provider: Provider name (matches a ProviderConfig key).
        base_url: API base URL for this model.
        api_format: API protocol format ("openai" or "anthropic").
        model_name: Original model name to pass to the API.
    """
    provider: str
    base_url: str | None
    api_format: str
    model_name: str | None = None


# ── Provider Registry (loaded from providers.json) ──────────────────────

_RAW_PROVIDER_REGISTRY: list[dict] = []


def load_providers() -> list[dict]:
    """Load provider registry from the bundled providers.json file.

    Returns a list of raw dict entries (each has name, api_format,
    base_url, model_patterns, and config sub-dict).
    """
    global _RAW_PROVIDER_REGISTRY
    if _RAW_PROVIDER_REGISTRY:
        return _RAW_PROVIDER_REGISTRY

    path = os.path.join(os.path.dirname(__file__), "providers.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _RAW_PROVIDER_REGISTRY = data.get("providers", [])
    except (FileNotFoundError, json.JSONDecodeError):
        _RAW_PROVIDER_REGISTRY = []
    return _RAW_PROVIDER_REGISTRY


def resolve_model(model_name: str) -> ModelInfo | None:
    """Look up a model name in the provider registry.

    Uses glob pattern matching (``fnmatch``) against each provider's
    ``model_patterns``.  Exact matches take priority over wildcard
    matches; the first matching provider wins for each category.

    Args:
        model_name: e.g. "deepseek-v4-flash", "gpt-4o", "llama-3-70b"

    Returns:
        A ``ModelInfo`` with the resolved provider, base URL, and
        API format, or ``None`` if no match is found.
    """
    providers = load_providers()
    if not providers:
        return None

    # Pass 1: exact match
    for entry in providers:
        for pattern in entry.get("model_patterns", []):
            if pattern == model_name:
                return ModelInfo(
                    provider=entry["name"],
                    base_url=entry.get("base_url"),
                    api_format=entry.get("api_format", "openai"),
                    model_name=model_name,
                )

    # Pass 2: wildcard match — pick the most specific (longest) pattern
    candidates: list[tuple[int, dict]] = []  # (pattern_length, entry)
    for entry in providers:
        for pattern in entry.get("model_patterns", []):
            if fnmatch(model_name, pattern):
                candidates.append((len(pattern), entry))
    if candidates:
        # Sort by pattern length descending — longer pattern = more specific
        candidates.sort(key=lambda x: -x[0])
        best = candidates[0][1]
        return ModelInfo(
            provider=best["name"],
            base_url=best.get("base_url"),
            api_format=best.get("api_format", "openai"),
            model_name=model_name,
        )

    return None


def infer_provider_from_url(upstream: str) -> str | None:
    """Infer a provider name from an upstream URL domain.

    Checks known domain patterns; returns ``None`` if unidentified.
    Uses **domain-first** matching so that path components like
    ``/openai/`` don't cause false positives (e.g. ``api.groq.com/openai/v1``).
    """
    u = upstream.lower()

    # Extract domain (netloc) for reliable matching
    from urllib.parse import urlparse
    try:
        domain = urlparse(u).netloc or u  # fallback to full string
    except Exception:
        domain = u

    # ── Match on domain first (most reliable) ──────────────────────
    if "deepseek.com" in domain:
        return "deepseek"
    if "anthropic.com" in domain:
        return "anthropic"
    if "openai.com" in domain or "openai.azure.com" in domain:
        return "openai"
    if "groq.com" in domain:
        return "groq"
    if "together.xyz" in domain:
        return "together"
    if "mistral.ai" in domain:
        return "mistral"
    if "fireworks.ai" in domain:
        return "fireworks"
    if "x.ai" in domain:
        return "xai"
    if "perplexity.ai" in domain:
        return "perplexity"
    if "githubcopilot.com" in domain:
        return "github-copilot"
    if "openrouter.ai" in domain:
        return "openrouter"
    if "googleapis.com" in domain or "generativelanguage.googleapis.com" in domain:
        return "google-gemini"

    # ── Fallback: check the full string for less common patterns ───
    if "azure.com" in u:
        return "openai"
    if "googleapis" in u or "generativelanguage" in u:
        return "google-gemini"

    return None


# ── Built-in Providers (legacy, now loaded from JSON) ───────────────────

PROVIDER_CONFIGS: dict[str, ProviderConfig] = {}


def _build_configs() -> dict[str, ProviderConfig]:
    """Build ``PROVIDER_CONFIGS`` from the JSON registry.

    Called once at module import time.  Merges with any legacy
    entries that may have been registered by user code.
    """
    configs = {}
    for entry in load_providers():
        name = entry["name"]
        cfg_data = entry.get("config", {})
        cfg = ProviderConfig(
            name=name,
            api_format=entry.get("api_format", "openai"),
            **cfg_data,
        )
        configs[name] = cfg
    return configs


# Overridable by user code via register_provider()
PROVIDER_CONFIGS = _build_configs()


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
        **overrides: Any ``ProviderConfig`` field to override.

    Returns:
        The newly created ``ProviderConfig``.
    """
    cfg = ProviderConfig(name=name, **overrides)
    PROVIDER_CONFIGS[name] = cfg
    return cfg
