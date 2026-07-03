"""Prompt Optimizer (popt) — maximize LLM API cache hit rates.

A lightweight structural optimizer that reorganizes prompt messages to
maximize prefix stability for Prompt Caching, without changing semantics.

Quick start:
    >>> from popt import optimize
    >>> messages = [
    ...     {"role": "system", "content": "You are a helpful assistant."},
    ...     {"role": "user", "content": "Hello!"},
    ... ]
    >>> optimized = optimize(messages, provider="anthropic")
    >>> # Use directly with the API:
    >>> client.messages.create(messages=optimized, ...)
"""

from typing import Optional

from .config import get_config, ProviderConfig, register_provider
from .normalizer import Normalizer
from .reorderer import Reorderer
from .aligner import Aligner
from .formatter import Formatter
from .diagnoser import (
    diagnose,
    report,
    CacheMetrics,
    CacheEntry,
    CacheReport,
    MissAnalyzer,
    HitRatioCalculator,
    PrefixShape,
    ShapeDiff,
    PrefixDiffReport,
    PrefixDiffer,
    capture_shape,
    compare_shapes,
)
from .safety import SafetyCheck


def optimize(
    messages: list[dict],
    provider: str = "anthropic",
    config: Optional[ProviderConfig] = None,
) -> list[dict]:
    """Optimize message structure for maximum cache hit rate.

    This is the main entry point. It runs the full pipeline:
    Normalizer → Reorderer → Aligner → Formatter.

    Args:
        messages: List of message dicts with 'role' and 'content'.
                 Format follows the standard Anthropic/OpenAI convention.
        provider: Target API provider ("anthropic", "openai", or custom).
                 Defaults to "anthropic".
        config: Optional ProviderConfig. If not provided, inferred from
               ``provider`` string.

    Returns:
        Optimized messages, structurally reordered and formatted for
        the target provider. The semantic content is unchanged.

    Raises:
        ValueError: If ``messages`` is empty or malformed.

    Examples:
        >>> from popt import optimize
        >>> msgs = [{"role": "system", "content": "Be concise."},
        ...         {"role": "user", "content": "Tell me a joke."}]
        >>> result = optimize(msgs, provider="anthropic")
        >>> result[0]["role"]
        'system'
    """
    if not messages or not isinstance(messages, list):
        raise ValueError("messages must be a non-empty list of dicts")

    cfg = config or get_config(provider)

    # ── Pipeline ────────────────────────────────────────────────────
    # 1. Normalize: clean whitespace, canonicalize JSON, tag types
    normalized = Normalizer.run(messages)

    # 2. Reorder: deterministic ordering for prefix stability
    reordered = Reorderer(cfg).run(normalized)

    # 3. Align: meet cache thresholds, add breakpoints
    aligned = Aligner(cfg).run(reordered)

    # 4. Format: provider-specific output
    formatted = Formatter(cfg).format(aligned, provider=cfg.name)

    return formatted


def _estimate_chars(text: str) -> int:
    """Rough char-to-token estimate (1 token ≈ 4 chars)."""
    if not isinstance(text, str):
        return 0
    return len(text)


def _count_tokens_est(messages: list[dict]) -> int:
    """Count chars for rough token estimation."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    total += _estimate_chars(block.get("text", ""))
        elif isinstance(content, str):
            total += _estimate_chars(content)
        else:
            total += _estimate_chars(str(content))
    return total // 4


def preview(messages: list[dict], provider: str = "anthropic") -> dict:
    """Show what the optimizer will change, without sending to any API.

    Returns a dict with 'before' (counts, structure) and 'after' (same)
    for comparison.

    Args:
        messages: Original messages.
        provider: Target provider.

    Returns:
        Comparison dict with token estimates and structural diffs.
    """
    cfg = get_config(provider)

    # Count original structure
    original_roles = [m.get("role", "unknown") for m in messages]

    # Compute optimized
    optimized = optimize(messages, provider=provider, config=cfg)

    optimized_roles = [m.get("role", "unknown") for m in optimized]
    sep_count = sum(1 for m in optimized if m.get("_is_separator"))

    return {
        "provider": provider,
        "message_count": {
            "before": len(messages),
            "after": len(optimized),
            "separators_added": sep_count,
        },
        "role_order_before": original_roles,
        "role_order_after": optimized_roles,
        "estimated_tokens": {
            "before": _count_tokens_est(messages),
            "after": _count_tokens_est(optimized),
        },
        "cache_threshold": cfg.cache_threshold,
        "meets_threshold": (_count_tokens_est(optimized)
                           >= cfg.cache_threshold),
    }


# ── Public API surface ─────────────────────────────────────────────────
__all__ = [
    "optimize",
    "preview",
    "diagnose",
    "report",
    "register_provider",
    "CacheMetrics",
    "CacheEntry",
    "CacheReport",
    "MissAnalyzer",
    "HitRatioCalculator",
    "PrefixShape",
    "ShapeDiff",
    "PrefixDiffReport",
    "PrefixDiffer",
    "capture_shape",
    "compare_shapes",
    "SafetyCheck",
]
