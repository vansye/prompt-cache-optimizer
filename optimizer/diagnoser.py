"""Module 5: Diagnoser — cache hit/miss analysis.

This module is a post-hoc analysis layer. It doesn't modify requests;
it analyzes API responses to tell you how well caching is working
and where misses are happening.

It provides two levels of diagnosis:
  1. Per-response parsing (CacheMetrics)
  2. Cross-request prefix diff (PrefixDiffer) — *why* a cache miss happened
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Optional


# ── Cache Metrics ────────────────────────────────────────────────────────────


@dataclass
class CacheMetrics:
    """Cache-related metrics extracted from an API response.

    These field names follow the Anthropic API convention.
    OpenAI uses slightly different names; the parser normalizes them.
    """
    input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class CacheReport:
    """Aggregated cache performance report across multiple requests."""
    total_requests: int = 0
    total_input_tokens: int = 0
    total_cache_creation: int = 0
    total_cache_read: int = 0
    total_output_tokens: int = 0

    # Derived
    hit_ratio: float = 0.0
    savings_ratio: float = 0.0
    estimated_cost_saved: float = 0.0

    entries: list["CacheEntry"] = field(default_factory=list)


@dataclass
class CacheEntry:
    """A single request's cache data."""
    request_index: int
    metrics: CacheMetrics
    is_hit: bool
    notes: str = ""


# ── Prefix Shape (cross-request snapshot) ──────────────────────────────────


@dataclass
class PrefixShape:
    """Hash snapshot of a request prefix, used to detect *what* changed.

    Two snapshots taken at different turns can be compared to explain
    why a prefix-cache miss happened — just like DeepSeek-Reasonix's
    CaptureShape / CompareShape.

    Each field is an 8-char hex hash of a prefix component.
    """
    system_hash: str = ""
    role_sequence_hash: str = ""
    content_prefix_hash: str = ""
    full_prefix_hash: str = ""


def _short_hash(data: object) -> str:
    """Deterministic 8-char hex hash of any JSON-serializable object."""
    raw = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:8]


def capture_shape(messages: list[dict]) -> PrefixShape:
    """Snapshot the prefix components that affect cache reuse.

    Args:
        messages: Request messages (raw or optimized).

    Returns:
        PrefixShape with hashes of system prompt, role sequence,
        combined content prefix, and full prefix.
    """
    # Extract system content (hash only if a system message actually exists)
    system_text = ""
    has_system = False
    for m in messages:
        if m.get("role") == "system" or m.get("_role_type") == "system":
            content = m.get("content", "")
            if isinstance(content, list):
                texts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
                content = "\n".join(texts)
            if isinstance(content, str):
                system_text = content
            has_system = True
            break

    # Role sequence (excluding separators)
    role_seq = [m.get("role") or m.get("_role_type", "?")
                for m in messages if not m.get("_is_separator")]

    # Content prefix — first ~2000 chars of concatenated content
    content_parts = []
    for m in messages:
        c = m.get("content", "")
        if isinstance(c, list):
            texts = [b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text"]
            c = "\n".join(texts)
        if isinstance(c, str):
            content_parts.append(c)
    full_content = "\n".join(content_parts)[:2000]

    return PrefixShape(
        system_hash=_short_hash(system_text) if has_system else "",
        role_sequence_hash=_short_hash(role_seq),
        content_prefix_hash=_short_hash(full_content),
        full_prefix_hash=_short_hash(messages),
    )


@dataclass
class ShapeDiff:
    """Result of comparing two PrefixShapes."""
    changed: bool
    reasons: list[str]
    # Sorted from most-impactful to least
    # e.g. ["system", "role_sequence", "content_prefix"]


def compare_shapes(a: PrefixShape, b: PrefixShape) -> ShapeDiff:
    """Compare two prefix snapshots and explain what changed.

    Args:
        a: Shape from the earlier request (the one that created the cache).
        b: Shape from the current request (the one that missed).

    Returns:
        ShapeDiff with ``changed`` boolean and ``reasons`` list.
    """
    reasons = []
    if a.system_hash and a.system_hash != b.system_hash:
        reasons.append("system")
    if a.role_sequence_hash and a.role_sequence_hash != b.role_sequence_hash:
        reasons.append("role_sequence")
    if a.content_prefix_hash and a.content_prefix_hash != b.content_prefix_hash:
        reasons.append("content_prefix")
    if a.full_prefix_hash and a.full_prefix_hash != b.full_prefix_hash:
        reasons.append("messages_structure")

    return ShapeDiff(
        changed=len(reasons) > 0,
        reasons=reasons,
    )


# ── Prefix Diffs (token-precise) ─────────────────────────────────────────────


def _serialize_message(msg: dict) -> str:
    """Serialize a single message to a canonical string for diffing."""
    role = msg.get("role") or msg.get("_role_type", "?")
    content = msg.get("content", "")
    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    texts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    texts.append(f"[tool_use:{block.get('name','')}]")
                else:
                    texts.append(json.dumps(block, sort_keys=True))
            else:
                texts.append(str(block))
        content = "\n".join(texts)
    elif not isinstance(content, str):
        content = str(content)

    # Include key metadata fields that affect the wire format
    extras = {}
    for key in ("tool_use_id", "name", "_is_separator"):
        if key in msg:
            extras[key] = msg[key]

    parts = [f"role:{role}", f"content:{content}"]
    if extras:
        parts.append(f"meta:{json.dumps(extras, sort_keys=True)}")
    return "\n".join(parts)


def _estimate_tokens(text: str) -> int:
    """Token count for diagnostics (same estimator as aligner)."""
    if not isinstance(text, str) or not text:
        return 0
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        return len(text) // 6


def _serialize_prefix(messages: list[dict], max_tokens: int = 0) -> tuple[str, int]:
    """Serialize messages into a single string, tracking approximate token position.

    Returns (serialized_text, total_estimated_tokens).
    """
    parts = []
    total_tokens = 0
    for msg in messages:
        s = _serialize_message(msg)
        parts.append(s)
        total_tokens += _estimate_tokens(s)

        if max_tokens > 0 and total_tokens >= max_tokens:
            break

    return "\n---\n".join(parts), total_tokens


def _find_first_byte_diff(a: str, b: str) -> Optional[int]:
    """Find the byte index of the first difference between two strings.

    Returns None if the strings are identical.
    """
    if a == b:
        return None
    for i, (ca, cb) in enumerate(zip(a, b)):
        if ca != cb:
            return i
    # One is a prefix of the other
    return min(len(a), len(b))


@dataclass
class PrefixDiffReport:
    """Structured report of the first difference between two message sequences.

    Attributes:
        position: Approximate token position of the first difference.
        block: Which 128-token cache block the difference falls in (0-indexed).
        severity: "CRITICAL" if in block 0 or 1, "MAJOR" if in early blocks,
                  "MINOR" if deep in prefix.
        msg_index: Index of the message containing the diff (0-based).
        msg_role: Role of that message.
        field: Which part differs ("role", "content", "metadata").
        snippet_a: Excerpt from request A around the diff.
        snippet_b: Excerpt from request B around the diff.
        suggestion: Human-readable action item.
    """
    position: int = 0
    block: int = 0
    severity: str = "INFO"
    msg_index: int = 0
    msg_role: str = ""
    field: str = ""
    snippet_a: str = ""
    snippet_b: str = ""
    suggestion: str = ""


class PrefixDiffer:
    """Token-precise diff between two message sequences.

    Compares at three levels:
      1. Message structure (count, roles) — detects reordering
      2. Per-message content — detects changed fields
      3. Byte-level content — finds the exact first differing token
    """

    @staticmethod
    def diff(
        messages_a: list[dict],
        messages_b: list[dict],
    ) -> Optional[PrefixDiffReport]:
        """Compare two message sequences and pinpoint the first cache-affecting difference.

        Args:
            messages_a: Earlier request (the one that established the cache).
            messages_b: Later request (the one that may have missed).

        Returns:
            PrefixDiffReport if a difference is found, or None if identical.
        """
        # --- Level 1: Message count ---
        if len(messages_a) != len(messages_b):
            return PrefixDiffReport(
                position=0,
                block=0,
                severity="CRITICAL",
                msg_index=0,
                msg_role="",
                field="message_count",
                snippet_a=f"{len(messages_a)} messages",
                snippet_b=f"{len(messages_b)} messages",
                suggestion="Message count differs. Check for added/removed turns.",
            )

        # --- Level 2: Per-message comparison ---
        for idx, (ma, mb) in enumerate(zip(messages_a, messages_b)):
            result = PrefixDiffer._compare_messages(ma, mb, idx)
            if result:
                return result

        return None

    @staticmethod
    def _compare_messages(
        ma: dict, mb: dict, idx: int,
    ) -> Optional[PrefixDiffReport]:
        """Compare two individual messages."""
        # Role
        role_a = ma.get("role") or ma.get("_role_type", "?")
        role_b = mb.get("role") or mb.get("_role_type", "?")
        if role_a != role_b:
            pos = _estimate_tokens(_serialize_message(ma)) // 2  # rough position
            block = pos // 128
            return PrefixDiffReport(
                position=pos,
                block=block,
                severity="CRITICAL" if block < 2 else "MAJOR",
                msg_index=idx,
                msg_role=role_a,
                field="role",
                snippet_a=role_a,
                snippet_b=role_b,
                suggestion=f"Message {idx} role changed from '{role_a}' to '{role_b}'. "
                           "Keep role assignments stable across requests.",
            )

        # Content
        content_a = _serialize_message(ma)
        content_b = _serialize_message(mb)
        diff_pos = _find_first_byte_diff(content_a, content_b)
        if diff_pos is not None:
            return PrefixDiffer._build_content_diff(
                ma, mb, content_a, content_b, diff_pos, idx,
            )

        return None

    @staticmethod
    def _build_content_diff(
        ma: dict,
        mb: dict,
        content_a: str,
        content_b: str,
        diff_pos: int,
        idx: int,
    ) -> PrefixDiffReport:
        """Build a diff report for content-level differences."""
        role = ma.get("role") or ma.get("_role_type", "?")

        # Estimate token position
        prefix_so_far = content_a[:max(0, diff_pos - 100)]
        pos = _estimate_tokens(prefix_so_far)
        block = pos // 128

        # Snippets: 80 chars around the diff
        start = max(0, diff_pos - 40)
        end = min(len(content_a), diff_pos + 40)
        snippet_a = content_a[start:end]
        snippet_b = content_b[start:end]

        # Detect what kind of field differs
        field = "content"
        if "\"role\":" in content_a[max(0, diff_pos-20):diff_pos+20]:
            field = "role"
        elif "\"name\":" in content_a[max(0, diff_pos-20):diff_pos+20]:
            field = "tool_name"
        elif "tool_use_id" in content_a[max(0, diff_pos-20):diff_pos+20]:
            field = "tool_call_id"

        # Build suggestion
        if field == "content":
            suggestion = (
                f"Message {idx} ({role}) content differs at ~token {pos} "
                f"(block {block}). "
                "This causes a prefix-cache miss for all subsequent blocks. "
                "If the difference is a variable field (date, user name), "
                "consider moving it to the end of the message."
            )
        else:
            suggestion = (
                f"Message {idx} {field} changed. "
                "Keep metadata fields stable for cache reuse."
            )

        severity = "CRITICAL" if block < 2 else "MAJOR" if block < 4 else "MINOR"

        return PrefixDiffReport(
            position=pos,
            block=block,
            severity=severity,
            msg_index=idx,
            msg_role=role,
            field=field,
            snippet_a=snippet_a,
            snippet_b=snippet_b,
            suggestion=suggestion,
        )


# ── Metrics Parser ──────────────────────────────────────────────────────────


def parse_usage(response: dict) -> CacheMetrics:
    # ... (unchanged, see above)
    """Extract cache metrics from an API response.

    Handles Anthropic, OpenAI, and DeepSeek response formats.
    """
    usage = response.get("usage", {})

    # Anthropic format
    if "cache_read_input_tokens" in usage:
        return CacheMetrics(
            input_tokens=usage.get("input_tokens", 0),
            cache_creation_input_tokens=usage.get(
                "cache_creation_input_tokens", 0),
            cache_read_input_tokens=usage.get("cache_read_input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )

    # DeepSeek format: prompt_cache_hit_tokens at usage top level
    prompt_cache_hit = usage.get("prompt_cache_hit_tokens", 0)
    prompt_cache_miss = usage.get("prompt_cache_miss_tokens", 0)
    prompt_tokens = usage.get("prompt_tokens", 0)

    # OpenAI format: prompt_tokens_details.cached_tokens
    prompt_details = usage.get("prompt_tokens_details", {})
    openai_cached = prompt_details.get("cached_tokens", 0) if prompt_details else 0

    cached_tokens = prompt_cache_hit or openai_cached or 0
    return CacheMetrics(
        input_tokens=prompt_tokens,
        cache_creation_input_tokens=prompt_cache_miss or cached_tokens,
        cache_read_input_tokens=cached_tokens,
        output_tokens=usage.get("completion_tokens", 0),
    )


# ── Miss Analyzer ───────────────────────────────────────────────────────────


class MissAnalyzer:
    """Analyze why a cache miss occurred.

    A cache miss means ``cache_read_input_tokens == 0``.
    This can happen because:
      1. The prefix is too short (below threshold)
      2. The prefix changed between requests
      3. This is the first request (no cache to read from)
    """

    @staticmethod
    def analyze(
        current_metrics: CacheMetrics,
        previous_metrics: Optional[CacheMetrics] = None,
        prefix_length: int = 0,
        threshold: int = 1024,
        shape_diff: Optional[ShapeDiff] = None,
        prefix_diff: Optional[PrefixDiffReport] = None,
    ) -> str:
        """Return a human-readable analysis of the cache miss.

        Args:
            current_metrics: Metrics from the current request.
            previous_metrics: Metrics from a previous request for comparison.
            prefix_length: Estimated token length of the prefix.
            threshold: Provider's minimum prefix length for caching.
            shape_diff: Optional shape comparison (from compare_shapes).
            prefix_diff: Optional token-precise diff (from PrefixDiffer.diff).

        Returns:
            Descriptive string explaining the cache outcome.
        """
        if current_metrics.cache_read_input_tokens > 0:
            parts = [
                f"Cache HIT: read {current_metrics.cache_read_input_tokens} "
                f"tokens from cache "
                f"({current_metrics.cache_read_input_tokens
                   / max(current_metrics.input_tokens, 1) * 100:.0f}% of input)"
            ]
            if prefix_diff:
                block_count = current_metrics.cache_read_input_tokens // 128
                parts.append(f"({block_count} blocks @ 128t)")
            return " | ".join(parts)

        reasons = []

        if previous_metrics is None:
            reasons.append("First request (no cache established yet)")

        if prefix_length < threshold:
            reasons.append(
                f"Prefix too short: ~{prefix_length}t (needs ≥{threshold}t)"
            )

        # Shape diff explanation
        if shape_diff and shape_diff.changed:
            changed_parts = ", ".join(shape_diff.reasons)
            reasons.append(f"Prefix changed: {changed_parts}")

        # Token-precise diff
        if prefix_diff:
            reasons.append(
                f"First difference at msg[{prefix_diff.msg_index}] "
                f"({prefix_diff.msg_role}), field={prefix_diff.field}, "
                f"~token {prefix_diff.position} (cache block {prefix_diff.block})"
            )
            if prefix_diff.suggestion not in [s for s in reasons if prefix_diff.suggestion in s]:
                reasons.append(f"Suggestion: {prefix_diff.suggestion}")

        if current_metrics.cache_creation_input_tokens > 0:
            reasons.append(
                f"Cache created ({current_metrics.cache_creation_input_tokens}t)"
            )
        elif not reasons:
            reasons.append("No cache was created (prefix too short or caching disabled)")

        return "Cache MISS. " + "; ".join(reasons)


# ── Hit Ratio Calculator ────────────────────────────────────────────────────


class HitRatioCalculator:
    """Calculate cache hit ratios across multiple requests."""

    @staticmethod
    def calculate(entries: list[CacheEntry]) -> CacheReport:
        """Aggregate multiple entries into a report.

        Args:
            entries: List of cache entries from sequential requests.

        Returns:
            Aggregated CacheReport with derived metrics.
        """
        total_input = sum(e.metrics.input_tokens for e in entries)
        total_creation = sum(e.metrics.cache_creation_input_tokens for e in entries)
        total_read = sum(e.metrics.cache_read_input_tokens for e in entries)
        total_output = sum(e.metrics.output_tokens for e in entries)
        total_requests = len(entries)

        # Hit ratio: cache_read / total_input
        hit_ratio = total_read / max(total_input, 1)

        # Savings ratio: cache_read / (creation + read)
        savings_ratio = total_read / max(total_creation + total_read, 1)

        # Cost estimate (approximate, at Anthropic Sonnet pricing):
        # Input: $3.00/Mtokens, Cache read: $0.30/Mtokens
        cached_input_cost = (total_input - total_read) * 3.0 / 1_000_000
        read_cost = total_read * 0.30 / 1_000_000
        cost_without_caching = total_input * 3.0 / 1_000_000
        estimated_cost_saved = cost_without_caching - (cached_input_cost + read_cost)

        return CacheReport(
            total_requests=total_requests,
            total_input_tokens=total_input,
            total_cache_creation=total_creation,
            total_cache_read=total_read,
            total_output_tokens=total_output,
            hit_ratio=hit_ratio,
            savings_ratio=savings_ratio,
            estimated_cost_saved=estimated_cost_saved,
            entries=entries[:50],  # Keep last 50 entries
        )


# ── Convenience ─────────────────────────────────────────────────────────────


def diagnose(response: dict, **context) -> CacheMetrics:
    """Quick shortcut: parse metrics from a single response."""
    return parse_usage(response)


def report(entries: list[CacheEntry]) -> CacheReport:
    """Quick shortcut: compute aggregated report."""
    return HitRatioCalculator.calculate(entries)


__all__ = [
    "CacheMetrics",
    "CacheEntry",
    "CacheReport",
    "PrefixShape",
    "ShapeDiff",
    "PrefixDiffReport",
    "PrefixDiffer",
    "capture_shape",
    "compare_shapes",
    "parse_usage",
    "diagnose",
    "report",
    "MissAnalyzer",
    "HitRatioCalculator",
]
