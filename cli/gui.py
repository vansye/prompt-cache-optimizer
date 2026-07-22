"""Web GUI + multi-session proxy server for popt.

Supports multiple concurrent sessions (each = one API key + model combo).
Auto-routes requests based on the API key in the incoming request.

Usage:
    popt gui --port 6123
    # Browser:  http://localhost:6123
    # AI tool:  ANTHROPIC_BASE_URL=http://localhost:6123
"""

import json
import os
import threading
import time
import logging
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from cli.proxy import (
    ProxyHandler, ThreadedProxyServer,
    _forward_headers, _detect_provider,
)
from optimizer import optimize
from optimizer.config import resolve_model, get_config, load_providers
from optimizer.diagnoser import parse_usage, CacheMetrics

logger = logging.getLogger("popt-gui")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


# ── Tool fingerprinting & API key extraction ────────────────────────────

def _extract_api_key(headers: dict) -> str | None:
    """Extract API key from request headers.

    Tries x-api-key (Anthropic style) first, then Authorization: Bearer.
    """
    key = headers.get("x-api-key") or headers.get("X-Api-Key")
    if key:
        return key
    auth = headers.get("Authorization") or headers.get("authorization")
    if auth and auth.startswith("Bearer "):
        return auth[7:]
    return None


def _auto_create_session(headers: dict, path: str, tool: str,
                         body: dict | None = None) -> Session | None:
    """Auto-create a session when a new (tool, api_key) combo arrives.

    Uses the model from the request body if available, else falls back to
    path-based provider detection. Persists the new session immediately.

    Returns the created session, or None if no API key was found.
    """
    api_key = _extract_api_key(headers)
    if not api_key:
        return None

    # Try to read model from request body
    model = ""
    if body and isinstance(body, dict):
        model = body.get("model", "")

    # Fallback: use path to pick a reasonable default model
    if not model:
        if "/v1/messages" in path:
            model = "claude-sonnet-5-20250601"  # Anthropic default
        else:
            model = "deepseek-chat"  # OpenAI-format default (most common here)

    # Resolve model → provider/upstream/api_format
    model_info = resolve_model(model)
    if not model_info:
        logger.warning(f"Auto-create: cannot resolve model '{model}'")
        return None

    # Path-aware format selection: providers like DeepSeek expose both
    # an Anthropic-format endpoint (/anthropic, /v1/messages) and an
    # OpenAI-format endpoint (root, /v1/chat/completions).  resolve_model
    # returns the Anthropic base_url by default; switch to alt_base_url
    # when the request actually arrived on an OpenAI-format path.
    # Match loosely: hermes sends /chat/completions (no /v1 prefix).
    upstream = model_info.base_url or ""
    api_format = model_info.api_format
    is_openai_path = "chat/completions" in path
    if is_openai_path and api_format == "anthropic":
        # Look up alt_base_url from the raw provider registry
        for entry in load_providers():
            if entry.get("name") == model_info.provider:
                alt = entry.get("alt_base_url")
                if alt:
                    upstream = alt
                    api_format = "openai"
                break

    name = f"auto-{tool}-{api_key[-4:]}"
    session = _registry.add(
        name=name,
        model=model,
        api_key=api_key,
        provider=model_info.provider,
        upstream=upstream,
        api_format=api_format,
        tool=tool,
    )
    logger.info(f"Auto-created session: {name} (tool={tool}, model={model})")

    # Persist so it survives restarts
    try:
        save_sessions()
    except Exception as e:
        logger.warning(f"Auto-create: failed to persist: {e}")

    return session


def _detect_tool(headers: dict, path: str, body: dict | None = None) -> str:
    """Identify the source AI tool from request fingerprints.

    Priority:
      1. User-Agent (most reliable)
      2. Header / path / body heuristics

    Returns one of: claude-code, hermes, codex, cursor, claude-desktop,
                    anthropic-sdk, openai-sdk, openai-tool, anthropic-tool, unknown
    """
    ua = (headers.get("User-Agent") or headers.get("user-agent") or "").lower()

    # 1. User-Agent based detection (most reliable)
    if "claude-cli" in ua or "claude-code" in ua:
        return "claude-code"
    if "hermes" in ua:
        return "hermes"
    if "codex" in ua:
        return "codex"
    if "cursor" in ua:
        return "cursor"
    if "claude-desktop" in ua:
        return "claude-desktop"

    # 2. Header heuristics
    has_anthropic_version = bool(
        headers.get("anthropic-version") or headers.get("Anthropic-Version")
    )
    has_anthropic_beta = bool(
        headers.get("anthropic-beta") or headers.get("Anthropic-Beta")
    )

    # 3. Path-based + header inference
    # Use substring matching (not prefix-anchored) so both /v1/chat/completions
    # and /chat/completions work — hermes sends the latter (no /v1 prefix).
    if "/v1/messages" in path or path.endswith("/messages"):
        # Anthropic API format
        if has_anthropic_version or has_anthropic_beta:
            # Likely Claude Code or Anthropic SDK
            # Claude Code typically sends anthropic-beta headers
            if has_anthropic_beta:
                return "claude-code"
            return "anthropic-sdk"
        return "anthropic-tool"

    if "chat/completions" in path:
        # OpenAI API format (covers /v1/chat/completions and /chat/completions)
        if "python-httpx" in ua or "openai" in ua:
            return "openai-sdk"
        # Generic OpenAI-format tool (could be hermes, codex, etc. without UA)
        return "openai-tool"

    return "unknown"


# ── Session Registry ────────────────────────────────────────────────────

class Session:
    """One AI tool session: tool + model + upstream + API key + stats.

    Session identity = (tool, api_key). Same key used by different tools
    (e.g. hermes vs claude-code) produces separate sessions, because their
    prompt prefixes diverge from token 0 and cannot share cache.
    """

    def __init__(self, name: str, model: str, api_key: str,
                 provider: str = "", upstream: str = "",
                 api_format: str = "", tool: str = ""):
        import hashlib
        self.tool = tool or "unknown"
        # id is derived from (tool, api_key) so same key + different tool => different id
        self.id = hashlib.sha256(f"{self.tool}:{api_key}".encode()).hexdigest()[:12]
        self.name = name
        self.model = model
        self.api_key = api_key
        self.provider = provider
        self.upstream = upstream
        self.api_format = api_format
        self.stats = SessionStats()
        self.created_at = time.time()

    def to_dict(self) -> dict:
        s = self.stats.get_summary()
        cost = self._calc_session_cost()
        return {
            "id": self.id,
            "name": self.name,
            "tool": self.tool,
            "model": self.model,
            "provider": self.provider,
            "api_format": self.api_format,
            "upstream": self.upstream,
            "api_key_prefix": self.api_key[:8] + "..." if len(self.api_key) > 8 else "***",
            "cost": cost,
            **s,
        }

    def update(self, name: str | None = None, model: str | None = None,
               api_key: str | None = None, provider: str | None = None,
               upstream: str | None = None, api_format: str | None = None,
               tool: str | None = None):
        """Update session fields in-place, preserving stats.

        If api_key or tool changes, session.id is regenerated.
        Caller is responsible for re-keying the registry dict.
        """
        import hashlib
        if name is not None:
            self.name = name
        if model is not None:
            self.model = model
        if tool is not None:
            self.tool = tool
        if api_key is not None and api_key != self.api_key:
            self.api_key = api_key
        # Regenerate id if identity components changed
        self.id = hashlib.sha256(f"{self.tool}:{self.api_key}".encode()).hexdigest()[:12]
        if provider is not None:
            self.provider = provider
        if upstream is not None:
            self.upstream = upstream
        if api_format is not None:
            self.api_format = api_format

    def _calc_session_cost(self) -> dict:
        """Calculate this session's cost."""
        cfg = get_config(self.provider)
        if not cfg.pricing:
            return {"actual": 0.0, "without_opt": 0.0, "saved": 0.0, "currency": "CNY"}
        p = cfg.pricing
        input_price = p.input / 1_000_000
        cache_read_price = p.cache_read / 1_000_000
        output_price = p.output / 1_000_000
        st = self.stats
        actual = (
            (st.total_input_tokens + st.total_cache_creation_tokens) * input_price
            + st.total_cache_read_tokens * cache_read_price
            + st.total_output_tokens * output_price
        )
        # No-cache baseline: every prompt token billed at input price
        # (disjoint buckets: input + creation + read), matching diagnoser.
        total_input = (st.total_input_tokens + st.total_cache_creation_tokens
                       + st.total_cache_read_tokens)
        without_opt = total_input * input_price + st.total_output_tokens * output_price
        return {
            "actual": round(actual, 6),
            "without_opt": round(without_opt, 6),
            "saved": round(without_opt - actual, 6),
            "currency": p.currency,
        }


class SessionStats:
    """Per-session token statistics with multi-dimensional hit rate.

    Three hit-rate dimensions:
      - steady:        continuous dialogue (excludes first-round cold starts)
      - first_round:   cold-start requests (>10min since previous)
      - latest:        most recent request's hit rate
    Plus cache_status: active / aging / expired based on idle time.
    """

    # Threshold: gap > this many seconds => treat as a new "first round"
    FIRST_ROUND_GAP_SEC = 600  # 10 minutes

    def __init__(self):
        self.total_requests = 0
        self.total_input_tokens = 0
        self.total_cache_read_tokens = 0
        self.total_cache_creation_tokens = 0
        self.total_output_tokens = 0
        self.recent_logs: list[dict] = []

        # Multi-dimensional tracking
        self.last_request_ts: float = 0.0
        self.latest_request_ts: float = 0.0
        self.latest_hit_ratio: float = 0.0
        self._recent_hit_ratios: list[float] = []  # sliding window (last 5)

        # Cold-start bucket: session's first ever request (cache empty, 0% hit baseline)
        self.cs_requests = 0
        self.cs_cache_read = 0
        self.cs_input = 0
        self.cs_creation = 0

        # Revival bucket: first request after a >10min gap (may hit historical cache)
        self.rv_requests = 0
        self.rv_cache_read = 0
        self.rv_input = 0
        self.rv_creation = 0

        # Steady bucket: continuous dialogue (gap <= 10min)
        self.st_requests = 0
        self.st_cache_read = 0
        self.st_input = 0
        self.st_creation = 0

    def add_entry(self, elapsed: float, status, metrics=None):
        now = time.time()
        # Classify request phase:
        #   cold_start: session's first ever request (cache empty, 0% baseline)
        #   revival:    first request after >FIRST_ROUND_GAP_SEC (may hit historical cache)
        #   steady:     continuous dialogue (gap <= FIRST_ROUND_GAP_SEC)
        if self.last_request_ts == 0.0:
            phase = "cold_start"
        elif (now - self.last_request_ts) > self.FIRST_ROUND_GAP_SEC:
            phase = "revival"
        else:
            phase = "steady"

        # Backward-compat flag: cold_start + revival both count as "first round"
        is_first_round = phase in ("cold_start", "revival")

        self.total_requests += 1
        entry = {
            "timestamp": time.strftime("%H:%M:%S"),
            "epoch": now,
            "elapsed": round(elapsed, 2),
            "status": status,
            "is_first_round": is_first_round,
            "phase": phase,
        }

        # Count every request into its phase bucket so the per-phase request
        # counts always sum to total_requests (usage-less passthroughs and
        # error responses included).
        if phase == "cold_start":
            self.cs_requests += 1
        elif phase == "revival":
            self.rv_requests += 1
        else:
            self.st_requests += 1

        # A real cache data point needs metrics with non-zero prompt tokens.
        # parse_usage returns a zero-filled CacheMetrics (not None) for error
        # bodies, so guard on token count — otherwise a 429/5xx would be
        # recorded as a cache miss and drag the hit ratios toward zero.
        has_usage = bool(metrics) and (
            metrics.input_tokens
            + metrics.cache_read_input_tokens
            + metrics.cache_creation_input_tokens
        ) > 0
        if has_usage:
            cr = metrics.cache_read_input_tokens
            it = metrics.input_tokens
            cc = metrics.cache_creation_input_tokens
            self.total_input_tokens += it
            self.total_cache_read_tokens += cr
            self.total_cache_creation_tokens += cc
            self.total_output_tokens += metrics.output_tokens
            entry["input_tokens"] = it
            entry["cache_read_tokens"] = cr
            entry["cache_creation_tokens"] = cc
            entry["output_tokens"] = metrics.output_tokens
            entry["hit"] = cr > 0

            # Phase token sums (request count already incremented above)
            if phase == "cold_start":
                self.cs_cache_read += cr
                self.cs_input += it
                self.cs_creation += cc
            elif phase == "revival":
                self.rv_cache_read += cr
                self.rv_input += it
                self.rv_creation += cc
            else:
                self.st_cache_read += cr
                self.st_input += it
                self.st_creation += cc

            # Latest hit ratio = sliding-window mean of last 5 real requests.
            # Strict denominator: cache_read + input + cache_creation
            # (cache_creation is uncached-but-written, i.e. a miss that invested
            #  in future hits — must count in denominator or ratio is inflated)
            tot = cr + it + cc
            ratio = cr / tot if tot > 0 else 0.0
            self._recent_hit_ratios.append(ratio)
            if len(self._recent_hit_ratios) > 5:
                self._recent_hit_ratios = self._recent_hit_ratios[-5:]
            self.latest_hit_ratio = (
                sum(self._recent_hit_ratios) / len(self._recent_hit_ratios)
                if self._recent_hit_ratios else 0.0
            )
            self.latest_request_ts = now
        else:
            entry["hit"] = None
            entry["input_tokens"] = 0
            entry["cache_read_tokens"] = 0
            entry["cache_creation_tokens"] = 0
            entry["output_tokens"] = 0

        self.last_request_ts = now
        self.recent_logs.append(entry)
        if len(self.recent_logs) > 50:
            self.recent_logs = self.recent_logs[-50:]

    def get_summary(self) -> dict:
        # Strict denominator: cache_read + input_tokens + cache_creation_tokens.
        # cache_creation is uncached-but-written (a miss that invested in future
        # hits) — counting it in the denominator prevents inflated hit rates.
        def _ratio(read: float, inp: float, creation: float) -> float:
            denom = read + inp + creation
            return read / denom if denom > 0 else 0.0

        # Cold-start: session's first request (baseline, normally 0%)
        cs_ratio = _ratio(self.cs_cache_read, self.cs_input, self.cs_creation)
        # Revival: first request after a gap (historical cache reuse)
        rv_ratio = _ratio(self.rv_cache_read, self.rv_input, self.rv_creation)
        # First-round (compat): cold_start + revival combined
        fr_ratio = _ratio(
            self.cs_cache_read + self.rv_cache_read,
            self.cs_input + self.rv_input,
            self.cs_creation + self.rv_creation,
        )
        # Steady: continuous dialogue (the real optimization effect)
        st_ratio = _ratio(self.st_cache_read, self.st_input, self.st_creation)
        # Overall (compat)
        overall = _ratio(
            self.total_cache_read_tokens,
            self.total_input_tokens,
            self.total_cache_creation_tokens,
        )

        # Cache liveness based on idle time since latest request
        if self.latest_request_ts > 0:
            idle = time.time() - self.latest_request_ts
            if idle < self.FIRST_ROUND_GAP_SEC:
                cache_status = "active"     # within TTL window
            elif idle < 3600:
                cache_status = "aging"      # likely still cached but risky
            else:
                cache_status = "expired"    # almost certainly evicted
        else:
            idle = 0.0
            cache_status = "idle"

        return {
            "total_requests": self.total_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_cache_read_tokens": self.total_cache_read_tokens,
            "total_cache_creation_tokens": self.total_cache_creation_tokens,
            "total_output_tokens": self.total_output_tokens,
            "hit_ratio": round(overall, 4),                       # overall (compat)
            "steady_hit_ratio": round(st_ratio, 4),               # steady (main)
            "first_round_hit_ratio": round(fr_ratio, 4),          # first-round (compat: cs+rv)
            "cold_start_hit_ratio": round(cs_ratio, 4),           # cold-start baseline
            "revival_hit_ratio": round(rv_ratio, 4),              # revival reuse
            "latest_hit_ratio": round(self.latest_hit_ratio, 4),  # sliding window (last 5)
            "cache_status": cache_status,
            "idle_seconds": int(idle),
            "steady_requests": self.st_requests,
            "first_round_requests": self.cs_requests + self.rv_requests,
            "cold_start_requests": self.cs_requests,
            "revival_requests": self.rv_requests,
        }

    def get_logs(self) -> list[dict]:
        return list(reversed(self.recent_logs))


class SessionRegistry:
    """Thread-safe registry of sessions, keyed by (tool, api_key).

    Same API key used by different tools produces separate sessions,
    because their prompt prefixes diverge from token 0 and cannot share cache.
    """

    def __init__(self):
        self._lock = threading.Lock()
        # dict key = f"{tool}:{api_key}" -> Session
        self._sessions: dict[str, Session] = {}

    @staticmethod
    def _make_key(tool: str, api_key: str) -> str:
        return f"{tool or 'unknown'}:{api_key}"

    def add(self, name: str, model: str, api_key: str,
            provider: str = "", upstream: str = "",
            api_format: str = "", tool: str = "") -> Session:
        with self._lock:
            session = Session(name, model, api_key, provider, upstream,
                              api_format, tool)
            k = self._make_key(session.tool, api_key)
            self._sessions[k] = session
            return session

    def remove(self, identifier: str) -> bool:
        """Remove by registry key, session id, or api_key (legacy)."""
        with self._lock:
            # Direct key match
            if identifier in self._sessions:
                del self._sessions[identifier]
                return True
            # Match by session id
            for key, session in self._sessions.items():
                if session.id == identifier:
                    del self._sessions[key]
                    return True
            # Legacy: match by api_key alone (first match)
            for key, session in self._sessions.items():
                if session.api_key == identifier:
                    del self._sessions[key]
                    return True
            return False

    def get_by_id(self, session_id: str) -> Session | None:
        """Find a session by its hashed id."""
        with self._lock:
            for session in self._sessions.values():
                if session.id == session_id:
                    return session
            return None

    def update_session(self, session_id: str, name: str | None = None,
                       model: str | None = None, api_key: str | None = None,
                       tool: str | None = None) -> Session | None:
        """Update a session by id, preserving stats.

        If api_key or tool changes, the registry dict is re-keyed.
        If model changes, provider/upstream/api_format are re-resolved.
        """
        with self._lock:
            old_key = None
            session = None
            for key, s in self._sessions.items():
                if s.id == session_id:
                    old_key = key
                    session = s
                    break
            if not session:
                return None

            # Re-resolve model info if model changed
            provider = session.provider
            upstream = session.upstream
            api_format = session.api_format
            if model is not None and model != session.model:
                model_info = resolve_model(model)
                if not model_info:
                    return None
                provider = model_info.provider
                upstream = model_info.base_url or ""
                api_format = model_info.api_format

            new_tool = tool if tool is not None else session.tool
            new_api_key = api_key if api_key is not None else session.api_key

            session.update(
                name=name, model=model, api_key=api_key,
                provider=provider, upstream=upstream, api_format=api_format,
                tool=tool,
            )

            new_key = self._make_key(new_tool, new_api_key)
            if new_key != old_key:
                del self._sessions[old_key]
                self._sessions[new_key] = session

            return session

    def get_all(self) -> list[Session]:
        with self._lock:
            return list(self._sessions.values())

    def find_by_auth_and_tool(self, headers: dict, tool: str) -> Session | None:
        """Find session by (api_key extracted from headers, tool).

        Returns the matching session, or None if no session has this combo.
        """
        api_key = _extract_api_key(headers)
        if not api_key:
            return None
        k = self._make_key(tool, api_key)
        with self._lock:
            return self._sessions.get(k)

    def get_dashboard(self) -> dict:
        """Return aggregate stats + per-session breakdown."""
        sessions = self.get_all()
        if not sessions:
            return {"sessions": [], "aggregate": self._empty_aggregate()}

        agg_input = 0
        agg_cache_read = 0
        agg_cache_create = 0
        agg_output = 0
        agg_requests = 0
        # Phase buckets: cold_start / revival / steady (each tracks read/input/creation)
        agg_cs_read = 0; agg_cs_input = 0; agg_cs_creation = 0
        agg_rv_read = 0; agg_rv_input = 0; agg_rv_creation = 0
        agg_st_read = 0; agg_st_input = 0; agg_st_creation = 0
        all_logs = []
        session_summaries = []

        for s in sessions:
            summary = s.stats.get_summary()
            agg_input += summary["total_input_tokens"]
            agg_cache_read += summary["total_cache_read_tokens"]
            agg_cache_create += summary["total_cache_creation_tokens"]
            agg_output += summary["total_output_tokens"]
            agg_requests += summary["total_requests"]
            agg_cs_read += s.stats.cs_cache_read
            agg_cs_input += s.stats.cs_input
            agg_cs_creation += s.stats.cs_creation
            agg_rv_read += s.stats.rv_cache_read
            agg_rv_input += s.stats.rv_input
            agg_rv_creation += s.stats.rv_creation
            agg_st_read += s.stats.st_cache_read
            agg_st_input += s.stats.st_input
            agg_st_creation += s.stats.st_creation
            session_summaries.append(s.to_dict())
            for log in s.stats.get_logs():
                log["session"] = s.name
                log["tool"] = s.tool
                all_logs.append(log)

        all_logs.sort(key=lambda x: x.get("epoch", 0), reverse=True)
        all_logs = all_logs[:100]

        # Strict denominator: read + input + creation (creation is a miss)
        def _ratio(read: float, inp: float, creation: float) -> float:
            denom = read + inp + creation
            return read / denom if denom > 0 else 0.0

        hit_ratio = _ratio(agg_cache_read, agg_input, agg_cache_create)
        steady_ratio = _ratio(agg_st_read, agg_st_input, agg_st_creation)
        cold_start_ratio = _ratio(agg_cs_read, agg_cs_input, agg_cs_creation)
        revival_ratio = _ratio(agg_rv_read, agg_rv_input, agg_rv_creation)
        # first_round (compat) = cold_start + revival combined
        first_round_ratio = _ratio(
            agg_cs_read + agg_rv_read,
            agg_cs_input + agg_rv_input,
            agg_cs_creation + agg_rv_creation,
        )

        costs = self._calc_costs(sessions)

        return {
            "sessions": session_summaries,
            "aggregate": {
                "total_requests": agg_requests,
                "total_input_tokens": agg_input,
                "total_cache_read_tokens": agg_cache_read,
                "total_cache_creation_tokens": agg_cache_create,
                "total_output_tokens": agg_output,
                "hit_ratio": round(hit_ratio, 4),
                "steady_hit_ratio": round(steady_ratio, 4),
                "first_round_hit_ratio": round(first_round_ratio, 4),
                "cold_start_hit_ratio": round(cold_start_ratio, 4),
                "revival_hit_ratio": round(revival_ratio, 4),
                "actual_cost": costs["actual_cost"],
                "cost_without_opt": costs["cost_without_opt"],
                "cost_saved": costs["cost_saved"],
                "currency": costs["currency"],
            },
            "logs": all_logs,
        }

    def _empty_aggregate(self) -> dict:
        return {
            "total_requests": 0,
            "total_input_tokens": 0,
            "total_cache_read_tokens": 0,
            "total_cache_creation_tokens": 0,
            "total_output_tokens": 0,
            "hit_ratio": 0.0,
            "steady_hit_ratio": 0.0,
            "first_round_hit_ratio": 0.0,
            "cold_start_hit_ratio": 0.0,
            "revival_hit_ratio": 0.0,
            "actual_cost": 0.0,
            "cost_without_opt": 0.0,
            "cost_saved": 0.0,
            "currency": "CNY",
        }

    def _calc_costs(self, sessions: list[Session]) -> dict:
        """Calculate costs across all sessions."""
        actual_cost = 0.0
        cost_without_opt = 0.0
        currency = "CNY"

        for s in sessions:
            cfg = get_config(s.provider)
            if not cfg.pricing:
                continue
            p = cfg.pricing
            currency = p.currency
            input_price = p.input / 1_000_000
            cache_read_price = p.cache_read / 1_000_000
            output_price = p.output / 1_000_000

            st = s.stats
            actual_cost += (
                (st.total_input_tokens + st.total_cache_creation_tokens) * input_price
                + st.total_cache_read_tokens * cache_read_price
                + st.total_output_tokens * output_price
            )
            # No-cache baseline includes creation (billed at input price)
            total_input = (st.total_input_tokens + st.total_cache_creation_tokens
                           + st.total_cache_read_tokens)
            cost_without_opt += (
                total_input * input_price
                + st.total_output_tokens * output_price
            )

        return {
            "actual_cost": round(actual_cost, 6),
            "cost_without_opt": round(cost_without_opt, 6),
            "cost_saved": round(cost_without_opt - actual_cost, 6),
            "currency": currency,
        }


# ── Persistence ─────────────────────────────────────────────────────────

def _persistence_path() -> str:
    """Path to sessions config file (.popt/sessions.json in project root)."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    popt_dir = os.path.join(project_root, ".popt")
    os.makedirs(popt_dir, exist_ok=True)
    return os.path.join(popt_dir, "sessions.json")


def save_sessions() -> None:
    """Persist all session configs (without stats) to .popt/sessions.json."""
    sessions = _registry.get_all()
    data = {
        "sessions": [
            {
                "name": s.name,
                "tool": s.tool,
                "model": s.model,
                "api_key": s.api_key,
                "provider": s.provider,
                "upstream": s.upstream,
                "api_format": s.api_format,
            }
            for s in sessions
        ]
    }
    try:
        path = _persistence_path()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(sessions)} sessions to {path}")
    except Exception as e:
        logger.warning(f"Failed to save sessions: {e}")


def load_sessions() -> int:
    """Load session configs from .popt/sessions.json into the registry.

    Returns the number of sessions loaded.
    """
    try:
        path = _persistence_path()
        if not os.path.exists(path):
            return 0
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        count = 0
        for s in data.get("sessions", []):
            try:
                _registry.add(
                    name=s.get("name", "loaded"),
                    model=s.get("model", ""),
                    api_key=s["api_key"],
                    provider=s.get("provider", ""),
                    upstream=s.get("upstream", ""),
                    api_format=s.get("api_format", ""),
                    tool=s.get("tool", "unknown"),
                )
                count += 1
            except (KeyError, ValueError) as e:
                logger.warning(f"Skipping malformed session entry: {e}")
        logger.info(f"Loaded {count} sessions from {path}")
        return count
    except Exception as e:
        logger.warning(f"Failed to load sessions: {e}")
        return 0


# Module-level singleton
_registry = SessionRegistry()

# ── Model list ──────────────────────────────────────────────────────────

EXAMPLE_MODELS = [
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "deepseek-v3.2",
    "deepseek-chat",
    "deepseek-reasoner",
    "claude-sonnet-5-20250601",
    "claude-haiku-4-20250506",
    "claude-opus-4-20250514",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1",
    "grok-beta",
    "llama-3.3-70b-versatile",
    "gemini-2.0-flash",
    "gemini-2.5-pro",
    "mistral-large-latest",
]


# ── GUI Proxy Handler ───────────────────────────────────────────────────

class GUIProxyHandler(ProxyHandler):
    """Multi-session proxy handler: web GUI + stats + auto-routing."""

    # ── GET ─────────────────────────────────────────────────────

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._serve_html()
        elif self.path == "/api/init":
            self._api_init()
        elif self.path == "/api/dashboard":
            self._api_dashboard()
        elif self.path == "/api/sessions":
            self._api_list_sessions()
        else:
            self._proxy_get_passthrough()

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Max-Age", "86400")
        self.end_headers()

    def do_POST(self):
        if self.path == "/api/config":
            self._api_add_session()
            return
        if self.path == "/api/session/delete":
            self._api_delete_session()
            return
        if self.path == "/api/session/update":
            self._api_update_session()
            return
        if self.path == "/login" or self.path == "/v1/login":
            self._send_json(200, {"status": "ok", "message": "login bypassed"})
            return
        self._proxy_with_stats()

    # ── API Endpoints ───────────────────────────────────────────

    def _serve_html(self):
        html_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "gui", "index.html",
        )
        try:
            with open(html_path, "r", encoding="utf-8") as f:
                content = f.read()
            body = content.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except FileNotFoundError:
            self._send_error(404, "GUI file not found")

    def _api_init(self):
        """Return model list + session count for page initialization."""
        sessions = _registry.get_all()
        self._send_json(200, {
            "models": EXAMPLE_MODELS,
            "session_count": len(sessions),
            "providers": list({s.provider for s in sessions if s.provider}),
        })

    def _api_list_sessions(self):
        """Return all sessions with stats."""
        self._send_json(200, {
            "sessions": [s.to_dict() for s in _registry.get_all()],
        })

    def _api_add_session(self):
        """Add a new session (model + API key)."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON"})
            return

        model = data.get("model", "").strip()
        api_key = data.get("api_key", "").strip()
        name = data.get("name", "").strip() or model
        tool = data.get("tool", "").strip() or "manual"

        if not model:
            self._send_json(400, {"error": "Model name is required"})
            return
        if not api_key:
            self._send_json(400, {"error": "API Key is required"})
            return

        model_info = resolve_model(model)
        if not model_info:
            self._send_json(400, {"error": f"Unknown model: {model}"})
            return

        # Check if session with this (tool, api_key) already exists
        existing = _registry.find_by_auth_and_tool({"x-api-key": api_key}, tool)
        if existing:
            _registry.update_session(
                session_id=existing.id,
                name=name, model=model, api_key=api_key, tool=tool,
            )
            save_sessions()
            self._send_json(200, {"message": "Session updated", "session": existing.to_dict()})
        else:
            session = _registry.add(
                name=name,
                model=model,
                api_key=api_key,
                provider=model_info.provider,
                upstream=model_info.base_url or "",
                api_format=model_info.api_format,
                tool=tool,
            )
            save_sessions()
            self._send_json(200, {"message": "Session created", "session": session.to_dict()})

    def _api_delete_session(self):
        """Delete a session by API key or session id."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON"})
            return

        # Accept both legacy `api_key` and frontend `session_id` field names
        identifier = data.get("session_id") or data.get("api_key") or ""
        if not identifier:
            self._send_json(400, {"error": "session_id or api_key is required"})
            return

        if _registry.remove(identifier):
            save_sessions()
            self._send_json(200, {"message": "Session deleted"})
        else:
            self._send_json(404, {"error": "Session not found"})

    def _api_update_session(self):
        """Update an existing session by id. Preserves stats."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON"})
            return

        session_id = data.get("session_id", "").strip()
        if not session_id:
            self._send_json(400, {"error": "session_id is required"})
            return

        name = data.get("name", "").strip() or None
        model = data.get("model", "").strip() or None
        api_key = data.get("api_key", "").strip() or None
        tool = data.get("tool", "").strip() or None

        # At least one field must be provided
        if name is None and model is None and api_key is None and tool is None:
            self._send_json(400, {"error": "Nothing to update"})
            return

        # If model provided, validate it resolves
        if model:
            model_info = resolve_model(model)
            if not model_info:
                self._send_json(400, {"error": f"Unknown model: {model}"})
                return

        updated = _registry.update_session(
            session_id=session_id,
            name=name, model=model, api_key=api_key, tool=tool,
        )
        if updated:
            save_sessions()
            self._send_json(200, {"message": "Session updated", "session": updated.to_dict()})
        else:
            self._send_json(404, {"error": "Session not found"})

    def _api_dashboard(self):
        self._send_json(200, _registry.get_dashboard())

    # ── Proxy: GET passthrough ────────────────────────────────

    def _proxy_get_passthrough(self):
        """Forward GET requests to upstream."""
        headers = dict(self.headers)
        tool = _detect_tool(headers, self.path)
        session = _registry.find_by_auth_and_tool(headers, tool)
        if not session:
            session = _auto_create_session(headers, self.path, tool)
        if not session:
            self._send_error(401, "No matching session. Add your API key in the panel first.")
            return

        upstream_url = self._resolve_upstream(session, self.path)
        if not upstream_url:
            self._send_error(502, f"Cannot resolve upstream for: {self.path}")
            return

        fwd_headers = self._build_fwd_headers(session)
        start_time = time.time()

        try:
            req = Request(upstream_url, headers=fwd_headers, method="GET")
            resp = urlopen(req, timeout=60)
            resp_body = resp.read()
            elapsed = time.time() - start_time
            session.stats.add_entry(elapsed, resp.status)
            self._send_response(resp.status, dict(resp.headers), resp_body)
        except HTTPError as e:
            err_body = e.read()
            elapsed = time.time() - start_time
            session.stats.add_entry(elapsed, e.code)
            self._send_response(e.code, dict(e.headers), err_body)
        except URLError as e:
            self._send_error(502, f"Upstream connection failed: {e.reason}")
        except Exception as e:
            self._send_error(500, f"Proxy error: {e}")

    # ── Proxy: POST with stats ────────────────────────────────

    def _proxy_with_stats(self):
        """Forward POST request with API key routing + stats."""
        start_time = time.time()

        # ─ Read request body ──────────────────────────────────
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        if not body:
            self._send_error(400, "Empty request body")
            return

        try:
            request_data = json.loads(body)
        except json.JSONDecodeError as e:
            self._send_error(400, f"Invalid JSON: {e}")
            return

        # ── Find session by (tool, api_key) ───────────────────
        headers = dict(self.headers)
        tool = _detect_tool(headers, self.path, request_data)
        # Debug: log what auth headers arrived (mask key value)
        _dbg_key = _extract_api_key(headers)
        _dbg_has_auth = bool(headers.get("Authorization") or headers.get("authorization"))
        _dbg_has_xkey = bool(headers.get("x-api-key") or headers.get("X-Api-Key"))
        logger.info(
            f"[proxy] path={self.path} tool={tool} "
            f"api_key_extracted={_dbg_key[:4] + '...' if _dbg_key else 'EMPTY'} "
            f"has_auth_header={_dbg_has_auth} has_x_api_key={_dbg_has_xkey}"
        )
        session = _registry.find_by_auth_and_tool(headers, tool)
        if not session:
            session = _auto_create_session(headers, self.path, tool, request_data)
        if not session:
            self._send_error(401, "No matching session. Add your API key in the panel first.")
            return

        # ── Resolve upstream URL ───────────────────────────────
        upstream_url = self._resolve_upstream(session, self.path)
        if not upstream_url:
            self._send_error(502, f"Cannot resolve upstream for: {self.path}")
            return

        # ── Optimize messages ──────────────────────────────────
        messages = request_data.get("messages", [])
        if messages:
            try:
                optimized_messages = optimize(messages, provider=session.provider)
                request_data["messages"] = optimized_messages

                if "system" in request_data:
                    sys_msg = {"role": "system", "content": request_data["system"]}
                    opt_msgs = optimize([sys_msg] + messages, provider=session.provider)
                    if opt_msgs and opt_msgs[0].get("role") == "system":
                        request_data["system"] = opt_msgs[0]["content"]

                body = json.dumps(request_data).encode("utf-8")
            except Exception as e:
                logger.error(f"Optimization failed: {e}")

        is_streaming = request_data.get("stream", False)

        # ── Build headers ──────────────────────────────────────
        fwd_headers = self._build_fwd_headers(session)

        # ── Forward to upstream ────────────────────────────────
        try:
            req = Request(upstream_url, data=body, headers=fwd_headers, method="POST")

            if is_streaming:
                resp = urlopen(req, timeout=60)
                self.send_response(resp.status)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                metrics = self._forward_streaming_with_stats(resp)
                elapsed = time.time() - start_time
                session.stats.add_entry(elapsed, "stream", metrics=metrics)
                logger.info(f"STREAM | {session.name} | {session.model} | {elapsed:.1f}s" +
                           (f" | in:{metrics.input_tokens} read:{metrics.cache_read_input_tokens}" if metrics else ""))
            else:
                resp = urlopen(req, timeout=120)
                resp_body = resp.read()
                elapsed = time.time() - start_time

                metrics = None
                try:
                    resp_text = resp_body.decode("utf-8", errors="replace")
                    resp_data = json.loads(resp_text)
                    metrics = parse_usage(resp_data)
                except Exception:
                    # Metrics are best-effort; a malformed or edge-case usage
                    # object must not turn a good 200 into a client 500.
                    pass

                session.stats.add_entry(elapsed, resp.status, metrics=metrics)
                logger.info(f"FORWARD | {session.name} | {session.model} | {resp.status} | {elapsed:.2f}s" +
                           (f" | in:{metrics.input_tokens} read:{metrics.cache_read_input_tokens}" if metrics else ""))

                self._send_response(resp.status, dict(resp.headers), resp_body)

        except HTTPError as e:
            err_body = e.read()
            elapsed = time.time() - start_time
            metrics = None
            try:
                err_text = err_body.decode("utf-8", errors="replace")
                err_data = json.loads(err_text)
                metrics = parse_usage(err_data)
            except Exception:
                # Best-effort metrics on the error path too.
                pass
            session.stats.add_entry(elapsed, e.code, metrics=metrics)
            logger.warning(f"UPSTREAM ERROR | {session.name} | {e.code} | {elapsed:.2f}s | {err_body[:200]}")
            self._send_response(e.code, dict(e.headers), err_body)

        except URLError as e:
            logger.error(f"Upstream connection failed: {e.reason}")
            self._send_error(502, f"Upstream connection failed: {e.reason}")

        except Exception as e:
            logger.error(f"Proxy error: {e}")
            self._send_error(500, f"Internal proxy error: {e}")

    # ── Streaming stats ───────────────────────────────────────

    def _forward_streaming_with_stats(self, upstream_resp):
        """Forward SSE stream while parsing usage data from events.

        Supports both Anthropic format (message_start/message_delta) and
        OpenAI format (choices[] with final usage chunk).
        """
        import socket as _socket
        try:
            if hasattr(upstream_resp, "fp") and hasattr(upstream_resp.fp, "raw"):
                upstream_resp.fp.raw._sock.settimeout(120)
        except (AttributeError, OSError):
            pass

        metrics = CacheMetrics()
        buffer = b""
        try:
            while True:
                chunk = upstream_resp.read(4096)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()

                buffer += chunk
                lines = buffer.split(b"\n")
                buffer = lines[-1]

                for line in lines[:-1]:
                    line = line.strip()
                    if line.startswith(b"data: "):
                        try:
                            data = line[6:].decode("utf-8")
                            if data == "[DONE]":
                                continue
                            event = json.loads(data)
                            evt_type = event.get("type", "")

                            # ── Anthropic format ──
                            if evt_type == "message_start" and event.get("message"):
                                msg = event["message"]
                                usage = msg.get("usage", {})
                                raw_input = usage.get("input_tokens", 0)
                                cache_read = usage.get("cache_read_input_tokens", 0)
                                cache_create = usage.get("cache_creation_input_tokens", 0)
                                # input_tokens is already the fresh (uncached)
                                # portion, disjoint from cache_read/creation —
                                # Anthropic convention, and DeepSeek /anthropic
                                # matches it (live-verified: warm request gave
                                # input=79, read=384, cold total=463).
                                metrics.input_tokens = raw_input
                                metrics.cache_read_input_tokens = cache_read
                                metrics.cache_creation_input_tokens = cache_create
                                metrics.output_tokens = usage.get("output_tokens", 0)

                            elif evt_type == "message_delta":
                                # Anthropic format: usage can be at event level or inside delta
                                usage = event.get("usage", {})
                                if not usage and event.get("delta"):
                                    usage = event["delta"].get("usage", {})
                                if "output_tokens" in usage:
                                    metrics.output_tokens = usage.get("output_tokens", 0)

                            # ── OpenAI format ──
                            # Usage arrives in the final chunk, identified by
                            # having a "usage" key at the top level (not nested
                            # under "message" or "delta").
                            elif "usage" in event and "choices" in event:
                                usage = event["usage"]
                                prompt_tokens = usage.get("prompt_tokens", 0)
                                completion_tokens = usage.get("completion_tokens", 0)
                                # DeepSeek OpenAI format: prompt_cache_hit_tokens
                                cache_read = usage.get("prompt_cache_hit_tokens", 0)
                                # OpenAI standard: prompt_tokens_details.cached_tokens
                                if not cache_read:
                                    details = usage.get("prompt_tokens_details") or {}
                                    cache_read = details.get("cached_tokens", 0)
                                # input_tokens = non-cached portion
                                metrics.input_tokens = max(0, prompt_tokens - cache_read)
                                metrics.cache_read_input_tokens = cache_read
                                metrics.cache_creation_input_tokens = 0
                                metrics.output_tokens = completion_tokens

                        except (json.JSONDecodeError, ValueError):
                            pass

        except (BrokenPipeError, ConnectionResetError):
            logger.warning("Client disconnected during streaming")
        except _socket.timeout:
            logger.warning("Streaming read timed out")
        except Exception as e:
            logger.error(f"Streaming error: {e}")

        return metrics

    # ── Helpers ────────────────────────────────────────────────

    # Paths that are NOT LLM API calls — pass through to upstream as-is
    # (e.g. hermes internal APIs like /api/show, /api/models, etc.)
    _NON_LLM_PATHS = frozenset({
        "/api/show", "/api/models", "/api/health", "/api/config",
    })

    def _resolve_upstream(self, session: Session, path: str) -> str | None:
        """Build the upstream URL for a session + path.

        Performs path-aware format adaptation: if the session's upstream
        is an Anthropic endpoint but the request path is OpenAI-format
        (or vice versa), switch to the correct endpoint for the provider.
        """
        base = session.upstream.rstrip("/")
        if not base:
            return None

        # Non-LLM paths: strip the path and return the base URL so the
        # client gets a 404 from upstream (or we could pass through).
        # These are tool-internal APIs (e.g. hermes /api/show) that the
        # upstream LLM provider doesn't serve.
        for non_llm in self._NON_LLM_PATHS:
            if path == non_llm or path.startswith(non_llm + "/"):
                # Return base URL — upstream will 404, which is correct
                return base

        # Prevent double-path
        if path and base.endswith(path):
            return base

        # Path-aware format adaptation:
        # If session upstream is Anthropic-format (/anthropic) but request
        # is OpenAI-format (/chat/completions), switch base to root.
        # Vice versa: if upstream is root but request is /v1/messages,
        # append /anthropic.
        is_openai_path = "chat/completions" in path
        is_anthropic_path = "/v1/messages" in path or path.endswith("/messages")
        if is_openai_path and "/anthropic" in base:
            base = base.replace("/anthropic", "")
        elif is_anthropic_path and "/anthropic" not in base:
            # Only providers that expose an /anthropic sub-path (e.g. DeepSeek:
            # api.deepseek.com/anthropic) need it appended. Native Anthropic
            # (api.anthropic.com) serves /v1/messages directly — appending 404s.
            entry = next(
                (e for e in load_providers() if e.get("name") == session.provider),
                None,
            )
            if entry and (entry.get("base_url") or "").rstrip("/").endswith("/anthropic"):
                base = base + "/anthropic"

        # Avoid doubling a version segment when the base already carries /v1
        # and the request path supplies it too (groq/mistral/together/... use
        # .../v1 bases; the client is told to use OPENAI_BASE_URL=.../v1).
        if base.endswith("/v1") and path.startswith("/v1/"):
            base = base[: -len("/v1")]

        return f"{base}{path}"

    def _build_fwd_headers(self, session: Session) -> dict:
        """Build forwarding headers with API key injection.

        Auth header is determined by the actual request path, not the
        session's stored api_format — because _resolve_upstream may
        switch endpoints (e.g. Anthropic upstream + OpenAI path).
        """
        fwd = _forward_headers(dict(self.headers))
        fwd.pop("Accept-Encoding", None)
        fwd.pop("accept-encoding", None)

        # Determine auth format from the request path, not session config
        is_openai_path = "chat/completions" in self.path
        if is_openai_path:
            fwd["Authorization"] = f"Bearer {session.api_key}"
            fwd.pop("x-api-key", None)
            fwd.pop("X-Api-Key", None)
        else:
            fwd["x-api-key"] = session.api_key
            fwd.pop("Authorization", None)
            fwd.pop("authorization", None)

        return fwd

    def _send_json(self, status: int, data: dict):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)


# ── Server Runner ───────────────────────────────────────────────────────

def run_gui(host: str = "127.0.0.1", port: int = 6123, model: str = "",
            open_browser: bool = True):
    """Start the GUI + proxy server on a single port."""
    # Load persisted session configs first
    loaded = load_sessions()

    # Pre-fill from environment (only if no persisted sessions exist)
    env_key = (
        os.environ.get("ANTHROPIC_API_KEY", "")
        or os.environ.get("OPENAI_API_KEY", "")
        or os.environ.get("DEEPSEEK_API_KEY", "")
    )
    env_model = model or os.environ.get("POPT_MODEL", "")

    if env_model and env_key and loaded == 0:
        model_info = resolve_model(env_model)
        if model_info:
            _registry.add(
                name=f"env:{env_model}",
                model=env_model,
                api_key=env_key,
                provider=model_info.provider,
                upstream=model_info.base_url or "",
                api_format=model_info.api_format,
                tool="env",
            )
            save_sessions()

    server = ThreadedProxyServer((host, port), GUIProxyHandler)

    sessions = _registry.get_all()
    print(f"\n  popt gui running on http://{host}:{port}")
    print(f"  {'=' * 50}")
    print(f"  [OK] Web panel:  http://{host}:{port}")
    print(f"  [OK] Proxy:      same address")
    if sessions:
        for s in sessions:
            print(f"  Session:  {s.name} ({s.model} via {s.provider})")
    else:
        print(f"  [!] No sessions configured -- open the panel to add")
    print(f"  {'=' * 50}")
    print(f"  Open browser:        http://{host}:{port}")
    print(f"  Point AI tool to:    http://{host}:{port}")
    print(f"  Press Ctrl+C to stop\n")

    # Auto-open browser after a short delay (server must be ready first)
    if open_browser:
        import threading
        import webbrowser
        url = f"http://{host}:{port}"
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Shutting down...")
        server.shutdown()
