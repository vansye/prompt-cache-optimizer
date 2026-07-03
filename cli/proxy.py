"""Local HTTP proxy — transparent prompt optimization.

Listens on localhost, accepts API requests, optimizes the prompt
structure, and forwards to the real API provider.

Usage:
    popt proxy --port 9999
    # Then set ANTHROPIC_BASE_URL=http://localhost:9999
    # or OPENAI_BASE_URL=http://localhost:9999
"""

import json
import logging
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from typing import Optional

from optimizer import optimize
from optimizer.config import PROVIDER_CONFIGS

logger = logging.getLogger("popt-proxy")

# ── Upstream routing ────────────────────────────────────────────────────

# Default upstreams for each provider
UPSTREAMS = {
    "anthropic": "https://api.anthropic.com",
    "openai": "https://api.openai.com",
}

# Custom upstream override (set via --upstream or set_custom_upstream)
_custom_upstream: str | None = None
_custom_provider: str | None = None


def set_custom_upstream(url: str | None):
    """Set a custom upstream URL, overriding auto-detection."""
    global _custom_upstream
    _custom_upstream = url


def set_custom_provider(provider: str | None):
    """Set a custom provider for optimization, overriding auto-detection."""
    global _custom_provider
    _custom_provider = provider

# Hop-by-hop headers that MUST NOT be forwarded
HOP_BY_HOP = frozenset({
    "host", "connection", "keep-alive", "proxy-authenticate",
    "proxy-authorization", "te", "trailers",
    "transfer-encoding", "upgrade",
})


def _detect_provider(path: str) -> str:
    """Detect API provider from the request path."""
    if "/v1/messages" in path:
        return "anthropic"
    if "/v1/chat/completions" in path:
        return "openai"
    # Default to anthropic for unknown paths
    return "unknown"


def _forward_headers(headers: dict) -> dict:
    """Forward headers, stripping hop-by-hop headers."""
    return {
        k: v for k, v in headers.items()
        if k.lower() not in HOP_BY_HOP
    }


# ── SSE passthrough ─────────────────────────────────────────────────────


def _forward_streaming(upstream_resp, handler):
    """Forward an SSE (Server-Sent Events) stream from upstream to client.

    Reads chunks from upstream as they arrive and writes them
    immediately to the client response stream.
    """
    try:
        while True:
            chunk = upstream_resp.read(4096)
            if not chunk:
                break
            handler.wfile.write(chunk)
            handler.wfile.flush()
    except (BrokenPipeError, ConnectionResetError):
        logger.warning("Client disconnected during streaming")
    except Exception as e:
        logger.error(f"Streaming error: {e}")


# ── Request Handler ─────────────────────────────────────────────────────


class ProxyHandler(BaseHTTPRequestHandler):
    """Handle incoming proxy requests, optimize, and forward."""

    # Silence default request logging (we do our own)
    def log_message(self, format, *args):
        pass

    def _get_upstream_url(self, path: str) -> tuple[str | None, str]:
        """Get (upstream_url, provider) for a given path.

        If a custom upstream is set via --upstream, use it directly.
        Otherwise auto-detect from the request path.
        """
        if _custom_upstream:
            provider = _detect_provider(path)
            return f"{_custom_upstream.rstrip('/')}{path}", provider

        provider = _detect_provider(path)
        upstream = UPSTREAMS.get(provider)
        if provider == "unknown":
            logger.warning(f"Unknown provider for path: {path}")
            return None, provider
        return f"{upstream}{path}", provider

    def _get_provider(self, path: str) -> str:
        return _detect_provider(path)

    def _send_response(self, status: int, headers: dict, body: bytes):
        """Send a complete response to the client."""
        self.send_response(status)
        for k, v in headers.items():
            if k.lower() not in HOP_BY_HOP:
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_POST(self):
        """Handle POST request — optimize and forward."""
        start_time = time.time()

        # ── Read request body ──────────────────────────────────────
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

        # ── Detect provider and upstream ───────────────────────────
        provider = self._get_provider(self.path)
        upstream_url, upstream_provider = self._get_upstream_url(self.path)
        provider = provider if upstream_provider == "unknown" else upstream_provider

        if not upstream_url or provider == "unknown":
            self._send_error(502, f"Unknown API path: {self.path}")
            return

        # ── Optimize messages ──────────────────────────────────────
        messages = request_data.get("messages", [])
        if messages:
            # Use custom provider if set, else detect from path
            opt_provider = _custom_provider or provider
            try:
                optimized_messages = optimize(messages, provider=opt_provider)
                request_data["messages"] = optimized_messages

                # Also handle Anthropic-style top-level "system" field
                if "system" in request_data:
                    sys_msg = {"role": "system", "content": request_data["system"]}
                    opt_msgs = optimize([sys_msg] + messages, provider=opt_provider)
                    # Take the optimized system content from first message
                    if opt_msgs and opt_msgs[0].get("role") == "system":
                        request_data["system"] = opt_msgs[0]["content"]

                body = json.dumps(request_data).encode("utf-8")

                # Log optimization info
                orig_tokens = len(json.dumps(messages)) // 4
                opt_tokens = len(json.dumps(optimized_messages)) // 4
                logger.info(
                    f"OPTIMIZE | {provider} | "
                    f"{len(messages)}→{len(optimized_messages)} msgs | "
                    f"~{orig_tokens}→~{opt_tokens} tokens"
                )
            except Exception as e:
                logger.error(f"Optimization failed (forwarding raw): {e}")
                # Forward raw — better than failing

        # ── Detect streaming ───────────────────────────────────────
        is_streaming = request_data.get("stream", False)

        # ── Forward to upstream ────────────────────────────────────
        try:
            req = Request(
                upstream_url,
                data=body,
                headers=_forward_headers(dict(self.headers)),
                method="POST",
            )

            if is_streaming:
                # Streaming forward
                resp = urlopen(req, timeout=60)
                self.send_response(resp.status)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                elapsed = time.time() - start_time
                _forward_streaming(resp, self)
                elapsed_total = time.time() - start_time
                logger.info(
                    f"STREAM   | {provider} | {self.client_address[0]} | "
                    f"{elapsed_total:.1f}s"
                )
            else:
                # Non-streaming forward
                resp = urlopen(req, timeout=120)
                resp_body = resp.read()
                elapsed = time.time() - start_time

                logger.info(
                    f"FORWARD  | {provider} | {resp.status} | "
                    f"{len(resp_body)} bytes | {elapsed:.2f}s"
                )

                self._send_response(
                    resp.status,
                    dict(resp.headers),
                    resp_body,
                )

        except HTTPError as e:
            # API returned an error — forward it
            err_body = e.read()
            elapsed = time.time() - start_time
            logger.warning(
                f"UPSTREAM ERROR | {provider} | {e.code} | {elapsed:.2f}s"
            )
            self._send_response(e.code, dict(e.headers), err_body)

        except URLError as e:
            logger.error(f"Upstream connection failed: {e.reason}")
            self._send_error(502, f"Upstream connection failed: {e.reason}")

        except Exception as e:
            logger.error(f"Proxy error: {e}")
            self._send_error(500, f"Internal proxy error: {e}")

    def _send_error(self, status: int, message: str):
        """Send an error response."""
        body = json.dumps({"error": {"message": message}}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)


# ── Threaded Server ──────────────────────────────────────────────────────


class ThreadedProxyServer(ThreadingMixIn, HTTPServer):
    """HTTP proxy server with threading support."""
    allow_reuse_address = True
    daemon_threads = True


# ── Server Runner ────────────────────────────────────────────────────────


def run_proxy(host: str = "127.0.0.1", port: int = 9999,
              upstream: str | None = None,
              provider: str | None = None):
    """Start the proxy server.

    Args:
        host: Bind address (default: 127.0.0.1).
        port: Bind port (default: 9999).
        upstream: Optional custom upstream URL. If set, all requests are
                 forwarded here instead of auto-detecting from path.
        provider: Provider for optimization logic (deepseek, anthropic, openai).
                 Required when --upstream is set to ensure correct cache config.
    """
    set_custom_upstream(upstream)
    if provider:
        set_custom_provider(provider)
    server = ThreadedProxyServer((host, port), ProxyHandler)
    print(f"\n  popt proxy running on http://{host}:{port}")
    if upstream:
        print(f"  Upstream: {upstream}")
        print(f"  Optimizer: {provider or 'auto (detect from path)'}")
    else:
        print(f"  Set ANTHROPIC_BASE_URL=http://{host}:{port} to use")
    print(f"  Press Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Shutting down proxy...")
        server.shutdown()
