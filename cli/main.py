"""CLI entry point for popt.

Usage:
    popt preview <file>                  -- Show optimization diff (no API call)
    popt diagnose <a.json> <b.json>      -- Diagnose why two requests differ
    popt proxy [--port PORT]             -- Start local transparent proxy
    popt stats [<file>]                  -- Analyze cache metrics from log
"""

import argparse
import json
import sys
import os

# ── Ensure optimizer is importable ──────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cmd_preview(args):
    """Show optimization preview without calling any API."""
    try:
        with open(args.file, "r", encoding="utf-8") as f:
            messages = json.load(f)
    except FileNotFoundError:
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON in {args.file}: {e}", file=sys.stderr)
        sys.exit(1)

    from optimizer import preview

    provider = args.provider or "anthropic"
    result = preview(messages, provider=provider)

    # Print human-readable report
    print(f"\n  popt preview - {provider}")
    print(f"  {'='*40}")
    print(f"  Messages:    {result['message_count']['before']} -> "
          f"{result['message_count']['after']}")
    if result['message_count']['separators_added']:
        print(f"  Separators:  +{result['message_count']['separators_added']} inserted")
    print(f"  Role order:  {' -> '.join(result['role_order_before'][:5])}")
    print(f"               {' -> '.join(result['role_order_after'][:5])}")
    print(f"  Est. tokens: {result['estimated_tokens']['before']} -> "
          f"{result['estimated_tokens']['after']}")
    print(f"  Cache threshold: {result['cache_threshold']} tokens")
    print(f"  Meets threshold: {'YES' if result['meets_threshold'] else 'NO'}")

    # Show prefix shape hashes
    from optimizer.diagnoser import capture_shape
    shape = capture_shape(messages)
    print(f"  {'-'*40}")
    print(f"  Prefix shape hashes:")
    print(f"    system:           {shape.system_hash or '(none)'}")
    print(f"    role_sequence:    {shape.role_sequence_hash}")
    print(f"    content_prefix:   {shape.content_prefix_hash}")
    print(f"    full_prefix:      {shape.full_prefix_hash}")

    if args.verbose:
        print(f"\n  Full result:")
        print(json.dumps(result, indent=2))

    # Also show the actual optimized messages if requested
    if args.show:
        from optimizer import optimize
        optimized = optimize(messages, provider=provider)
        # Strip internal markers for clean display
        clean = []
        for m in optimized:
            clean.append({k: v for k, v in m.items() if not k.startswith("_")})
        print(f"\n  Optimized messages:")
        print(json.dumps(clean, indent=2, ensure_ascii=False))


def _load_messages(path: str, label: str) -> list[dict]:
    """Load messages from a JSON file with error handling."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Accept both a plain list and {"messages": [...]} wrapper
        if isinstance(data, dict) and "messages" in data:
            data = data["messages"]
        if not isinstance(data, list):
            print(f"Error: {label} must contain a JSON array of messages, "
                  f"got {type(data).__name__}", file=sys.stderr)
            sys.exit(1)
        return data
    except FileNotFoundError:
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON in {path}: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_diagnose(args):
    """Diagnose why two requests produce different cache behavior."""
    msgs_a = _load_messages(args.file_a, "request_a")
    msgs_b = _load_messages(args.file_b, "request_b")

    from optimizer.config import get_config
    from optimizer.diagnoser import (
        capture_shape, compare_shapes, PrefixDiffer,
    )

    provider = args.provider or "deepseek"
    cfg = get_config(provider)

    print(f"\n  popt diagnose - {provider}")
    print(f"  {'='*50}")
    print(f"  Request A: {args.file_a} ({len(msgs_a)} messages)")
    print(f"  Request B: {args.file_b} ({len(msgs_b)} messages)")
    print(f"  Cache block: 128 tokens  |  Threshold: {cfg.cache_threshold}t")

    # -- Shape comparison --
    shape_a = capture_shape(msgs_a)
    shape_b = capture_shape(msgs_b)
    shape_diff = compare_shapes(shape_a, shape_b)

    print(f"\n  -- Shape comparison --")
    for label, ha, hb in [
        ("System",         shape_a.system_hash,      shape_b.system_hash),
        ("Role sequence",  shape_a.role_sequence_hash, shape_b.role_sequence_hash),
        ("Content prefix", shape_a.content_prefix_hash, shape_b.content_prefix_hash),
    ]:
        if not ha and not hb:
            icon = "-"
        elif ha == hb:
            icon = "="
        else:
            icon = "X"
        print(f"    {icon} {label:<20} {ha or '(none)':>10}  {hb or '(none)':>10}")

    if shape_diff.reasons:
        print(f"    Changed: {', '.join(shape_diff.reasons)}")
    else:
        print(f"    No prefix changes detected.")

    # -- Token-precise diff --
    pdiff = PrefixDiffer.diff(msgs_a, msgs_b)

    print(f"\n  -- First difference --")
    if pdiff is None:
        print(f"    No differences found -- requests are identical.")
    else:
        sev_icon = {"CRITICAL": "!!", "MAJOR": "!", "MINOR": "?", "INFO": "i"}
        icon = sev_icon.get(pdiff.severity, "?")
        print(f"    Position:   ~token {pdiff.position}")
        print(f"    Block:      {pdiff.block}  {icon} {pdiff.severity}")
        print(f"    Message:    [{pdiff.msg_index}] role={pdiff.msg_role}")
        print(f"    Field:      {pdiff.field}")
        print(f"    Snippet A:  {pdiff.snippet_a[:80]}")
        print(f"    Snippet B:  {pdiff.snippet_b[:80]}")
        print(f"\n    Suggestion: {pdiff.suggestion}")

    # -- Token estimate comparison --
    def _est(text):
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(str(text)))
        except ImportError:
            return len(str(text)) // 6

    total_a = sum(_est(m.get("content", "")) for m in msgs_a)
    total_b = sum(_est(m.get("content", "")) for m in msgs_b)
    blocks_a = total_a // 128
    blocks_b = total_b // 128

    print(f"\n  -- Cache estimate --")
    print(f"    {'':>20} {'Request A':>12} {'Request B':>12}")
    print(f"    {'-'*20} {'-'*12} {'-'*12}")
    print(f"    {'Est. tokens':>20} {total_a:>12} {total_b:>12}")
    print(f"    {'Cache blocks':>20} {blocks_a:>12} {blocks_b:>12}")
    print(f"    {'Below threshold':>20} "
          f"{'YES' if total_a < cfg.cache_threshold else '  no':>12} "
          f"{'YES' if total_b < cfg.cache_threshold else '  no':>12}")

    if args.verbose:
        print(f"\n  -- Full shape data --")
        print(f"    shape_a: {shape_a}")
        print(f"    shape_b: {shape_b}")
        if pdiff:
            print(f"    diff:    {pdiff}")

    print()


def _resolve_upstream_provider(
    cli_upstream: str, cli_provider: str, cli_model: str,
) -> tuple[str, str, str]:
    """Resolve upstream URL, provider name, and API format from all sources.

    Priority chain (highest first):
      1. CLI flags (--upstream, --provider, --model)
      2. Environment variables
      3. .poptimerc config file
      4. Registry inference from URL or model name
      5. Built-in defaults

    Returns:
        Tuple of (upstream_url, provider_name, api_format).
        ``api_format`` is ``"openai"`` or ``"anthropic"``.
    """
    from cli.rcconfig import load_config
    from optimizer.config import resolve_model, infer_provider_from_url

    # ── Step 1: Load .poptimerc ───────────────────────────────────
    rcfg = load_config()

    # ── Step 2: Resolve model name (if given) ─────────────────────
    model_sources = [
        cli_model,
        os.environ.get("POPT_MODEL", ""),
        rcfg.model,
    ]
    model_name = next((m for m in model_sources if m), "")
    model_info = resolve_model(model_name) if model_name else None

    # ── Step 3: Upstream detection ─────────────────────────────────
    upstream = (
        cli_upstream
        or (model_info.base_url if model_info else None)
        or os.environ.get("POPT_UPSTREAM")
        or os.environ.get("ANTHROPIC_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or rcfg.upstream
        or ""
    )

    # ── Step 4: Provider detection ─────────────────────────────────
    provider = (
        cli_provider
        or (model_info.provider if model_info else None)
        or os.environ.get("POPT_PROVIDER", "")
        or (infer_provider_from_url(upstream) if upstream else "")
        or rcfg.provider
        or ""
    )

    # ── Step 5: API format ─────────────────────────────────────────
    # If model_info is available, use its api_format.
    # Otherwise infer from provider name.
    api_format = "openai"  # default
    if model_info:
        api_format = model_info.api_format
    elif provider:
        from optimizer.config import get_config
        cfg = get_config(provider)
        api_format = getattr(cfg, "api_format", "openai")

    return upstream, provider, api_format


def cmd_proxy(args):
    """Start the local transparent proxy."""
    upstream, provider, api_format = _resolve_upstream_provider(
        args.upstream or "", args.provider or "", args.model or "",
    )
    from cli.proxy import run_proxy
    run_proxy(host=args.host, port=args.port,
              upstream=upstream or None,
              provider=provider or None,
              api_format=api_format)


def cmd_run(args):
    """Run a command with the popt proxy automatically in front.

    Starts the proxy on a random port, sets the relevant env vars
    (OPENAI_BASE_URL, ANTHROPIC_BASE_URL) to point to it, then
    executes the target command.  When the command exits the proxy
    is shut down automatically.

    Usage:
        popt run -- python my_script.py
        popt run --model deepseek-v4-flash -- claude
    """
    import subprocess
    from cli.proxy import run_proxy, ProxyHandler, ThreadedProxyServer
    from cli.proxy import set_custom_upstream, set_custom_provider

    # Pick a free port
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        proxy_port = s.getsockname()[1]

    # Resolve upstream + provider + api_format via new detection chain
    upstream, provider, api_format = _resolve_upstream_provider(
        args.upstream or "", args.provider or "", args.model or "",
    )

    # Configure the proxy with the detected upstream so it actually forwards
    # to the right place (not the default api.anthropic.com / api.openai.com).
    set_custom_upstream(upstream or None)
    if provider:
        set_custom_provider(provider)

    # Start the proxy
    from threading import Thread
    server = ThreadedProxyServer(("127.0.0.1", proxy_port), ProxyHandler)
    t = Thread(target=server.serve_forever, daemon=True)
    t.start()

    # Set env vars so the child process uses the proxy
    # We set the correct env var based on api_format, which is more
    # robust than guessing from the URL string.
    child_env = os.environ.copy()
    if api_format == "openai":
        child_env["OPENAI_BASE_URL"] = f"http://127.0.0.1:{proxy_port}/v1"
    if api_format == "anthropic":
        child_env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{proxy_port}"
    # Also set the other one on a best-effort basis from the upstream URL
    if upstream:
        u = upstream.lower()
        if "/v1" in u or "openai" in u:
            child_env["OPENAI_BASE_URL"] = f"http://127.0.0.1:{proxy_port}/v1"
        if "/v1/messages" in u or "anthropic" in u:
            child_env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{proxy_port}"

    # Strip leading '--' if argparse REMAINDER collected it
    cmd = args.cmd_args[:]
    while cmd and cmd[0] == "--":
        cmd.pop(0)

    # ── Build a user-friendly status banner ─────────────────────────
    banner = []
    banner.append(f"  popt run -- proxy on :{proxy_port}")
    banner.append(f"  {'='*45}")

    if upstream and provider:
        from optimizer.config import get_config
        cfg = get_config(provider)
        block_size = cfg.cache_threshold
        banner.append(f"  [OK] Upstream: {upstream}")
        banner.append(f"  [OK] Provider: {provider} ({block_size}t cache blocks, {api_format} API)")
        # Show what env vars the child will see
        if "ANTHROPIC_BASE_URL" in child_env and child_env["ANTHROPIC_BASE_URL"].startswith("http://127.0.0.1"):
            banner.append(f"  -> ANTHROPIC_BASE_URL = http://127.0.0.1:{proxy_port}")
        if "OPENAI_BASE_URL" in child_env and child_env["OPENAI_BASE_URL"].startswith("http://127.0.0.1"):
            banner.append(f"  -> OPENAI_BASE_URL     = http://127.0.0.1:{proxy_port}/v1")
        if args.model:
            banner.append(f"  Model: {args.model}")
    elif upstream:
        banner.append(f"  Upstream: {upstream}")
        banner.append(f"  [!] No provider detected -- set POPT_PROVIDER or --provider")
    else:
        banner.append(f"  [!] No upstream detected -- proxying only, NOT optimizing")
        banner.append(f"")
        banner.append(f"  To enable optimization, set an env var or --model:")
        banner.append(f"    $env:ANTHROPIC_BASE_URL = 'https://api.deepseek.com/anthropic'")
        banner.append(f"    $env:OPENAI_BASE_URL    = 'https://api.openai.com'")
        banner.append(f"    --model deepseek-v4-flash")

    banner.append(f"  Command: {' '.join(cmd)}")
    banner.append(f"  {'='*45}\n")

    print("\n".join(banner), flush=True)

    try:
        # On Windows, use shell=True so batch files (*.cmd, *.bat) resolve
        # via PATHEXT, just like typing in PowerShell/cmd.
        use_shell = sys.platform == "win32"
        result = subprocess.run(cmd, env=child_env, shell=use_shell)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n  Interrupted, shutting down...")
    finally:
        server.shutdown()


def cmd_stats(args):
    """Analyze cache metrics from a log file or stdin."""
    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                entries = [json.loads(line) for line in f if line.strip()]
        except FileNotFoundError:
            print(f"Error: file not found: {args.file}", file=sys.stderr)
            sys.exit(1)
    else:
        # Read from stdin
        entries = [json.loads(line) for line in sys.stdin if line.strip()]

    if not entries:
        print("No data to analyze.")
        return

    from optimizer.diagnoser import (
        parse_usage, CacheEntry, HitRatioCalculator,
    )

    cache_entries = []
    for i, entry in enumerate(entries):
        metrics = parse_usage(entry.get("response", entry))
        is_hit = metrics.cache_read_input_tokens > 0
        cache_entries.append(CacheEntry(i, metrics, is_hit))

    # Per-provider pricing (falls back to Anthropic USD estimate)
    pricing = None
    if getattr(args, "provider", ""):
        from optimizer.config import get_config
        pricing = get_config(args.provider).pricing
        if pricing is None:
            print(f"  (no pricing configured for '{args.provider}', "
                  f"using default Anthropic USD estimate)")

    report = HitRatioCalculator.calculate(cache_entries, pricing=pricing)

    print(f"\n  popt stats -- {report.total_requests} requests")
    print(f"  {'='*40}")
    print(f"  Total input tokens:   {report.total_input_tokens:,}")
    print(f"  Total cache created:  {report.total_cache_creation:,}")
    print(f"  Total cache read:     {report.total_cache_read:,}")
    print(f"  Total output tokens:  {report.total_output_tokens:,}")
    print(f"  {'─'*40}")
    print(f"  Hit ratio:            {report.hit_ratio:.1%}")
    print(f"  Savings ratio:        {report.savings_ratio:.1%}")
    print(f"  Est. cost saved:      {report.estimated_cost_saved:.4f} "
          f"{report.currency}")


def cmd_gui(args):
    """Start the web GUI + proxy server."""
    from cli.gui import run_gui
    run_gui(host=args.host, port=args.port, model=args.model or "",
            open_browser=not args.no_browser)


# ── Argument parser ─────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="popt",
        description="Prompt structure optimizer -- maximize LLM cache hit rates",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # preview
    p_preview = sub.add_parser(
        "preview", help="Show optimization diff (no API call)")
    p_preview.add_argument("file", help="JSON file with messages")
    p_preview.add_argument(
        "--provider", "-p", default="",
        help="API provider (anthropic, openai, deepseek, ...)")
    p_preview.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show full preview data")
    p_preview.add_argument(
        "--show", "-s", action="store_true",
        help="Show the actual optimized messages")

    # diagnose
    p_diag = sub.add_parser(
        "diagnose", help="Diagnose why two requests differ")
    p_diag.add_argument("file_a", help="First request JSON (e.g. the one that established cache)")
    p_diag.add_argument("file_b", help="Second request JSON (e.g. the one that missed)")
    p_diag.add_argument(
        "--provider", "-p", default="deepseek",
        help="Provider for threshold info (default: deepseek)")
    p_diag.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show full diff detail")

    # proxy
    p_proxy = sub.add_parser(
        "proxy", help="Start local transparent proxy (auto-detect upstream from env)")
    p_proxy.add_argument(
        "--host", default="127.0.0.1",
        help="Bind address (default: 127.0.0.1)")
    p_proxy.add_argument(
        "--port", "-p", type=int, default=9999,
        help="Bind port (default: 9999)")
    p_proxy.add_argument(
        "--upstream", "-u", default="",
        help="Upstream API base URL (default: auto-detect from ANTHROPIC_BASE_URL / OPENAI_BASE_URL / POPT_UPSTREAM)")
    p_proxy.add_argument(
        "--provider", default="",
        help="Optimization logic: deepseek (128t blocks), anthropic (1024t), openai. "
             "Auto-detected from upstream URL if not set.")
    p_proxy.add_argument(
        "--model", "-m", default="",
        help="Model name (e.g. deepseek-v4-flash, gpt-4o). Auto-configures "
             "upstream and provider from the built-in registry.")

    # run
    p_run = sub.add_parser(
        "run", help="Start proxy + run any command (auto-detects upstream from env)")
    p_run.add_argument(
        "--upstream", "-u", default="",
        help="Upstream API base URL (default: auto-detect from ANTHROPIC_BASE_URL / OPENAI_BASE_URL / POPT_UPSTREAM)")
    p_run.add_argument(
        "--provider", default="",
        help="Optimization logic: deepseek (128t blocks), anthropic (1024t), openai")
    p_run.add_argument(
        "--model", "-m", default="",
        help="Model name (e.g. deepseek-v4-flash, gpt-4o). Auto-configures "
             "upstream and provider from the built-in registry.")
    p_run.add_argument(
        "cmd_args", nargs=argparse.REMAINDER,
        help="Command to run. Prefix with -- to separate from popt args. "
             "Examples: 'run -- claude', 'run -- python script.py'")

    # stats
    p_stats = sub.add_parser(
        "stats", help="Analyze cache metrics from log")
    p_stats.add_argument(
        "file", nargs="?", help="JSON lines file (one response per line)")
    p_stats.add_argument(
        "--provider", default="",
        help="Provider name for pricing from providers.json (e.g. deepseek)")

    # gui
    p_gui = sub.add_parser(
        "gui", help="Start web GUI + proxy server (same port)")
    p_gui.add_argument(
        "--host", default="127.0.0.1",
        help="Bind address (default: 127.0.0.1)")
    p_gui.add_argument(
        "--port", "-p", type=int, default=6123,
        help="Bind port (default: 6123)")
    p_gui.add_argument(
        "--model", "-m", default="",
        help="Model name (e.g. deepseek-v4-flash, gpt-4o)")
    p_gui.add_argument(
        "--no-browser", action="store_true",
        help="Do not auto-open browser on start")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "preview":
        cmd_preview(args)
    elif args.command == "diagnose":
        cmd_diagnose(args)
    elif args.command == "proxy":
        cmd_proxy(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "gui":
        cmd_gui(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
