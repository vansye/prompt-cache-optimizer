"""CLI entry point for popt.

Usage:
    popt preview <file>                  — Show optimization diff (no API call)
    popt diagnose <a.json> <b.json>      — Diagnose why two requests differ
    popt proxy [--port PORT]             — Start local transparent proxy
    popt stats [<file>]                  — Analyze cache metrics from log
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


def cmd_proxy(args):
    """Start the local transparent proxy."""
    from cli.proxy import run_proxy
    run_proxy(host=args.host, port=args.port)


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

    report = HitRatioCalculator.calculate(cache_entries)

    print(f"\n  popt stats — {report.total_requests} requests")
    print(f"  {'='*40}")
    print(f"  Total input tokens:   {report.total_input_tokens:,}")
    print(f"  Total cache created:  {report.total_cache_creation:,}")
    print(f"  Total cache read:     {report.total_cache_read:,}")
    print(f"  Total output tokens:  {report.total_output_tokens:,}")
    print(f"  {'─'*40}")
    print(f"  Hit ratio:            {report.hit_ratio:.1%}")
    print(f"  Savings ratio:        {report.savings_ratio:.1%}")
    print(f"  Est. cost saved:      ${report.estimated_cost_saved:.4f}")


# ── Argument parser ─────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="popt",
        description="Prompt structure optimizer — maximize LLM cache hit rates",
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
        "proxy", help="Start local transparent proxy")
    p_proxy.add_argument(
        "--host", default="127.0.0.1",
        help="Bind address (default: 127.0.0.1)")
    p_proxy.add_argument(
        "--port", "-p", type=int, default=9999,
        help="Bind port (default: 9999)")

    # stats
    p_stats = sub.add_parser(
        "stats", help="Analyze cache metrics from log")
    p_stats.add_argument(
        "file", nargs="?", help="JSON lines file (one response per line)")

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
    elif args.command == "stats":
        cmd_stats(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
