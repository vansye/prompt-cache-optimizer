"""Example: run a cache hit rate experiment.

This script demonstrates how to test whether the optimizer improves
cache hit rates. It requires valid API keys.

Usage:
    export ANTHROPIC_API_KEY=sk-...
    python examples/experiment.py

What it does:
    1. Sends the same prompt 3 times (no optimizations) → baseline
    2. Sends the optimized prompt 3 times → optimized
    3. Compares cache_read_input_tokens between the two
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from optimizer import optimize, diagnose, preview


# ── Test prompt data ────────────────────────────────────────────────────

RAW_MESSAGES = [
    {"role": "system", "content": "You are a code assistant. "
     "Help the user write clean, efficient Python code."},
    {"role": "user", "content": "Write a function to find all prime numbers up to N."},
    {"role": "assistant", "content": "Here's a Sieve of Eratosthenes implementation."},
    {"role": "user", "content": "Now make it async and handle large N (up to 10^8)."},
]

PROVIDER = os.environ.get("POPT_PROVIDER", "anthropic")
MODEL = os.environ.get("POPT_MODEL", "claude-sonnet-5-20250601")


# ── Helpers ─────────────────────────────────────────────────────────────


def send_messages(messages, label=""):
    """Send messages to the API and return usage metrics."""
    try:
        if PROVIDER == "anthropic":
            import anthropic
            client = anthropic.Anthropic()
            resp = client.messages.create(
                model=MODEL,
                max_tokens=100,
                messages=messages,
            )
            return diagnose(resp.model_dump())
        elif PROVIDER == "openai":
            from openai import OpenAI
            client = OpenAI()
            # For the experiment, inject messages into chat format
            system_content = None
            chat_msgs = []
            for m in messages:
                if m["role"] == "system":
                    system_content = m["content"]
                else:
                    chat_msgs.append(m)
            kwargs = {"model": MODEL, "messages": chat_msgs, "max_tokens": 100}
            if system_content:
                kwargs["messages"] = [
                    {"role": "system", "content": system_content}
                ] + chat_msgs
            resp = client.chat.completions.create(**kwargs)
            return diagnose(resp.model_dump())
        else:
            print(f"  Unknown provider: {PROVIDER}")
            return None
    except Exception as e:
        print(f"  API error: {e}")
        return None


def print_report(metrics_list, label):
    """Pretty-print a list of metrics."""
    total_read = sum(m.cache_read_input_tokens for m in metrics_list if m)
    total_input = sum(m.input_tokens for m in metrics_list if m)
    print(f"\n  [{label}]")
    for i, m in enumerate(metrics_list):
        if m:
            hit = "🟢 HIT" if m.cache_read_input_tokens > 0 else "⚪ MISS"
            print(f"    Request {i+1}: {hit} | "
                  f"read={m.cache_read_input_tokens} | "
                  f"creation={m.cache_creation_input_tokens} | "
                  f"input={m.input_tokens}")
    print(f"    Total cache_read: {total_read} / {total_input} input tokens")


# ── Main experiment ─────────────────────────────────────────────────────


def main():
    if PROVIDER == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("❌ Set ANTHROPIC_API_KEY to run the experiment.")
        print("   Or: POPT_PROVIDER=openai OPENAI_API_KEY=sk-... python experiment.py")
        sys.exit(1)

    print(f"=== Prompt Caching Experiment ===\n")
    print(f"Provider: {PROVIDER}")
    print(f"Model:    {MODEL}")
    print(f"Messages: {len(RAW_MESSAGES)}")

    # Preview what the optimizer will do
    p = preview(RAW_MESSAGES, provider=PROVIDER)
    print(f"\nOptimizer preview:")
    print(f"  {p['message_count']['before']} → {p['message_count']['after']} messages")
    print(f"  Roles: {p['role_order_before']} → {p['role_order_after']}")
    print(f"  Est. tokens: {p['estimated_tokens']['before']} → {p['estimated_tokens']['after']}")

    # ── Baseline: 3 rounds of raw messages ──────────────────────────
    print(f"\n{'='*50}")
    print("Phase 1: Baseline (raw messages, 3 rounds)")
    print(f"{'='*50}")
    baseline_metrics = []
    for round_num in range(3):
        print(f"\n  Round {round_num + 1}...")
        m = send_messages(RAW_MESSAGES, label="baseline")
        if m:
            baseline_metrics.append(m)
        time.sleep(0.5)  # Slight delay between requests

    # ── Optimized: 3 rounds of optimized messages ───────────────────
    print(f"\n{'='*50}")
    print("Phase 2: Optimized messages (3 rounds)")
    print(f"{'='*50}")
    OPTIMIZED = optimize(RAW_MESSAGES, provider=PROVIDER)
    optimized_metrics = []
    for round_num in range(3):
        print(f"\n  Round {round_num + 1}...")
        m = send_messages(OPTIMIZED, label="optimized")
        if m:
            optimized_metrics.append(m)
        time.sleep(0.5)

    # ── Report ──────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("RESULTS")
    print(f"{'='*50}")
    print_report(baseline_metrics, "BASELINE (raw)")
    print_report(optimized_metrics, "OPTIMIZED")

    if baseline_metrics and optimized_metrics:
        baseline_hits = sum(m.cache_read_input_tokens for m in baseline_metrics)
        optimized_hits = sum(m.cache_read_input_tokens for m in optimized_metrics)
        improvement = optimized_hits - baseline_hits
        if improvement > 0:
            print(f"\n✅ Improvement: +{improvement} cache-read tokens")
        elif improvement == 0:
            print(f"\nℹ️  No measurable difference (both at {baseline_hits})")
        else:
            print(f"\n⚠️  Optimized performed worse by {-improvement} tokens")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
