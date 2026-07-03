"""Rigorous cache hit rate experiment against DeepSeek API.

Validates the claims in ANALYSIS.md after fixing Aligner token estimation.
Methodology (controls warmup bias):
  For each scenario -> 6 sequential requests:
    Rounds 1-3: raw messages (round 1 = warmup, rounds 2-3 = stabilized raw)
    Rounds 4-6: optimized messages (round 4 = warmup, rounds 5-6 = stabilized opt)
"""

import json
import os
import sys
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from optimizer import optimize


# -- DeepSeek Config ----------------------------------------------------------

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = "deepseek-v4-flash"


# -- Test Scenarios -----------------------------------------------------------

def make_short_prompt():
    """~10 tokens -- well below the 128-token block size."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello briefly."},
    ]


def make_medium_prompt():
    """~200 tokens -- above one block, tests block-level caching."""
    return [
        {"role": "system", "content": (
            "You are a code assistant specialized in Python, JavaScript, and Rust. "
            "You follow best practices, write clean code, and provide thorough "
            "explanations. You always consider edge cases and error handling. "
            "When possible, you suggest alternative approaches. "
            "You format your responses with proper markdown."
        )},
        {"role": "user", "content": "Write a function to cache API responses with TTL support."},
        {"role": "assistant", "content": (
            "Here's a cache utility with TTL support using Python's functools:\n\n"
            "```python\nfrom functools import lru_cache\nfrom datetime import datetime, timedelta\n"
            "class TTLCache:\n    def __init__(self, ttl_seconds=300):\n        self.cache = {}\n"
            "        self.ttl = timedelta(seconds=ttl_seconds)\n\n    def get(self, key):\n"
            "        entry = self.cache.get(key)\n        if entry and datetime.now() - entry['time'] < self.ttl:\n"
            "            return entry['value']\n        return None\n```"
        )},
        {"role": "user", "content": "Now make it async-safe with asyncio locks."},
    ]


# -- DeepSeek API Client ------------------------------------------------------

def deepseek_chat(messages: list[dict], round_num: int, label: str):
    """Send a chat completion request and return parsed response."""
    import urllib.request
    import urllib.error

    # Build the request body as the API will see it
    body = json.dumps({
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "max_tokens": 50,
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{DEEPSEEK_BASE}/v1/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"  !! Round {round_num} [{label}] HTTP {e.code}: {e.read().decode()[:200]}")
        return None
    except Exception as e:
        print(f"  !! Round {round_num} [{label}] Error: {e}")
        return None

    return raw


def parse_cache(raw: dict, round_num: int, label: str) -> Optional[dict]:
    """Parse cache metrics from DeepSeek response."""
    if not raw:
        return None
    usage = raw.get("usage", {})
    input_tokens = usage.get("prompt_tokens", 0)

    # DeepSeek-specific top-level fields
    prompt_cache_hit = usage.get("prompt_cache_hit_tokens", 0)
    prompt_cache_miss = usage.get("prompt_cache_miss_tokens", 0)

    # OpenAI-compatible fallback
    prompt_details = usage.get("prompt_tokens_details", {}) or {}
    cached = prompt_cache_hit or prompt_details.get("cached_tokens", 0) or 0

    return {
        "round": round_num,
        "label": label,
        "input_tokens": input_tokens,
        "hit_tokens": prompt_cache_hit,
        "miss_tokens": prompt_cache_miss,
        "cached": cached,
        "is_hit": cached > 0,
    }


# -- Experiment Runner --------------------------------------------------------

def run_scenario(name: str, messages_fn):
    """Run 3 raw + 3 optimized requests, report per-round + stabilized."""
    print(f"\n{'='*65}")
    print(f"  SCENARIO: {name}")
    print(f"{'='*65}")

    raw_msgs = messages_fn()
    opt_msgs = optimize(raw_msgs, provider="deepseek")

    # Show structural diff
    raw_len = sum(len(str(m.get("content", ""))) for m in raw_msgs)
    opt_len = sum(len(str(m.get("content", ""))) for m in opt_msgs)
    print(f"  Raw:        {len(raw_msgs)} msgs, ~{raw_len//4} est. tokens")
    print(f"  Optimized:  {len(opt_msgs)} msgs, ~{opt_len//4} est. tokens")
    delta = opt_len - raw_len
    if delta:
        print(f"  Size delta: {delta:+d} chars ({delta//4:+d} est. tokens)")

    # Phase 1: Raw
    print(f"\n  -- RAW (rounds 1-3) --")
    raw_results = []
    for i in range(3):
        time.sleep(1.5)
        r = parse_cache(deepseek_chat(raw_msgs, i + 1, "raw"), i + 1, "raw")
        if r:
            raw_results.append(r)
            icon = "HIT" if r["is_hit"] else "MISS"
            ratio = r["cached"] / max(r["input_tokens"], 1) * 100
            print(f"    Round {i+1}: {icon:>4} | input={r['input_tokens']:>4} | "
                  f"hit={r['hit_tokens']:>4} | miss={r['miss_tokens']:>4} | "
                  f"cached={r['cached']:>4} ({ratio:5.1f}%)")

    # Phase 2: Optimized
    print(f"  -- OPTIMIZED (rounds 4-6) --")
    opt_results = []
    for i in range(3):
        time.sleep(1.5)
        r = parse_cache(deepseek_chat(opt_msgs, i + 1, "opt"), i + 4, "opt")
        if r:
            opt_results.append(r)
            icon = "HIT" if r["is_hit"] else "MISS"
            ratio = r["cached"] / max(r["input_tokens"], 1) * 100
            print(f"    Round {i+4}: {icon:>4} | input={r['input_tokens']:>4} | "
                  f"hit={r['hit_tokens']:>4} | miss={r['miss_tokens']:>4} | "
                  f"cached={r['cached']:>4} ({ratio:5.1f}%)")

    return raw_results, opt_results


def summarize(results_raw, results_opt, name: str):
    """Show stabilized metrics (skip round 1 of each phase)."""
    print(f"\n  -- SUMMARY: {name} --")

    def stable(results):
        if not results or len(results) < 2:
            return None
        s = results[1:]  # skip warmup
        total_in = sum(r["input_tokens"] for r in s)
        total_cached = sum(r["cached"] for r in s)
        hits = sum(1 for r in s if r["is_hit"])
        return {
            "n": len(s),
            "total_in": total_in,
            "total_cached": total_cached,
            "hits": hits,
            "ratio": total_cached / max(total_in, 1) * 100,
        }

    rs = stable(results_raw)
    os_ = stable(results_opt)

    if not rs or not os_:
        print("    Insufficient data (API errors)")
        return None

    print(f"    {'':>20} {'RAW (r2-3)':>15} {'OPT (r5-6)':>15}")
    print(f"    {'-'*20} {'-'*15} {'-'*15}")
    print(f"    {'Input tokens':>20} {rs['total_in']:>15} {os_['total_in']:>15}")
    print(f"    {'Cache hits':>20} {rs['hits']:>15}/{rs['n']:>9} {os_['hits']:>15}/{os_['n']:>9}")
    print(f"    {'Hit ratio':>20} {rs['ratio']:>14.1f}% {os_['ratio']:>14.1f}%")
    print(f"    {'Cached tokens':>20} {rs['total_cached']:>15} {os_['total_cached']:>15}")

    delta = os_["total_cached"] - rs["total_cached"]
    if delta > 0:
        print(f"\n    => OPTIMIZED: +{delta} cached tokens (+{os_['ratio']-rs['ratio']:.1f}pp)")
    elif delta == 0 and rs["ratio"] > 0:
        print(f"\n    => No difference (both cache at same rate)")
    elif delta == 0 and rs["ratio"] == 0:
        print(f"\n    => BOTH MISS (prompt too short even after padding)")
    else:
        print(f"\n    => RAW better by {-delta} tokens (unexpected)")
    print()

    return {"raw": rs["ratio"], "opt": os_["ratio"]}


def main():
    print("=" * 65)
    print("  DeepSeek Cache Hit Rate Experiment (v2)")
    print("  Model: deepseek-v4-flash")
    print("  Methodology: 3 raw -> 3 opt per scenario")
    print("  Stabilized: rounds 2-3 (raw), rounds 5-6 (opt)")
    print("  Fix: tiktoken estimation + 1.3x safety margin")
    print("=" * 65)

    results = {}

    r, o = run_scenario("SHORT PROMPT (~10t)", make_short_prompt)
    results["short"] = summarize(r, o, "SHORT PROMPT")

    r, o = run_scenario("MEDIUM PROMPT (~200t)", make_medium_prompt)
    results["medium"] = summarize(r, o, "MEDIUM PROMPT")

    # Final verdict
    print(f"\n{'='*65}")
    print("  VERDICT")
    print(f"{'='*65}")
    for name, key in [("Short prompt (~10t)", "short"),
                       ("Medium prompt (~200t)", "medium")]:
        if results.get(key):
            r = results[key]
            arrow = "OPT WINS" if r["opt"] > r["raw"] else "TIE" if r["opt"] == r["raw"] else "RAW WINS"
            print(f"  {name:<25} raw={r['raw']:>5.1f}%  ->  opt={r['opt']:>5.1f}%  [{arrow}]")
    print()


if __name__ == "__main__":
    main()
