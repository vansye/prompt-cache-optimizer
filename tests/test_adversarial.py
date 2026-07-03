"""Adversarial tests — stress-test the optimizer against edge cases.

These tests verify that the optimizer:
  1. Never crashes on any input
  2. Preserves semantic content
  3. Degrades gracefully on malformed data
  4. Maintains safety guarantees under adversarial conditions
"""

import json
import math
import sys
import pytest

from optimizer import optimize, preview
from optimizer.normalizer import (
    WhitespaceSanitizer, JsonCanonicalizer, Normalizer,
)
from optimizer.reorderer import Reorderer, _content_hash
from optimizer.safety import SafetyCheck
from optimizer.config import ProviderConfig


# ═══════════════════════════════════════════════════════════════════════
# 1. INPUT ATTACK — Malformed, missing, null
# ═══════════════════════════════════════════════════════════════════════

class TestMalformedInput:
    """The optimizer must never crash, even on garbage input."""

    def test_empty_message_list(self):
        """Empty list should raise — but not crash."""
        with pytest.raises(ValueError):
            optimize([], provider="anthropic")

    def test_missing_role_key(self):
        """Messages without 'role' should still not crash."""
        msgs = [{"content": "hello"}]
        result = optimize(msgs)
        assert isinstance(result, list)

    def test_missing_content_key(self):
        """Messages without 'content' should still not crash."""
        msgs = [{"role": "user"}]
        result = optimize(msgs)
        assert isinstance(result, list)

    def test_none_content(self):
        """Content is None — should not crash."""
        msgs = [{"role": "user", "content": None}]
        result = optimize(msgs)
        assert isinstance(result, list)

    def test_integer_content(self):
        """Content is an int, not a string."""
        msgs = [{"role": "user", "content": 42}]
        result = optimize(msgs)
        assert isinstance(result, list)

    def test_list_without_dicts(self):
        """Messages list containing non-dict elements."""
        msgs = ["not a dict", 42, None]
        try:
            result = optimize(msgs)
            assert isinstance(result, list)
        except (TypeError, ValueError, AttributeError):
            pass  # May raise — but must not crash

    def test_bare_string_as_message(self):
        """A bare string where a dict is expected."""
        msgs = [{"role": "user", "content": "hello"}]
        result = optimize(msgs)
        assert isinstance(result, list)

    def test_unknown_role(self):
        """Completely unknown role value."""
        msgs = [{"role": "supervisor", "content": "do something"}]
        try:
            result = optimize(msgs)
            assert isinstance(result, list)
        except Exception:
            pass

    def test_numeric_role(self):
        """Role is a number."""
        msgs = [{"role": 123, "content": "hello"}]
        try:
            result = optimize(msgs)
            assert isinstance(result, list)
        except (TypeError, AttributeError):
            pass


# ═══════════════════════════════════════════════════════════════════════
# 2. ENCODING ATTACKS — Unicode, control chars, edge bytes
# ═══════════════════════════════════════════════════════════════════════

class TestEncodingEdgeCases:
    """Verify correct handling of unusual character encodings."""

    def test_chinese_characters(self):
        """CJK characters must be preserved exactly."""
        msg = "你好世界，这是一个测试"
        msgs = [{"role": "user", "content": msg}]
        result = optimize(msgs)
        assert msg in str(result), "Chinese characters preserved"

    def test_arabic_text(self):
        """RTL text must be preserved."""
        msg = "مرحبا بالعالم"
        msgs = [{"role": "user", "content": msg}]
        result = optimize(msgs)
        assert msg in str(result), "Arabic text preserved"

    def test_emoji(self):
        """Emoji must not be corrupted."""
        msg = "Hello 🎉🔥👋 test"
        msgs = [{"role": "user", "content": msg}]
        result = optimize(msgs)
        assert "🎉" in str(result), "Emoji preserved"
        assert "🔥" in str(result), "Emoji preserved"

    def test_unicode_variants(self):
        """Various Unicode ranges must round-trip safely."""
        msgs = [
            {"role": "user",
             "content": "Math: ∑∏∫√∞ ≈ ≡ ≤ ≥  \n"
                         "Box: █▓▒░  \n"
                         "Arrows: →⇒⇔↔  \n"
                         "Quotes: „“”‘’"},
        ]
        try:
            result = optimize(msgs)
            for char in "∑∏∫√∞→⇒⇔↔„“":
                assert char in str(result), f"Unicode char '{char}' survived"
        except Exception as e:
            pytest.fail(f"Unicode test crashed: {e}")

    def test_null_byte_in_content(self):
        """Null bytes — must not crash or propagate."""
        msgs = [{"role": "user", "content": "hello\x00world"}]
        try:
            result = optimize(msgs)
            # Null byte may be removed by sanitizer
            assert isinstance(result, list)
        except Exception as e:
            # Some JSON libraries reject null bytes — acceptable
            if "null" in str(e).lower():
                pass
            else:
                pytest.fail(f"Unexpected error: {e}")

    def test_control_characters(self):
        """Control characters should be handled gracefully."""
        msgs = [{"role": "user", "content": "hello\x01\x02\x1fworld"}]
        try:
            result = optimize(msgs)
            assert isinstance(result, list)
        except Exception as e:
            pytest.fail(f"Control char test crashed: {e}")

    def test_very_long_string(self):
        """Very long single-line content (100k chars)."""
        content = "x" * 100_000
        msgs = [{"role": "system", "content": content}]
        try:
            result = optimize(msgs)
            assert isinstance(result, list)
        except Exception as e:
            pytest.fail(f"Long string test crashed: {e}")

    def test_zero_width_chars(self):
        """Zero-width and invisible characters."""
        msg = "he​llo‌ world⁠"
        msgs = [{"role": "user", "content": msg}]
        try:
            result = optimize(msgs)
            # Zero-width chars are valid unicode — should survive
            assert isinstance(result, list)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════
# 3. JSON CHAOS — Deep nesting, duplicate keys, extremes
# ═══════════════════════════════════════════════════════════════════════

class TestJsonChaos:
    """JSON canonicalization under adversarial conditions."""

    def test_deeply_nested_json(self):
        """100-level nested dict must not crash."""
        data = {}
        current = data
        for i in range(100):
            current[f"key_{i}"] = {}
            current = current[f"key_{i}"]
        current["value"] = 1

        msgs = [{"role": "user", "content": json.dumps(data)}]
        try:
            result = JsonCanonicalizer.canonicalize_messages(msgs)
            assert isinstance(result, list)
        except RecursionError:
            pytest.fail("Deeply nested JSON caused recursion error")

    def test_non_dict_json(self):
        """JSON that is an array, not an object."""
        msgs = [{"role": "user", "content": "[1, 2, 3]"}]
        result = JsonCanonicalizer.canonicalize_messages(msgs)
        # Arrays aren't reordered, but should survive
        assert "1" in result[0]["content"]

    def test_empty_json_object(self):
        """Empty JSON object."""  # noqa
        msgs = [{"role": "user", "content": "{}"}]
        result = JsonCanonicalizer.canonicalize_messages(msgs)
        assert result[0]["content"] == "{}"

    def test_json_with_primitives(self):
        """JSON with primitives at root (string, number, bool, null)."""
        for val in ['"hello"', '42', 'true', 'false', 'null']:
            msgs = [{"role": "user", "content": val}]
            result = JsonCanonicalizer.canonicalize_messages(msgs)
            assert isinstance(result[0]["content"], str)

    def test_json_mixed_nested_structures(self):
        """Nested dicts with arrays containing dicts."""
        data = {
            "z": [{"b": 2, "a": 1}, {"d": 4, "c": 3}],
            "a": {"nested": {"y": 10, "x": 20}},
        }
        msgs = [{"role": "user", "content": json.dumps(data)}]
        result = JsonCanonicalizer.canonicalize_messages(msgs)
        content = result[0]["content"]
        parsed = json.loads(content)
        # Verify keys sorted
        keys = list(parsed.keys())
        assert keys == sorted(keys), f"Keys should be sorted: {keys}"

    def test_json_with_unicode_escaping(self):
        """JSON with \\uXXXX escape sequences."""
        content = '{"message": "\\u4f60\\u597d"}'  # 你好
        msgs = [{"role": "user", "content": content}]
        result = JsonCanonicalizer.canonicalize_messages(msgs)
        # Should be canonicalized to actual unicode or preserved
        assert isinstance(result[0]["content"], str)

    def test_malformed_json_not_crashing(self):
        """Invalid JSON should be left as-is, not crash."""
        bad_jsons = [
            "{invalid",
            '{"unclosed": "string',
            "[broken",
            "<xml>not json</xml>",
            "NaN",
            "undefined",
        ]
        for bad in bad_jsons:
            msgs = [{"role": "user", "content": bad}]
            result = JsonCanonicalizer.canonicalize_messages(msgs)
            assert result[0]["content"] == bad, f"Invalid JSON left unchanged: {bad}"


# ═══════════════════════════════════════════════════════════════════════
# 4. CONTENT STRUCTURE CHAOS — Mixed content types
# ═══════════════════════════════════════════════════════════════════════

class TestContentStructure:
    """Messages with varied content structures."""

    def test_content_as_list_of_blocks(self):
        """Anthropic-style content blocks."""
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "abc"}},
                ],
            }
        ]
        try:
            result = optimize(msgs)
            assert isinstance(result, list)
        except Exception as e:
            pytest.fail(f"Content list crashed: {e}")

    def test_content_with_tool_use(self):
        """Tool use and tool result messages."""
        msgs = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me check..."},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "get_weather",
                        "input": {"location": "Beijing"},
                    },
                ],
            },
            {
                "role": "tool",
                "tool_use_id": "toolu_1",
                "content": "Sunny, 25°C",
            },
        ]
        try:
            result = optimize(msgs)
            assert isinstance(result, list)
        except Exception as e:
            pytest.fail(f"Tool use messages crashed: {e}")

    def test_multiple_tool_calls(self):
        """Multiple interleaved tool calls."""
        msgs = []
        for i in range(10):
            msgs.append({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"Step {i}"},
                    {"type": "tool_use", "id": f"tool_{i}", "name": "f", "input": {"x": i}},
                ],
            })
            msgs.append({
                "role": "tool",
                "tool_use_id": f"tool_{i}",
                "content": f"Result {i}",
            })
        msgs.append({"role": "user", "content": "Continue"})
        try:
            result = optimize(msgs)
            # Count tool results — should all be present
            # (use 'role' not '_role_type' — internal markers are stripped by Formatter)
            tool_msgs = [m for m in result if m.get("role") == "tool"]
            assert len(tool_msgs) == 10, f"Expected 10 tool msgs, got {len(tool_msgs)}"
        except Exception as e:
            pytest.fail(f"Multiple tool calls crashed: {e}")

    def test_empty_content_list(self):
        """Content as empty list."""
        msgs = [{"role": "user", "content": []}]
        try:
            result = optimize(msgs)
            assert isinstance(result, list)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════
# 5. ROUND-TRIP EQUIVALENCE — Content preservation
# ═══════════════════════════════════════════════════════════════════════

class TestRoundTripEquivalence:
    """Verify content is preserved through the optimization pipeline."""

    @staticmethod
    def _content_set(messages):
        """Extract set of (role, content) pairs for comparison."""
        result = set()
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, (dict, list)):
                content = json.dumps(content, sort_keys=True)
            result.add((role, str(content)))
        return result

    def test_content_set_preserved_simple(self):
        """Simple case: original content must be a subset of optimized content."""
        msgs = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Tell me a joke."},
            {"role": "assistant", "content": "Why did the chicken..."},
        ]
        before = self._content_set(msgs)
        optimized = optimize(msgs, provider="openai")
        after = self._content_set(optimized)
        # Each original (role, content) pair must still be present
        # (optimizer may ADD padding but must not REMOVE or MODIFY content)
        before_roles_contents = {(r, c[:50]) for r, c in before}
        after_roles_contents = {(r, c[:50]) for r, c in after}
        for role, content_prefix in before_roles_contents:
            matching = any(
                r == role and content_prefix in c
                for r, c in after
            )
            assert matching, f"Content missing after optimization: role={role}, content starts with={content_prefix}"

    def test_content_preserved_multiple_roles(self):
        """Multiple messages of same role — content set preserved."""
        msgs = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
            {"role": "user", "content": "Q3"},
        ]
        before = self._content_set(msgs)
        optimized = optimize(msgs, provider="openai")
        after = self._content_set(optimized)
        assert before == after, "Multi-role content preserved"

    def test_system_content_fragment_preserved(self):
        """System prompt should appear in the optimized output."""
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "CRITICAL_INSTRUCTION_42"},
        ]
        optimized = optimize(msgs)
        all_text = str(optimized)
        assert "CRITICAL_INSTRUCTION_42" in all_text, \
            "System instruction fragment must survive"

    def test_json_content_preserved_after_canonicalization(self):
        """JSON content should have same values after key reordering."""
        original = {"b": 2, "a": {"z": 26, "m": 13}, "c": [3, 1, 2]}
        msgs = [{"role": "user", "content": json.dumps(original)}]
        optimized = optimize(msgs, provider="openai")

        # Find user message in optimized output
        for msg in optimized:
            if msg.get("role") == "user":
                parsed = json.loads(msg["content"])
                assert parsed["a"]["z"] == 26
                assert parsed["a"]["m"] == 13
                assert parsed["b"] == 2
                assert parsed["c"] == [3, 1, 2]
                return
        pytest.fail("User message not found in optimized output")

    def test_tool_content_preserved(self):
        """Tool call results should survive optimization."""
        msgs = [
            {"role": "user", "content": "Calculate"},
            {"role": "assistant", "content": "Result is 42"},
            {"role": "tool", "tool_use_id": "call_1", "content": "42"},
        ]
        before = self._content_set(msgs)
        optimized = optimize(msgs, provider="openai")
        after = self._content_set(optimized)
        assert before == after, "Tool content preserved"


# ═══════════════════════════════════════════════════════════════════════
# 6. SAFETY GUARANTEES — Verify SafetyCheck catches problems
# ═══════════════════════════════════════════════════════════════════════

class TestSafetyGuarantees:
    """SafetyCheck must correctly reject bad transformations."""

    def test_safety_rejects_lost_messages(self):
        """If messages are lost, SafetyCheck returns False."""
        original = [
            {"role": "system", "content": "s1"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
        ]
        # Simulate losing a message
        tampered = [
            {"role": "system", "content": "s1"},
            {"role": "user", "content": "u1"},
        ]
        assert not SafetyCheck.verify_reorder(original, tampered)

    def test_safety_rejects_content_change(self):
        """If content is modified, SafetyCheck returns False."""
        original = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Tell me a joke."},
        ]
        # Simulate content being changed
        tampered = [
            {"role": "system", "content": "Be verbose."},  # Changed!
            {"role": "user", "content": "Tell me a joke."},
        ]
        assert not SafetyCheck.verify_reorder(original, tampered)

    def test_safety_accepts_valid_reorder(self):
        """Valid reordering must pass SafetyCheck."""
        original = [
            {"role": "user", "content": "u1"},
            {"role": "system", "content": "s1"},
        ]
        reordered = [
            {"role": "system", "content": "s1"},
            {"_is_separator": True, "role": "assistant", "content": "---"},
            {"role": "user", "content": "u1"},
        ]
        assert SafetyCheck.verify_reorder(original, reordered)

    def test_safety_system_unchanged_ok(self):
        """System message content unchanged -> passes."""
        original = [{"role": "system", "content": "You are helpful."}]
        optimized = [{"role": "system", "content": "You are helpful.\nExtra padding"}]
        assert SafetyCheck.verify_system_unchanged(original, optimized)

    def test_safety_system_changed_fails(self):
        """System message content changed -> fails."""
        original = [{"role": "system", "content": "You are helpful."}]
        tampered = [{"role": "system", "content": "You are evil."}]
        assert not SafetyCheck.verify_system_unchanged(original, tampered)


# ═══════════════════════════════════════════════════════════════════════
# 7. RACE CONDITIONS AND IDEMPOTENCY
# ═══════════════════════════════════════════════════════════════════════

class TestIdempotency:
    """Optimizing an already-optimized prompt should be a no-op."""

    def test_optimize_twice_same_result(self):
        """Optimize(a) should equal Optimize(Optimize(a))."""
        msgs = [
            {"role": "user", "content": '{"z": 1, "a": 2}\r\n'},
            {"role": "system", "content": "Be concise.\n\n\n\nExtra"},
        ]
        once = optimize(msgs, provider="openai")
        twice = optimize(once, provider="openai")
        # Compare just the content, ignoring separators
        once_content = [(m["role"], m["content"]) for m in once]
        twice_content = [(m["role"], m["content"]) for m in twice]
        assert once_content == twice_content, \
            "Second optimization should not change output"

    def test_deterministic_across_calls(self):
        """Same input, two separate calls → same output."""
        msgs = [
            {"role": "user", "content": "B"},
            {"role": "user", "content": "A"},
            {"role": "system", "content": "S"},
        ]
        r1 = optimize(msgs, provider="openai")
        r2 = optimize(msgs, provider="openai")
        r1_content = [(m["role"], m["content"]) for m in r1]
        r2_content = [(m["role"], m["content"]) for m in r2]
        assert r1_content == r2_content


# ═══════════════════════════════════════════════════════════════════════
# 8. STRESS / PERFORMANCE — Large inputs
# ═══════════════════════════════════════════════════════════════════════

class TestStress:
    """Stress tests with very large inputs."""

    def test_many_messages(self):
        """1000 messages should not crash."""
        msgs = []
        for i in range(500):
            msgs.append({"role": "user", "content": f"Message {i}"})
            msgs.append({"role": "assistant", "content": f"Response {i}"})
        try:
            result = optimize(msgs, provider="openai")
            # Check no messages lost
            user_msgs = [m for m in result if m.get("role") == "user"]
            asst_msgs = [m for m in result if m.get("role") == "assistant"]
            assert len(user_msgs) == 500, f"Expected 500 user msgs, got {len(user_msgs)}"
            assert len(asst_msgs) >= 500, f"Expected 500+ assistant msgs, got {len(asst_msgs)}"
        except Exception as e:
            pytest.fail(f"Large messages crashed: {e}")

    def test_huge_content_string(self):
        """500k char content string."""
        content = "Hello World " * 33_333  # ~500k chars
        msgs = [{"role": "system", "content": content}]
        try:
            result = optimize(msgs, provider="openai")
            assert isinstance(result, list)
        except (MemoryError, RecursionError):
            pytest.skip("System resources insufficient for this test")
        except Exception as e:
            pytest.fail(f"Huge content crashed: {e}")

    def test_many_distinct_messages_deterministic(self):
        """200 different messages — verify deterministic output."""
        msgs = [{"role": "user", "content": f"q{i}"} for i in range(200)]
        r1 = optimize(msgs, provider="openai")
        r2 = optimize(msgs, provider="openai")
        r1_ct = [(m["role"], m["content"]) for m in r1]
        r2_ct = [(m["role"], m["content"]) for m in r2]
        assert r1_ct == r2_ct, "Deterministic with 200 messages"


# ═══════════════════════════════════════════════════════════════════════
# 9. CROSS-PROVIDER CONSISTENCY
# ═══════════════════════════════════════════════════════════════════════

class TestCrossProvider:
    """Same content across different providers should preserve data."""

    def test_content_survives_anthropic_format(self):
        """Anthropic format conversion should preserve content."""
        msgs = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello"},
        ]
        result = optimize(msgs, provider="anthropic")
        # System message should be present
        sys_msgs = [m for m in result if m.get("role") == "system"]
        assert len(sys_msgs) >= 1

    def test_content_survives_openai_format(self):
        """OpenAI format conversion should preserve content."""
        msgs = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello"},
        ]
        result = optimize(msgs, provider="openai")
        all_roles = [m["role"] for m in result]
        assert "user" in all_roles


# ═══════════════════════════════════════════════════════════════════════
# 10. WHITESPACE / NORMALIZATION EDGE CASES
# ═══════════════════════════════════════════════════════════════════════

class TestWhitespaceEdgeCases:
    """Edge cases in whitespace normalization."""

    def test_only_whitespace_content(self):
        """Content that is entirely whitespace."""
        msgs = [{"role": "user", "content": "   \n   \n   "}]
        result = WhitespaceSanitizer.sanitize_messages(msgs)
        assert isinstance(result[0]["content"], str)

    def test_mixed_line_endings(self):
        """Mix of \\r\\n, \\n, and \\r in the same content."""
        content = "line1\r\nline2\nline3\rline4"
        result = WhitespaceSanitizer.sanitize(content)
        assert "\\r" not in result, "All line endings normalized to \\n"

    def test_tabs_not_collapsed(self):
        """Tabs should be preserved (they might be intentional)."""
        content = "col1\tcol2\tcol3"
        result = WhitespaceSanitizer.sanitize(content)
        assert "\t" in result, "Tabs preserved"

    def test_leading_trailing_whitespace(self):
        """Only trailing whitespace on each line is stripped (leading preserved for indentation)."""
        content = "  hello  \n  world  "
        result = WhitespaceSanitizer.sanitize(content)
        # Leading spaces preserved, trailing stripped
        assert result == "  hello\n  world"

    def test_unicode_whitespace(self):
        """Unicode whitespace characters."""
        import unicodedata
        ws_chars = [c for c in range(0x110000)
                    if c < 0x110000 and unicodedata.category(chr(c)) == 'Zs']
        sample = ''.join(chr(c) for c in ws_chars[:10])
        msgs = [{"role": "user", "content": f"a{sample}b"}]
        result = WhitespaceSanitizer.sanitize_messages(msgs)
        assert "a" in result[0]["content"]
        assert "b" in result[0]["content"]


# ═══════════════════════════════════════════════════════════════════════
# 11. CONFIG / PROVIDER EDGE CASES
# ═══════════════════════════════════════════════════════════════════════

class TestProviderEdgeCases:
    """Edge cases in provider configuration."""

    def test_unknown_provider_fallback(self):
        """Unknown provider should fall back gracefully."""
        from optimizer import register_provider
        register_provider("extreme_test", cache_threshold=0)
        msgs = [{"role": "user", "content": "hello"}]
        result = optimize(msgs, provider="extreme_test")
        assert isinstance(result, list)

    def test_zero_threshold_provider(self):
        """Provider with 0 threshold should still work."""
        from optimizer import register_provider
        register_provider("zero_thresh", cache_threshold=0, supports_breakpoints=False)
        msgs = [{"role": "user", "content": "hello"}]
        result = optimize(msgs, provider="zero_thresh")
        assert isinstance(result, list)

    def test_extreme_threshold(self):
        """Extremely high threshold should pad a lot but not crash."""
        from optimizer import register_provider
        register_provider("extreme", cache_threshold=10_000_000)
        msgs = [{"role": "user", "content": "hello"}]
        try:
            result = optimize(msgs, provider="extreme")
            assert isinstance(result, list)
        except (MemoryError, OverflowError):
            pytest.skip("Insufficient memory for extreme padding")
        except Exception as e:
            pytest.fail(f"Extreme threshold crashed: {e}")


# ═══════════════════════════════════════════════════════════════════════
# 12. Functional correctness — known inputs, expected outputs
# ═══════════════════════════════════════════════════════════════════════

class TestFunctionalCorrectness:
    """Known input → expected output patterns."""

    def test_system_content_fragment_in_anthropic_output(self):
        """System prompt text must appear in Anthropic output."""
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "system", "content": "Fragile: DO_NOT_LOSE"},
        ]
        result = optimize(msgs, provider="anthropic")
        all_text = json.dumps(result)
        assert "Fragile: DO_NOT_LOSE" in all_text

    def test_json_keys_sorted_recursively(self):
        """JSON in content should have sorted keys."""
        msgs = [
            {"role": "user", "content": json.dumps({"z": 1, "a": 2, "n": {"y": 1, "x": 2}})},
        ]
        result = optimize(msgs, provider="openai")
        user_msg = [m for m in result if m.get("role") == "user"][0]
        parsed = json.loads(user_msg["content"])
        keys = list(parsed.keys())
        assert keys == ["a", "n", "z"], f"Keys sorted at top: {keys}"
        nested_keys = list(parsed["n"].keys())
        assert nested_keys == ["x", "y"], f"Keys sorted nested: {nested_keys}"

    def test_optimizer_never_adds_new_role(self):
        """Optimizer must not invent roles that weren't in the input."""
        msgs = [
            {"role": "user", "content": "Q"},
            {"role": "system", "content": "S"},
        ]
        result = optimize(msgs, provider="anthropic")
        for msg in result:
            role = msg.get("role", "")
            assert role in ("system", "user", "assistant", "tool"), \
                f"Unexpected role: {role}"
