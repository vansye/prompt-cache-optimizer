"""Tests for Diagnoser module."""

import pytest
from optimizer.diagnoser import (
    parse_usage,
    MissAnalyzer,
    HitRatioCalculator,
    CacheMetrics,
    CacheEntry,
    CacheReport,
    PrefixShape,
    PrefixDiffer,
    PrefixDiffReport,
    capture_shape,
    compare_shapes,
    ShapeDiff,
)


# ── Prefix Shape ─────────────────────────────────────────────────────────────


class TestPrefixShape:
    def test_capture_shape_basic(self):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello."},
        ]
        shape = capture_shape(msgs)
        assert isinstance(shape, PrefixShape)
        assert len(shape.system_hash) == 8
        assert len(shape.role_sequence_hash) == 8
        assert len(shape.content_prefix_hash) == 8
        assert len(shape.full_prefix_hash) == 8

    def test_capture_shape_empty(self):
        shape = capture_shape([])
        assert shape.system_hash == ""
        assert shape.role_sequence_hash != ""
        assert shape.content_prefix_hash != ""

    def test_capture_shape_deterministic(self):
        msgs = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Tell a joke."},
        ]
        a = capture_shape(msgs)
        b = capture_shape(msgs)
        assert a.system_hash == b.system_hash
        assert a.full_prefix_hash == b.full_prefix_hash

    def test_shape_differs_on_content_change(self):
        base = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello."},
        ]
        changed = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi there!"},
        ]
        s1 = capture_shape(base)
        s2 = capture_shape(changed)
        assert s1.system_hash == s2.system_hash  # system unchanged
        assert s1.content_prefix_hash != s2.content_prefix_hash  # user content changed

    def test_shape_differs_on_role_change(self):
        base = [
            {"role": "system", "content": "Help."},
            {"role": "user", "content": "Hi."},
        ]
        changed = [
            {"role": "system", "content": "Help."},
            {"role": "assistant", "content": "Hi."},
        ]
        s1 = capture_shape(base)
        s2 = capture_shape(changed)
        assert s1.role_sequence_hash != s2.role_sequence_hash


class TestCompareShapes:
    def test_identical_shapes(self):
        s = PrefixShape(system_hash="abc", role_sequence_hash="def")
        result = compare_shapes(s, s)
        assert not result.changed
        assert result.reasons == []

    def test_system_changed(self):
        a = PrefixShape(system_hash="abc", content_prefix_hash="xyz")
        b = PrefixShape(system_hash="def", content_prefix_hash="xyz")
        result = compare_shapes(a, b)
        assert result.changed
        assert "system" in result.reasons

    def test_empty_previous_doesnt_report(self):
        a = PrefixShape()  # empty — no cache established yet
        b = PrefixShape(system_hash="abc")
        result = compare_shapes(a, b)
        assert not result.changed


# ── Prefix Differ ────────────────────────────────────────────────────────────


class TestPrefixDiffer:
    def test_identical_messages(self):
        a = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hello."},
        ]
        b = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hello."},
        ]
        result = PrefixDiffer.diff(a, b)
        assert result is None  # No diff

    def test_content_changed(self):
        a = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hello."},
        ]
        b = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hi there!"},
        ]
        result = PrefixDiffer.diff(a, b)
        assert result is not None
        assert result.msg_index == 1  # user message
        assert result.msg_role == "user"
        assert result.field == "content"

    def test_role_changed(self):
        a = [
            {"role": "system", "content": "Help."},
            {"role": "user", "content": "Hi."},
        ]
        b = [
            {"role": "system", "content": "Help."},
            {"role": "assistant", "content": "Hi."},
        ]
        result = PrefixDiffer.diff(a, b)
        assert result is not None
        assert result.field == "role"

    def test_message_count_differs(self):
        a = [
            {"role": "system", "content": "Help."},
            {"role": "user", "content": "Hi."},
        ]
        b = [
            {"role": "system", "content": "Help."},
            {"role": "user", "content": "Hi."},
            {"role": "user", "content": "Another."},
        ]
        result = PrefixDiffer.diff(a, b)
        assert result is not None
        assert result.field == "message_count"

    def test_empty_sequences(self):
        result = PrefixDiffer.diff([], [])
        assert result is None

    def test_one_empty_one_not(self):
        result = PrefixDiffer.diff([], [{"role": "user", "content": "Hi."}])
        assert result is not None
        assert result.field == "message_count"

    def test_system_change_is_critical(self):
        a = [
            {"role": "system", "content": "Old system prompt here."},
            {"role": "user", "content": "Hello."},
        ]
        b = [
            {"role": "system", "content": "New system prompt here!"},
            {"role": "user", "content": "Hello."},
        ]
        result = PrefixDiffer.diff(a, b)
        assert result is not None
        assert result.severity == "CRITICAL"
        assert result.msg_index == 0

    def test_tool_content_blocks(self):
        a = [
            {"role": "assistant", "content": [
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "name": "get_weather", "input": {"city": "NYC"}},
            ]},
        ]
        b = [
            {"role": "assistant", "content": [
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "name": "get_temperature", "input": {"city": "NYC"}},
            ]},
        ]
        result = PrefixDiffer.diff(a, b)
        assert result is not None
        assert result.field == "content"

    def test_miss_analyzer_with_diff(self):
        """MissAnalyzer should include diff info when provided."""
        a_msgs = [{"role": "system", "content": "Be helpful."}]
        b_msgs = [{"role": "system", "content": "Be concise."}]
        pdiff = PrefixDiffer.diff(a_msgs, b_msgs)
        shape = compare_shapes(capture_shape(a_msgs), capture_shape(b_msgs))

        metrics = CacheMetrics(input_tokens=100)
        result = MissAnalyzer.analyze(
            metrics,
            previous_metrics=CacheMetrics(input_tokens=200),
            prefix_diff=pdiff,
            shape_diff=shape,
        )
        assert "MISS" in result
        assert "0" in str(pdiff.msg_index) if pdiff else True
        assert "system" in result


class TestParseUsage:
    def test_anthropic_format(self):
        response = {
            "usage": {
                "input_tokens": 100,
                "cache_creation_input_tokens": 80,
                "cache_read_input_tokens": 20,
                "output_tokens": 50,
            }
        }
        metrics = parse_usage(response)
        assert metrics.input_tokens == 100
        assert metrics.cache_creation_input_tokens == 80
        assert metrics.cache_read_input_tokens == 20
        assert metrics.output_tokens == 50

    def test_openai_format(self):
        response = {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "prompt_tokens_details": {"cached_tokens": 30},
            }
        }
        metrics = parse_usage(response)
        assert metrics.input_tokens == 100
        assert metrics.cache_read_input_tokens == 30
        assert metrics.output_tokens == 50

    def test_empty_response(self):
        metrics = parse_usage({})
        assert metrics.input_tokens == 0
        assert metrics.output_tokens == 0

    def test_missing_cache_fields(self):
        response = {"usage": {"input_tokens": 50}}
        metrics = parse_usage(response)
        assert metrics.cache_read_input_tokens == 0
        assert metrics.cache_creation_input_tokens == 0


class TestMissAnalyzer:
    def test_hit_detection(self):
        metrics = CacheMetrics(
            input_tokens=100,
            cache_read_input_tokens=30,
        )
        result = MissAnalyzer.analyze(metrics)
        assert "HIT" in result
        assert "30" in result

    def test_miss_first_request(self):
        metrics = CacheMetrics(input_tokens=100)
        result = MissAnalyzer.analyze(metrics, previous_metrics=None)
        assert "MISS" in result
        assert "First request" in result

    def test_miss_short_prefix(self):
        metrics = CacheMetrics(input_tokens=100)
        result = MissAnalyzer.analyze(
            metrics, prefix_length=50, threshold=1024
        )
        assert "MISS" in result
        assert "short" in result.lower()

    def test_miss_cache_created(self):
        metrics = CacheMetrics(
            input_tokens=100,
            cache_creation_input_tokens=80,
        )
        result = MissAnalyzer.analyze(metrics, prefix_length=1000, threshold=500)
        assert "MISS" in result
        assert "Cache created" in result


class TestHitRatioCalculator:
    def test_empty_entries(self):
        report = HitRatioCalculator.calculate([])
        assert report.total_requests == 0

    def test_all_misses(self):
        entries = [
            CacheEntry(0, CacheMetrics(input_tokens=100), is_hit=False),
            CacheEntry(1, CacheMetrics(input_tokens=100), is_hit=False),
        ]
        report = HitRatioCalculator.calculate(entries)
        assert report.hit_ratio == 0.0

    def test_mixed_hits(self):
        entries = [
            CacheEntry(0, CacheMetrics(input_tokens=100), is_hit=False),
            CacheEntry(1, CacheMetrics(
                input_tokens=100,
                cache_read_input_tokens=80,
            ), is_hit=True),
        ]
        report = HitRatioCalculator.calculate(entries)
        assert report.hit_ratio > 0
        assert report.savings_ratio > 0

    def test_cost_estimate_positive(self):
        entries = [
            CacheEntry(0, CacheMetrics(
                input_tokens=1000,
                cache_creation_input_tokens=800,
                cache_read_input_tokens=200,
            ), is_hit=True),
            CacheEntry(1, CacheMetrics(
                input_tokens=200,
                cache_read_input_tokens=200,
            ), is_hit=True),
        ]
        report = HitRatioCalculator.calculate(entries)
        assert report.estimated_cost_saved > 0
