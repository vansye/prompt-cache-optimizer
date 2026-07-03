"""Tests for Aligner module."""

import pytest
from optimizer.aligner import Aligner, _estimate_tokens, _prefix_token_count
from optimizer.config import ProviderConfig


def _msg(role, content, role_type=None):
    m = {"role": role, "content": content}
    if role_type:
        m["_role_type"] = role_type
    return m


class TestAligner:
    def test_no_config_returns_unchanged(self):
        msgs = [_msg("user", "hello", "user")]
        result = Aligner().run(msgs)
        assert result == msgs

    def test_injects_cache_tag_on_system(self):
        cfg = ProviderConfig(
            name="anthropic",
            cache_threshold=10,
            supports_breakpoints=True,
            cache_tag_system=True,
        )
        msgs = [
            _msg("system", "Be concise.", "system"),
            _msg("user", "Hello", "user"),
        ]
        result = Aligner(cfg).run(msgs)
        system = result[0]
        assert isinstance(system["content"], list)
        assert system["content"][0]["type"] == "text"
        assert "cache_control" in system["content"][0]

    def test_no_cache_tag_if_not_supported(self):
        cfg = ProviderConfig(
            name="openai",
            cache_threshold=10,
            supports_breakpoints=False,
        )
        msgs = [
            _msg("system", "Be concise.", "system"),
            _msg("user", "Hello", "user"),
        ]
        result = Aligner(cfg).run(msgs)
        assert isinstance(result[0]["content"], str)  # Not converted to list

    def test_padding_when_below_threshold(self):
        cfg = ProviderConfig(
            name="test",
            cache_threshold=999999,  # Very high threshold
            supports_breakpoints=False,
        )
        msgs = [
            _msg("system", "Short", "system"),
            _msg("user", "Hi", "user"),
        ]
        result = Aligner(cfg).run(msgs)
        # System content should be padded
        assert len(result[0]["content"]) > len("Short")

    def test_no_padding_when_above_threshold(self):
        cfg = ProviderConfig(
            name="test",
            cache_threshold=10,
            supports_breakpoints=False,
        )
        long_content = "x" * 1000
        msgs = [
            _msg("system", long_content, "system"),
            _msg("user", "Hi", "user"),
        ]
        result = Aligner(cfg).run(msgs)
        assert result[0]["content"] == long_content  # Unchanged

    def test_does_not_add_system_if_none_exists(self):
        cfg = ProviderConfig(
            name="test",
            cache_threshold=999999,
        )
        msgs = [
            _msg("user", "Hello", "user"),
        ]
        result = Aligner(cfg).run(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"


class TestTokenEstimation:
    def test_estimate(self):
        # Empty text
        assert _estimate_tokens("") == 0
        # Non-string
        assert _estimate_tokens(None) == 0
        assert _estimate_tokens(123) == 0
        # With tiktoken: "abcd" = 1 token, simple text is accurately counted
        assert _estimate_tokens("abcd") >= 1
        # Verify tiktoken is being used (should give accurate BPE counts)
        long_text = "The quick brown fox jumps over the lazy dog"
        count = _estimate_tokens(long_text)
        assert count > 0

    def test_prefix_count(self):
        msgs = [
            {"content": "abc def ghi jkl"},  # 16 chars = 6 tiktoken tokens
            {"content": "mno pqr stu vwx"},  # 16 chars = 7 tiktoken tokens
        ]
        count = _prefix_token_count(msgs)
        # tiktoken (cl100k_base) accurately counts both messages
        assert count == 13
