"""Tests for Formatter module."""

import pytest
from optimizer.formatter import Formatter
from optimizer.config import ProviderConfig


def _msg(role, content, **kwargs):
    m = {"role": role, "content": content}
    m.update(kwargs)
    return m


class TestFormatter:
    def test_strip_internal_markers(self):
        msgs = [
            _msg("system", "s", _role_type="system", _is_separator=False),
            _msg("user", "u", _role_type="user"),
        ]
        result = Formatter._strip_internal(msgs)
        for msg in result:
            assert "_role_type" not in msg
            assert not any(k.startswith("_") for k in msg)

    def test_remove_separator_messages(self):
        msgs = [
            _msg("system", "s"),
            _msg("assistant", "---", _is_separator=True),
            _msg("user", "u"),
        ]
        result = Formatter._strip_internal(msgs)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"

    def test_anthropic_format(self):
        msgs = [_msg("user", "hello")]
        result = Formatter._format_anthropic(msgs)
        assert result == msgs  # Anthropic passes through unchanged

    def test_openai_format_flattens_content_blocks(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "text", "text": "world"},
                ],
            }
        ]
        result = Formatter._format_openai(msgs)
        assert result[0]["content"] == "hello\nworld"

    def test_openai_format_string_content_unchanged(self):
        msgs = [_msg("user", "hello")]
        result = Formatter._format_openai(msgs)
        assert result[0]["content"] == "hello"

    def test_format_with_provider(self):
        msgs = [_msg("user", "hi", _role_type="user")]
        result = Formatter().format(msgs, provider="openai")
        assert "_role_type" not in result[0]

    def test_format_unknown_provider(self):
        msgs = [_msg("user", "hi", _role_type="user")]
        result = Formatter().format(msgs, provider="unknown")
        assert "_role_type" not in result[0]
