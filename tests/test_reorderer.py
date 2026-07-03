"""Tests for Reorderer module."""

import pytest
from optimizer.reorderer import Reorderer, _content_hash
from optimizer.config import ProviderConfig


def _msg(role, content, role_type=None):
    m = {"role": role, "content": content}
    if role_type:
        m["_role_type"] = role_type
    return m


class TestReorderer:
    def test_system_first(self):
        msgs = [
            _msg("user", "Hello", "user"),
            _msg("system", "Be concise.", "system"),
        ]
        result = Reorderer().run(msgs)
        assert result[0]["role"] == "system"
        # Find the user message (might be after separator)
        user_msgs = [m for m in result if m.get("_role_type") == "user"]
        assert len(user_msgs) == 1
        assert user_msgs[0]["content"] == "Hello"

    def test_deterministic_order(self):
        msgs = [
            _msg("user", "B", "user"),
            _msg("user", "A", "user"),
            _msg("system", "Be concise.", "system"),
        ]
        r1 = Reorderer().run(msgs)
        r2 = Reorderer().run(msgs)
        # Both runs should produce identical order
        assert [m["content"] for m in r1] == [m["content"] for m in r2]

    def test_same_content_same_order(self):
        msgs = [
            _msg("user", "same", "user"),
            _msg("user", "same", "user"),
            _msg("user", "same", "user"),
        ]
        r1 = Reorderer().run(msgs)
        # All "same" have same hash, order among them may vary
        # but non-separator count should be same
        non_sep = [m for m in r1 if not m.get("_is_separator")]
        assert len(non_sep) == 3

    def test_separator_insertion(self):
        msgs = [
            _msg("system", "Be concise.", "system"),
            _msg("user", "Hello", "user"),
            _msg("assistant", "Hi!", "assistant"),
        ]
        result = Reorderer().run(msgs)
        # Should have: system → separator → user → separator → assistant
        roles = [m.get("_role_type") for m in result]
        non_sep = [r for r in roles if r != "separator"]
        assert non_sep == ["system", "user", "assistant"]

    def test_empty_input(self):
        assert Reorderer().run([]) == []

    def test_single_message(self):
        msgs = [_msg("user", "hello", "user")]
        result = Reorderer().run(msgs)
        assert len(result) == 1

    def test_preserves_system_content(self):
        msgs = [
            _msg("system", "Important system prompt", "system"),
            _msg("user", "Hi", "user"),
        ]
        result = Reorderer().run(msgs)
        assert result[0]["content"] == "Important system prompt"

    def test_provider_separator(self):
        cfg = ProviderConfig(name="test", separator="===BREAK===")
        msgs = [
            _msg("system", "s", "system"),
            _msg("user", "u", "user"),
        ]
        result = Reorderer(cfg).run(msgs)
        # Find separator
        seps = [m for m in result if m.get("_is_separator")]
        assert len(seps) == 1
        assert seps[0]["content"] == "===BREAK==="


class TestContentHash:
    def test_identical_content(self):
        m1 = {"content": "hello"}
        m2 = {"content": "hello"}
        assert _content_hash(m1) == _content_hash(m2)

    def test_different_content(self):
        m1 = {"content": "hello"}
        m2 = {"content": "world"}
        assert _content_hash(m1) != _content_hash(m2)

    def test_list_content(self):
        m1 = {"content": [{"type": "text", "text": "hello"}]}
        m2 = {"content": [{"type": "text", "text": "hello"}]}
        assert _content_hash(m1) == _content_hash(m2)
