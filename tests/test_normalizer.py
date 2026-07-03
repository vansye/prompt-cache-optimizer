"""Tests for Normalizer module."""

import json
import pytest
from optimizer.normalizer import (
    WhitespaceSanitizer,
    JsonCanonicalizer,
    TypeMarker,
    Normalizer,
)


class TestWhitespaceSanitizer:
    def test_normalize_line_endings(self):
        result = WhitespaceSanitizer.sanitize("hello\r\nworld\r\n")
        assert result == "hello\nworld"

    def test_strip_bom(self):
        result = WhitespaceSanitizer.sanitize("﻿hello")
        assert result == "hello"

    def test_trailing_whitespace_per_line(self):
        result = WhitespaceSanitizer.sanitize("hello   \nworld  \n")
        assert result == "hello\nworld"

    def test_collapse_blank_lines(self):
        result = WhitespaceSanitizer.sanitize("a\n\n\n\n\nb")
        assert result == "a\n\nb"

    def test_non_string_passthrough(self):
        assert WhitespaceSanitizer.sanitize(42) == 42
        assert WhitespaceSanitizer.sanitize(None) is None

    def test_sanitize_messages(self):
        msgs = [
            {"role": "user", "content": "hello\r\nworld"},
            {"role": "assistant", "content": "ok\r\nbye"},
        ]
        result = WhitespaceSanitizer.sanitize_messages(msgs)
        assert result[0]["content"] == "hello\nworld"
        assert result[1]["content"] == "ok\nbye"


class TestJsonCanonicalizer:
    def test_sort_object_keys(self):
        data = {"z": 1, "a": 2, "m": 3}
        result = JsonCanonicalizer.canonicalize(data)
        assert result == '{"a":2,"m":3,"z":1}'

    def test_nested_sort(self):
        data = {"b": {"y": 1, "x": 2}, "a": 3}
        result = JsonCanonicalizer.canonicalize(data)
        assert result == '{"a":3,"b":{"x":2,"y":1}}'

    def test_list_preserved(self):
        data = {"z": [3, 1, 2], "a": "hello"}
        result = JsonCanonicalizer.canonicalize(data)
        assert json.loads(result)["z"] == [3, 1, 2]

    def test_try_canonicalize_valid_json(self):
        result = JsonCanonicalizer.try_canonicalize_content('{"b":1,"a":2}')
        assert result == '{"a":2,"b":1}'

    def test_try_canonicalize_invalid_json(self):
        result = JsonCanonicalizer.try_canonicalize_content("not json")
        assert result == "not json"

    def test_try_canonicalize_non_string(self):
        assert JsonCanonicalizer.try_canonicalize_content(42) == 42

    def test_canonicalize_messages(self):
        msgs = [
            {"role": "user", "content": '{"z": 1, "a": 2}'},
            {"role": "tool", "content": {"b": 3, "a": 4}},
        ]
        result = JsonCanonicalizer.canonicalize_messages(msgs)
        assert result[0]["content"] == '{"a":2,"z":1}'
        assert result[1]["content"] == {"a": 4, "b": 3}


class TestTypeMarker:
    def test_mark_valid_roles(self):
        msgs = [
            {"role": "system"},
            {"role": "user"},
            {"role": "assistant"},
            {"role": "tool"},
        ]
        result = TypeMarker.mark(msgs)
        assert result[0]["_role_type"] == "system"
        assert result[1]["_role_type"] == "user"
        assert result[2]["_role_type"] == "assistant"
        assert result[3]["_role_type"] == "tool"

    def test_mark_unknown_role(self):
        msgs = [{"role": "custom_role"}]
        result = TypeMarker.mark(msgs)
        assert result[0]["_role_type"] == "other"

    def test_preserves_original(self):
        msgs = [{"role": "user", "content": "hello", "extra": True}]
        result = TypeMarker.mark(msgs)
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "hello"
        assert result[0]["extra"] is True
        assert result[0]["_role_type"] == "user"


class TestNormalizer:
    def test_full_pipeline(self):
        msgs = [
            {"role": "user", "content": '{"z": 1, "a": 2}\r\n'},
            {"role": "system", "content": "be concise"},
        ]
        result = Normalizer.run(msgs)
        assert len(result) == 2
        # First message: JSON should be canonicalized, whitespace cleaned
        assert result[0]["_role_type"] == "user"
        # Second: system
        assert result[1]["_role_type"] == "system"

    def test_empty_input(self):
        assert Normalizer.run([]) == []
