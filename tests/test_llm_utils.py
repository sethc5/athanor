"""Tests for strip_json_fences() in llm_utils."""
from __future__ import annotations

from athanor.llm_utils import strip_json_fences


class TestStripJsonFences:
    def test_plain_json(self):
        assert strip_json_fences('{"a":1}') == '{"a":1}'

    def test_json_code_fence(self):
        raw = '```json\n{"a":1}\n```'
        assert strip_json_fences(raw) == '{"a":1}'

    def test_bare_code_fence(self):
        raw = '```\n{"a":1}\n```'
        assert strip_json_fences(raw) == '{"a":1}'

    def test_leading_trailing_whitespace(self):
        raw = '  \n```json\n{"a":1}\n```\n  '
        assert strip_json_fences(raw) == '{"a":1}'

    def test_no_fences_with_whitespace(self):
        assert strip_json_fences('  {"a":1}  ') == '{"a":1}'

    def test_nested_backticks_in_value(self):
        raw = '```json\n{"code":"use `x`"}\n```'
        result = strip_json_fences(raw)
        assert '"code"' in result

    def test_multiline_json(self):
        raw = '```json\n{\n  "a": 1,\n  "b": 2\n}\n```'
        result = strip_json_fences(raw)
        import json
        assert json.loads(result) == {"a": 1, "b": 2}

    def test_empty_string(self):
        assert strip_json_fences("") == ""

    def test_only_fences(self):
        assert strip_json_fences("``````") == ""

    def test_triple_backtick_json_label_no_newline(self):
        raw = '```json{"a":1}```'
        result = strip_json_fences(raw)
        assert '{"a":1}' in result

    def test_no_closing_fence(self):
        """If there's an opening fence but no closing, return inner content."""
        raw = '```json\n{"a":1}'
        result = strip_json_fences(raw)
        assert '{"a":1}' in result
