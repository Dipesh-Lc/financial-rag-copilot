"""
test_utils.py
Tests for utility modules: json_utils, text_utils, date_utils, file_io.
"""

import json
import pytest
from pathlib import Path


# ── json_utils ────────────────────────────────────────────────────────────────

class TestJsonUtils:
    def setup_method(self):
        from app.utils.json_utils import safe_parse
        self.parse = safe_parse

    def test_valid_json_dict(self):
        result = self.parse('{"key": "value"}')
        assert result == {"key": "value"}

    def test_valid_json_list(self):
        result = self.parse('[1, 2, 3]')
        assert result == [1, 2, 3]

    def test_strips_json_fence(self):
        result = self.parse('```json\n{"a": 1}\n```')
        assert result == {"a": 1}

    def test_strips_plain_fence(self):
        result = self.parse('```\n{"a": 1}\n```')
        assert result == {"a": 1}

    def test_returns_none_on_invalid(self):
        assert self.parse("not json") is None
        assert self.parse("{bad json}") is None
        assert self.parse("") is None


# ── text_utils ────────────────────────────────────────────────────────────────

class TestTextUtils:
    def test_truncate_long(self):
        from app.utils.text_utils import truncate
        text = "A" * 500
        result = truncate(text, max_chars=100)
        assert result.endswith("...")
        assert len(result) == 103

    def test_truncate_short(self):
        from app.utils.text_utils import truncate
        text = "short"
        assert truncate(text, max_chars=100) == "short"

    def test_count_tokens_approx(self):
        from app.utils.text_utils import count_tokens_approx
        assert count_tokens_approx("") == 0
        assert count_tokens_approx("A" * 400) == 100

    def test_normalize_whitespace(self):
        from app.utils.text_utils import normalize_whitespace
        result = normalize_whitespace("  hello   world  \n  test  ")
        assert result == "hello world test"


# ── date_utils ────────────────────────────────────────────────────────────────

class TestDateUtils:
    def test_iso_format(self):
        from app.utils.date_utils import parse_filing_date
        dt = parse_filing_date("2024-09-28")
        assert dt is not None
        assert dt.year == 2024
        assert dt.month == 9

    def test_slash_format(self):
        from app.utils.date_utils import parse_filing_date
        dt = parse_filing_date("09/28/2024")
        assert dt is not None
        assert dt.year == 2024

    def test_compact_format(self):
        from app.utils.date_utils import parse_filing_date
        dt = parse_filing_date("20240928")
        assert dt is not None
        assert dt.day == 28

    def test_invalid_returns_none(self):
        from app.utils.date_utils import parse_filing_date
        assert parse_filing_date("not a date") is None
        assert parse_filing_date("") is None


# ── file_io ───────────────────────────────────────────────────────────────────

class TestFileIO:
    def test_write_and_read_dict(self, tmp_path):
        from app.utils.file_io import write_json, read_json
        path = tmp_path / "test.json"
        data = {"key": "value", "num": 42}
        write_json(data, path)
        result = read_json(path)
        assert result == data

    def test_write_and_read_list(self, tmp_path):
        from app.utils.file_io import write_json, read_json
        path = tmp_path / "list.json"
        data = [1, 2, 3]
        write_json(data, path)
        assert read_json(path) == [1, 2, 3]

    def test_creates_parent_dirs(self, tmp_path):
        from app.utils.file_io import write_json
        path = tmp_path / "deep" / "nested" / "file.json"
        write_json({"a": 1}, path)
        assert path.exists()
