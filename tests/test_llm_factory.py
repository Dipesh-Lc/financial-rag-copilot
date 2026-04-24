"""
test_llm_factory.py
Tests for the multi-provider LLM factory.
AppConfig is a frozen dataclass, so tests patch module-level constants
in app.llm.factory directly rather than mutating CONFIG.
"""

from __future__ import annotations

import pytest

from app.llm.factory import (
    LLMProvider,
    LLMConfigError,
    build_llm,
    build_llm_with_fallback,
    detect_available_providers,
)

import app.llm.factory as factory_module


class TestLLMProvider:
    def test_provider_enum_values(self):
        assert LLMProvider.ANTHROPIC == "anthropic"
        assert LLMProvider.OPENAI == "openai"
        assert LLMProvider.GEMINI == "gemini"

    def test_provider_str_comparison(self):
        assert LLMProvider.ANTHROPIC.value == "anthropic"


class TestBuildLLM:
    def test_unknown_provider_raises(self):
        with pytest.raises(LLMConfigError, match="Unknown LLM provider"):
            build_llm(provider="unknown_provider")

    def test_anthropic_missing_key_raises(self, monkeypatch):
        # Patch the module-level CONFIG reference inside the factory
        import app.llm.factory as fm
        from unittest.mock import MagicMock
        fake_config = MagicMock()
        fake_config.anthropic_api_key = ""
        fake_config.openai_api_key = "sk-test"
        fake_config.google_api_key = "gkey"
        fake_config.llm_provider = "anthropic"
        fake_config.llm_model_name = ""
        fake_config.llm_temperature = 0.0
        fake_config.llm_max_tokens = 1000
        monkeypatch.setattr(fm, "CONFIG", fake_config)
        with pytest.raises(LLMConfigError, match="ANTHROPIC_API_KEY"):
            build_llm(provider="anthropic")

    def test_openai_missing_key_raises(self, monkeypatch):
        import app.llm.factory as fm
        from unittest.mock import MagicMock
        fake_config = MagicMock()
        fake_config.openai_api_key = ""
        fake_config.anthropic_api_key = "sk-ant"
        fake_config.google_api_key = ""
        fake_config.llm_provider = "openai"
        fake_config.llm_model_name = ""
        fake_config.llm_temperature = 0.0
        fake_config.llm_max_tokens = 1000
        monkeypatch.setattr(fm, "CONFIG", fake_config)
        with pytest.raises(LLMConfigError, match="OPENAI_API_KEY"):
            build_llm(provider="openai")

    def test_gemini_missing_key_raises(self, monkeypatch):
        import app.llm.factory as fm
        from unittest.mock import MagicMock
        fake_config = MagicMock()
        fake_config.google_api_key = ""
        fake_config.anthropic_api_key = ""
        fake_config.openai_api_key = ""
        fake_config.llm_provider = "gemini"
        fake_config.llm_model_name = ""
        fake_config.llm_temperature = 0.0
        fake_config.llm_max_tokens = 1000
        monkeypatch.setattr(fm, "CONFIG", fake_config)
        with pytest.raises(LLMConfigError, match="GOOGLE_API_KEY"):
            build_llm(provider="gemini")

    def test_fallback_raises_when_no_keys(self, monkeypatch):
        import app.llm.factory as fm
        from unittest.mock import MagicMock
        fake_config = MagicMock()
        fake_config.anthropic_api_key = ""
        fake_config.openai_api_key = ""
        fake_config.google_api_key = ""
        fake_config.llm_provider = "anthropic"
        fake_config.llm_model_name = ""
        fake_config.llm_temperature = 0.0
        fake_config.llm_max_tokens = 1000
        monkeypatch.setattr(fm, "CONFIG", fake_config)
        with pytest.raises(LLMConfigError, match="No LLM provider could be configured"):
            build_llm_with_fallback()


class TestDetectAvailableProviders:
    def _fake_config(self, *, anthropic="", openai="", google=""):
        from unittest.mock import MagicMock
        c = MagicMock()
        c.anthropic_api_key = anthropic
        c.openai_api_key = openai
        c.google_api_key = google
        return c

    def test_no_keys_returns_empty(self, monkeypatch):
        import app.llm.factory as fm
        monkeypatch.setattr(fm, "CONFIG", self._fake_config())
        assert detect_available_providers() == []

    def test_detects_anthropic_key(self, monkeypatch):
        import app.llm.factory as fm
        monkeypatch.setattr(fm, "CONFIG", self._fake_config(anthropic="sk-ant-test"))
        result = detect_available_providers()
        assert "anthropic" in result
        assert "openai" not in result

    def test_detects_multiple_keys(self, monkeypatch):
        import app.llm.factory as fm
        monkeypatch.setattr(fm, "CONFIG", self._fake_config(anthropic="sk-ant", openai="sk-oai"))
        result = detect_available_providers()
        assert "anthropic" in result
        assert "openai" in result
        assert "gemini" not in result

    def test_detects_all_three(self, monkeypatch):
        import app.llm.factory as fm
        monkeypatch.setattr(fm, "CONFIG", self._fake_config(
            anthropic="sk-ant", openai="sk-oai", google="AIza"))
        result = detect_available_providers()
        assert set(result) == {"anthropic", "openai", "gemini"}
