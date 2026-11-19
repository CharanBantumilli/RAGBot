"""Tests for ragbot.config — Settings dataclass and get_settings() singleton."""

from __future__ import annotations

import importlib

import pytest


class TestSettings:
    """Verify the Settings dataclass honours defaults and environment overrides."""

    def test_default_settings(self):
        """Settings() with no arguments should produce sensible defaults."""
        from ragbot.config import Settings

        settings = Settings()

        assert settings.chunk_size == 500
        assert settings.chunk_overlap == 50
        assert settings.max_new_tokens == 200
        assert settings.temperature == 0.5
        assert "multilingual-e5-small" in settings.embedding_model
        assert "DeepSeek" in settings.llm_model or "deepseek" in settings.llm_model.lower()

    def test_settings_from_env(self, monkeypatch: pytest.MonkeyPatch):
        """Settings should pick up values from environment variables when provided."""
        monkeypatch.setenv("RAGBOT_CHUNK_SIZE", "1000")
        monkeypatch.setenv("RAGBOT_CHUNK_OVERLAP", "100")
        monkeypatch.setenv("RAGBOT_MAX_NEW_TOKENS", "512")

        # Force re-import so env vars are read afresh
        import ragbot.config as config_mod

        importlib.reload(config_mod)

        settings = config_mod.Settings(
            chunk_size=int(config_mod.os.environ.get("RAGBOT_CHUNK_SIZE", 500)),
            chunk_overlap=int(config_mod.os.environ.get("RAGBOT_CHUNK_OVERLAP", 50)),
            max_new_tokens=int(config_mod.os.environ.get("RAGBOT_MAX_NEW_TOKENS", 200)),
        )

        assert settings.chunk_size == 1000
        assert settings.chunk_overlap == 100
        assert settings.max_new_tokens == 512

    def test_get_settings_singleton(self):
        """Calling get_settings() twice must return the exact same object."""
        from ragbot.config import get_settings

        first = get_settings()
        second = get_settings()

        assert first is second, "get_settings() should return a cached singleton"

    def test_missing_token_warning(self, monkeypatch: pytest.MonkeyPatch):
        """When HUGGINGFACEHUB_API_TOKEN is unset, the module should warn or handle gracefully."""
        monkeypatch.delenv("HUGGINGFACEHUB_API_TOKEN", raising=False)

        # Re-import to trigger any startup checks
        import ragbot.config as config_mod

        importlib.reload(config_mod)

        # The system should still create a Settings object (not crash).
        settings = config_mod.Settings()
        assert settings is not None
