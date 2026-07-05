"""Tests for config.py environment-variable parsing helpers.

`config.Config` field defaults are evaluated once at import time (a
frozen dataclass with module-level singleton), so these tests exercise
the `_env_*` parsing helpers directly with monkeypatched environment
variables rather than re-importing the module per test case.
"""
import importlib

import config as config_module


def test_env_str_returns_default_when_unset(monkeypatch):
    monkeypatch.delenv("RAG_TEST_STR", raising=False)
    assert config_module._env_str("RAG_TEST_STR", "default-value") == "default-value"


def test_env_str_returns_env_value_when_set(monkeypatch):
    monkeypatch.setenv("RAG_TEST_STR", "overridden")
    assert config_module._env_str("RAG_TEST_STR", "default-value") == "overridden"


def test_env_int_returns_default_when_unset(monkeypatch):
    monkeypatch.delenv("RAG_TEST_INT", raising=False)
    assert config_module._env_int("RAG_TEST_INT", 42) == 42


def test_env_int_parses_env_value_when_set(monkeypatch):
    monkeypatch.setenv("RAG_TEST_INT", "123")
    assert config_module._env_int("RAG_TEST_INT", 42) == 123


def test_env_float_returns_default_when_unset(monkeypatch):
    monkeypatch.delenv("RAG_TEST_FLOAT", raising=False)
    assert config_module._env_float("RAG_TEST_FLOAT", 0.5) == 0.5


def test_env_float_parses_env_value_when_set(monkeypatch):
    monkeypatch.setenv("RAG_TEST_FLOAT", "0.25")
    assert config_module._env_float("RAG_TEST_FLOAT", 0.5) == 0.25


def test_config_singleton_has_expected_defaults():
    cfg = config_module.config
    assert cfg.chunk_size > 0
    assert cfg.chunk_overlap >= 0
    assert cfg.retrieval_k > 0
    assert isinstance(cfg.use_hybrid_retrieval, bool)
    assert isinstance(cfg.use_reranking, bool)
    assert cfg.rerank_candidate_k >= cfg.rerank_k


def test_config_env_override_end_to_end(monkeypatch):
    """A full reload of config.py picks up env vars set before import."""
    monkeypatch.setenv("RAG_CHUNK_SIZE", "999")
    monkeypatch.setenv("RAG_USE_HYBRID_RETRIEVAL", "false")
    monkeypatch.setenv("RAG_USE_RERANKING", "true")

    reloaded = importlib.reload(config_module)
    try:
        assert reloaded.config.chunk_size == 999
        assert reloaded.config.use_hybrid_retrieval is False
        assert reloaded.config.use_reranking is True
    finally:
        # Restore the module to its environment-default state for other tests.
        monkeypatch.undo()
        importlib.reload(config_module)
