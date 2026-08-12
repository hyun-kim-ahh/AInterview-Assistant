"""config.py 계약 확인 테스트 — get_model()이 .env(환경변수) 오버라이드를 호출 시점에
평가하는지 확인한다(모듈 import 시점에 굳혀버리면 monkeypatch.setenv가 이미 import된
모듈엔 반영 안 되는 late-binding 함정에 빠짐 — 함수형이라 이 문제 자체가 없다)."""

from __future__ import annotations

from interview_assistant import config


def test_get_model_returns_default_when_env_var_unset(monkeypatch):
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)

    assert config.get_model() == "anthropic/claude-haiku-4.5"


def test_get_model_returns_env_override_when_set(monkeypatch):
    monkeypatch.setenv("OPENROUTER_MODEL", "anthropic/claude-opus-5")

    assert config.get_model() == "anthropic/claude-opus-5"


def test_get_structuring_model_returns_default_when_env_var_unset(monkeypatch):
    monkeypatch.delenv("OPENROUTER_STRUCTURING_MODEL", raising=False)

    assert config.get_structuring_model() == "anthropic/claude-sonnet-5"


def test_get_structuring_model_returns_env_override_when_set(monkeypatch):
    monkeypatch.setenv("OPENROUTER_STRUCTURING_MODEL", "anthropic/claude-opus-5")

    assert config.get_structuring_model() == "anthropic/claude-opus-5"
