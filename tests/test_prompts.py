"""Prompt-registry tests."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent_system import prompts as prompts_module
from agent_system.config import Settings


@pytest.fixture
def settings_dev(tmp_path):
    return Settings(
        google_api_key="x",
        dev_mode=True,
        data_dir=tmp_path,
        sqlite_path=tmp_path / "x.sqlite",
        chroma_dir=tmp_path / "chroma",
        token_log_path=tmp_path / "token_log.jsonl",
    )


def test_load_prompt_returns_real_orchestrator_intent(settings_dev):
    prompts_module.reset_prompt_cache()
    with patch("agent_system.prompts.get_settings", return_value=settings_dev):
        text = prompts_module.load_prompt("orchestrator_intent")
    assert "Intent Classifier" in text
    # JSON braces survive {{ → { un-escaping in str.format.
    assert '{"intent"' in text


def test_load_prompt_substitutes_variables(settings_dev):
    prompts_module.reset_prompt_cache()
    with patch("agent_system.prompts.get_settings", return_value=settings_dev):
        text = prompts_module.load_prompt(
            "orchestrator_planner",
            available_sources="anthropic, openai, deepmind",
        )
    assert "anthropic, openai, deepmind" in text


def test_load_prompt_unknown_name_raises_with_helpful_message(settings_dev):
    prompts_module.reset_prompt_cache()
    with patch("agent_system.prompts.get_settings", return_value=settings_dev):
        with pytest.raises(FileNotFoundError) as exc:
            prompts_module.load_prompt("does_not_exist")
    msg = str(exc.value)
    assert "does_not_exist" in msg
    assert "Available" in msg
