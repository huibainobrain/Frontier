"""Tests for agent_system.config — the AGENTIC_REPLAN_ENABLED ablation
switch and Settings.for_eval_run isolation, both added for Eval
Readiness."""

from __future__ import annotations

from pathlib import Path

from agent_system.config import Settings
from agent_system.schemas import RawPost
from agent_system.storage import db as storage_db


def _raw_post(post_id: str) -> RawPost:
    return RawPost(
        post_id=post_id,
        source="anthropic",
        url=f"https://example.com/{post_id}",
        title=f"Post {post_id}",
        authors=[],
        published_at="2026-04-25",
        content="Body.",
        content_type="blog",
    )


class TestAgenticReplanEnabled:
    def test_defaults_true(self):
        s = Settings(google_api_key="k")
        assert s.agentic_replan_enabled is True

    def test_env_false_disables(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "k")
        monkeypatch.setenv("AGENTIC_REPLAN_ENABLED", "false")
        s = Settings.from_env()
        assert s.agentic_replan_enabled is False

    def test_env_true_enables(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "k")
        monkeypatch.setenv("AGENTIC_REPLAN_ENABLED", "true")
        s = Settings.from_env()
        assert s.agentic_replan_enabled is True

    def test_env_unset_defaults_true(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "k")
        monkeypatch.delenv("AGENTIC_REPLAN_ENABLED", raising=False)
        s = Settings.from_env()
        assert s.agentic_replan_enabled is True


class TestEvalRunIsolation:
    def test_for_eval_run_isolates_storage_paths(self, tmp_path, monkeypatch):
        """Test C3: two variants (or two cases) of an eval run get
        distinct SQLite/Chroma/token-log paths, and a post cached under
        one variant is invisible to the other — a later-run variant
        must not inherit an earlier one's cache/RAG memory."""

        monkeypatch.setenv("GOOGLE_API_KEY", "k")
        workflow_settings = Settings.for_eval_run(
            "workflow", "case_001", base_dir=tmp_path
        )
        agent_settings = Settings.for_eval_run("agent", "case_001", base_dir=tmp_path)

        assert workflow_settings.sqlite_path != agent_settings.sqlite_path
        assert workflow_settings.chroma_dir != agent_settings.chroma_dir
        assert workflow_settings.token_log_path != agent_settings.token_log_path

        storage_db.save_raw_post(_raw_post("only-in-workflow"), settings=workflow_settings)

        workflow_posts = storage_db.list_raw_posts(settings=workflow_settings)
        agent_posts = storage_db.list_raw_posts(settings=agent_settings)
        assert {p.post_id for p in workflow_posts} == {"only-in-workflow"}
        assert agent_posts == []  # the other variant sees nothing

    def test_for_eval_run_isolates_across_cases(self, tmp_path, monkeypatch):
        """A later case in the same variant must not inherit an earlier
        case's cache either — case order must not affect results."""

        monkeypatch.setenv("GOOGLE_API_KEY", "k")
        case1 = Settings.for_eval_run("agent", "case_001", base_dir=tmp_path)
        case2 = Settings.for_eval_run("agent", "case_002", base_dir=tmp_path)

        storage_db.save_raw_post(_raw_post("from-case-1"), settings=case1)

        assert storage_db.list_raw_posts(settings=case2) == []

    def test_for_eval_run_keeps_non_storage_settings_from_env(self, tmp_path, monkeypatch):
        """Only the four storage paths are redirected — everything else
        (model names, agentic_replan_enabled, thresholds) still comes
        from the normal env, so a variant's own config (e.g.
        AGENTIC_REPLAN_ENABLED=false for the workflow variant) is
        respected."""

        monkeypatch.setenv("GOOGLE_API_KEY", "k")
        monkeypatch.setenv("AGENTIC_REPLAN_ENABLED", "false")
        s = Settings.for_eval_run("workflow", "case_001", base_dir=tmp_path)

        assert s.agentic_replan_enabled is False
        assert s.sqlite_path == Path(tmp_path) / "workflow" / "case_001" / "agent_system.sqlite"
