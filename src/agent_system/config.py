"""Runtime configuration for the agent system.

Loads environment variables from ``.env`` (via ``python-dotenv``) and
exposes a frozen-ish ``Settings`` dataclass containing API keys, model
names, data paths, and behavioural thresholds. Agents import
``Settings`` rather than reading env vars directly so tests can inject
alternative configurations.

Loud-failure contract: if ``GOOGLE_API_KEY`` is missing the very first
``get_settings()`` call raises ``RuntimeError`` with a message pointing
the user at ``.env.example``. Other keys (Langfuse) are optional — the
observability layer falls back to local-only token logging when they
are absent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

try:
    from dotenv import load_dotenv  # type: ignore
except ImportError:  # python-dotenv not installed yet — env vars only
    def load_dotenv(*_args, **_kwargs):  # type: ignore
        return False

# repo root: src/agent_system/config.py -> parents[2]
REPO_ROOT: Path = Path(__file__).resolve().parents[2]


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _resolve(p: Path | str) -> Path:
    """Resolve a path, treating relative paths as relative to REPO_ROOT."""

    pp = Path(p)
    return pp if pp.is_absolute() else (REPO_ROOT / pp)


def _parse_int(value: str | None, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return default


@dataclass
class Settings:
    """Runtime configuration. Construct via :meth:`from_env`."""

    google_api_key: str

    # gemini-2.0-flash / gemini-2.5-pro / text-embedding-004 were the
    # original defaults but are retired by Google as of 2026-09 (confirmed
    # live via a real API call). Real pro-tier models (e.g.
    # gemini-3.1-pro-preview) need billing enabled on the Google AI Studio
    # project — the free tier's quota for pro-tier text generation is 0,
    # not "limited". This is a portfolio/demo project (getting it to run
    # end-to-end matters more than output quality or a "real" cost tier
    # split), so model_pro defaults to the same free-tier flash-lite model
    # as model_flash. The two settings stay separate so the cost-tiering
    # design is still legible — point model_pro at a real pro model in
    # .env once billing is enabled, no code changes needed.
    model_flash: str = "gemini-3.1-flash-lite"
    model_pro: str = "gemini-3.1-flash-lite"
    embedding_model: str = "gemini-embedding-001"

    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    data_dir: Path = field(default_factory=lambda: REPO_ROOT / "data")
    sqlite_path: Path = field(
        default_factory=lambda: REPO_ROOT / "data" / "agent_system.sqlite"
    )
    chroma_dir: Path = field(default_factory=lambda: REPO_ROOT / "data" / "chroma")
    token_log_path: Path = field(
        default_factory=lambda: REPO_ROOT / "data" / "token_log.jsonl"
    )

    max_revisions: int = 2
    critic_unsupported_threshold: int = 1
    # Deterministic hard cap on the Orchestrator's research loop (Scout
    # fetch -> Analyst -> Evaluator -> replan?). The Evaluator (an LLM
    # call) decides whether evidence is sufficient and what to search
    # for next, but never how many rounds it's allowed — this is the
    # workflow-owned budget boundary that overrides it regardless. 1
    # reproduces the original single-shot behavior (no replanning); 2
    # allows exactly one targeted replan round.
    max_research_rounds: int = 2
    # Global hard cap on total *new unique* posts analyzed across every
    # research round combined — max_posts alone only bounds a single
    # round, so max_research_rounds rounds could otherwise add up to
    # max_research_rounds * max_posts. Duplicates (by post_id or
    # canonical URL) never count against this — only genuinely new
    # evidence consumes budget.
    max_total_posts: int = 30
    dev_mode: bool = False
    # Ablation switch for Workflow-vs-Agent Eval: True (default) runs the
    # current agentic loop (Evaluator judges sufficiency, may trigger a
    # targeted replan round). False is the Fixed Workflow baseline —
    # Intent -> Planner -> Scout -> Analyst -> Synthesizer -> Critic ->
    # Output, with the Evaluator never consulted at all (not called-and-
    # ignored: literally zero calls), so a benchmark comparing the two
    # variants isn't measuring "Agent vs. Agent capped at 1 round" but
    # "Agent vs. a workflow that genuinely never judges or replans". See
    # Orchestrator._run_research_loop.
    agentic_replan_enabled: bool = True

    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls, env_file: Path | str | None = None) -> Settings:
        """Build a Settings from environment variables.

        Priority: explicit ``env_file`` > project ``.env`` > current
        process environment. Missing ``GOOGLE_API_KEY`` raises.
        """

        if env_file is not None:
            load_dotenv(Path(env_file), override=False)
        else:
            default_env = REPO_ROOT / ".env"
            if default_env.exists():
                load_dotenv(default_env, override=False)

        google_api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
        if not google_api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is required but missing. "
                "Copy .env.example to .env and fill in your Gemini API key."
            )

        data_dir = _resolve(os.environ.get("DATA_DIR", "data"))
        sqlite_path = _resolve(
            os.environ.get("SQLITE_PATH", str(data_dir / "agent_system.sqlite"))
        )
        chroma_dir = _resolve(
            os.environ.get("CHROMA_DIR", str(data_dir / "chroma"))
        )
        token_log_path = _resolve(
            os.environ.get("TOKEN_LOG_PATH", str(data_dir / "token_log.jsonl"))
        )

        return cls(
            google_api_key=google_api_key,
            model_flash=os.environ.get("GEMINI_MODEL_FLASH", "gemini-3.1-flash-lite"),
            model_pro=os.environ.get("GEMINI_MODEL_PRO", "gemini-3.1-flash-lite"),
            embedding_model=os.environ.get(
                "GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"
            ),
            langfuse_public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip(),
            langfuse_secret_key=os.environ.get("LANGFUSE_SECRET_KEY", "").strip(),
            langfuse_host=os.environ.get(
                "LANGFUSE_HOST", "https://cloud.langfuse.com"
            ).strip(),
            data_dir=data_dir,
            sqlite_path=sqlite_path,
            chroma_dir=chroma_dir,
            token_log_path=token_log_path,
            max_revisions=_parse_int(os.environ.get("MAX_REVISIONS"), 2),
            critic_unsupported_threshold=_parse_int(
                os.environ.get("CRITIC_UNSUPPORTED_THRESHOLD"), 1
            ),
            max_research_rounds=_parse_int(os.environ.get("MAX_RESEARCH_ROUNDS"), 2),
            max_total_posts=_parse_int(os.environ.get("MAX_TOTAL_POSTS"), 30),
            dev_mode=_truthy(os.environ.get("DEV_MODE")),
            agentic_replan_enabled=(
                _truthy(os.environ["AGENTIC_REPLAN_ENABLED"])
                if "AGENTIC_REPLAN_ENABLED" in os.environ
                else True
            ),
        )

    @classmethod
    def for_eval_run(
        cls,
        variant: str,
        case_id: str,
        *,
        base_dir: Path | str | None = None,
        env_file: Path | str | None = None,
    ) -> Settings:
        """Settings isolated for one Eval run — e.g. comparing a
        ``"workflow"`` (agentic_replan_enabled=False) variant against an
        ``"agent"`` variant on the same case, without either one's
        SQLite cache / Chroma RAG memory / telemetry leaking into the
        other, and without case order affecting results (a later case
        must not inherit an earlier one's accumulated cache).

        Everything else (API key, model names, thresholds,
        agentic_replan_enabled, ...) still comes from the normal
        env/``.env`` via :meth:`from_env` — only the four storage paths
        are redirected, reusing the same env-var override mechanism
        ``from_env`` already supports rather than a separate storage
        architecture. Each call builds a fresh ``Settings`` (bypassing
        the ``get_settings()`` singleton on purpose) so a caller running
        multiple variants/cases in one process gets a distinct instance
        per run to hand to its own ``Orchestrator``.
        """

        base = _resolve(base_dir) if base_dir is not None else REPO_ROOT / "eval_runs"
        run_dir = Path(base) / variant / case_id
        settings = cls.from_env(env_file)
        settings.data_dir = run_dir
        settings.sqlite_path = run_dir / "agent_system.sqlite"
        settings.chroma_dir = run_dir / "chroma"
        settings.token_log_path = run_dir / "token_log.jsonl"
        return settings

    @property
    def has_langfuse(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    def ensure_dirs(self) -> None:
        """Create on-disk dirs lazily (``data/`` is gitignored)."""

        for p in (self.data_dir, self.chroma_dir, self.sqlite_path.parent):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide singleton. Loaded on first access; raises if invalid."""

    return Settings.from_env()


def reset_settings_cache() -> None:
    """Drop the cached Settings — used by tests that mutate env vars."""

    get_settings.cache_clear()
