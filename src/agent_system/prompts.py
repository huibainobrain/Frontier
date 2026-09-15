"""Prompt registry.

Reads markdown files from ``prompts/`` and performs ``str.format``-style
``{var}`` substitution. Use ``{{`` / ``}}`` to escape literal braces in
prompt text (e.g. for embedded JSON examples).

Caching:
* In normal mode prompt files are read once per process and cached.
* When ``Settings.dev_mode`` is true the cache is bypassed so prompt
  edits take effect without restarting.

Failure mode: ``load_prompt("does_not_exist")`` raises
``FileNotFoundError`` with a list of available prompt names so the
caller sees what they typo'd.
"""

from __future__ import annotations

import functools
from pathlib import Path

from agent_system.config import REPO_ROOT, get_settings

PROMPTS_DIR: Path = REPO_ROOT / "prompts"


@functools.lru_cache(maxsize=64)
def _read_cached(name: str) -> str:
    return _read_uncached(name)


def _read_uncached(name: str) -> str:
    path = PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        available = sorted(p.stem for p in PROMPTS_DIR.glob("*.md"))
        raise FileNotFoundError(
            f"prompt {name!r} not found at {path}. Available: {available}"
        )
    return path.read_text(encoding="utf-8")


def load_prompt(name: str, **vars: object) -> str:
    """Load ``prompts/<name>.md`` and apply ``{var}`` substitution.

    >>> load_prompt("orchestrator_planner", available_sources="anthropic, openai")
    """

    try:
        dev = get_settings().dev_mode
    except Exception:
        # Settings may be unavailable in tests that exercise prompts in
        # isolation; treat that as "not in dev mode" rather than crashing.
        dev = False

    text = _read_uncached(name) if dev else _read_cached(name)
    return text.format(**vars) if vars else text.format()


def reset_prompt_cache() -> None:
    """Drop the in-process prompt cache — used by tests."""

    _read_cached.cache_clear()
