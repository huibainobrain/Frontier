"""A2A client — discover and call remote A2A agents.

Usage::

    from agent_system.a2a.client import A2AClient

    client = A2AClient("http://localhost:8080")
    card = client.discover()           # fetch AgentCard
    task = client.send("Find papers on RLHF")  # send a task
    print(task)                        # completed task with artifacts
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

logger = logging.getLogger(__name__)


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


class A2AClient:
    """Minimal A2A protocol client."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._card: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover(self) -> dict[str, Any]:
        """Fetch the remote agent's AgentCard."""

        import requests

        url = f"{self.base_url}/.well-known/agent.json"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        self._card = resp.json()
        logger.info(
            "Discovered agent: %s (%s)",
            self._card.get("name"),
            self._card.get("version"),
        )
        return self._card

    @property
    def card(self) -> dict[str, Any] | None:
        return self._card

    def list_skills(self) -> list[dict[str, Any]]:
        """Return skills from the discovered AgentCard."""

        if self._card is None:
            self.discover()
        return self._card.get("skills", []) if self._card else []

    # ------------------------------------------------------------------
    # Task execution
    # ------------------------------------------------------------------

    def send(self, text: str, timeout: int = 60) -> dict[str, Any]:
        """Send a text message to the remote agent and return the Task.

        This is a synchronous convenience wrapper around the JSON-RPC
        ``message/send`` method.
        """

        import requests

        message_id = _new_id()
        payload = {
            "jsonrpc": "2.0",
            "id": _new_id(),
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": message_id,
                    "role": "user",
                    "parts": [{"text": text}],
                }
            },
        }

        logger.info("A2A send to %s: %s", self.base_url, text[:80])
        resp = requests.post(self.base_url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()

        if "error" in data:
            raise RuntimeError(f"A2A error: {data['error']}")

        result = data.get("result", data)
        logger.info(
            "A2A task %s: %s",
            result.get("id", "?"),
            result.get("status", {}).get("state", "?"),
        )
        return result

    def get_task(self, task_id: str) -> dict[str, Any]:
        """Fetch a task by ID from the remote agent."""

        import requests

        payload = {
            "jsonrpc": "2.0",
            "id": _new_id(),
            "method": "tasks/get",
            "params": {"id": task_id},
        }
        resp = requests.post(self.base_url, json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if "error" in data:
            raise RuntimeError(f"A2A error: {data['error']}")

        return data.get("result", data)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def search(self, query: str) -> list[dict[str, Any]]:
        """Send a search query and extract results from the artifact."""

        task = self.send(query)
        artifacts = task.get("artifacts", [])
        if not artifacts:
            return []

        import json

        for artifact in artifacts:
            for part in artifact.get("parts", []):
                if "text" in part:
                    try:
                        return json.loads(part["text"])
                    except json.JSONDecodeError:
                        continue
        return []
