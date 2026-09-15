"""A2A protocol data models.

Minimal representations of the A2A specification objects:
AgentCard, Task, Message, Artifact, Part, TaskStatus.

Spec: https://a2a-protocol.org/latest/specification/
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskState(str, Enum):
    """A2A task lifecycle states."""

    SUBMITTED = "TASK_STATE_SUBMITTED"
    WORKING = "TASK_STATE_WORKING"
    COMPLETED = "TASK_STATE_COMPLETED"
    FAILED = "TASK_STATE_FAILED"
    CANCELED = "TASK_STATE_CANCELED"
    INPUT_REQUIRED = "TASK_STATE_INPUT_REQUIRED"


# ---------------------------------------------------------------------------
# AgentCard
# ---------------------------------------------------------------------------


@dataclass
class AgentCapability:
    streaming: bool = False
    push_notifications: bool = False

    def to_dict(self) -> dict[str, bool]:
        return {
            "streaming": self.streaming,
            "pushNotifications": self.push_notifications,
        }


@dataclass
class AgentSkill:
    id: str
    name: str
    description: str
    tags: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "examples": self.examples,
        }


@dataclass
class AgentCard:
    """Describes an A2A agent for discovery."""

    name: str
    description: str
    url: str
    version: str = "0.1.0"
    capabilities: AgentCapability = field(default_factory=AgentCapability)
    skills: list[AgentSkill] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "url": self.url,
            "version": self.version,
            "capabilities": self.capabilities.to_dict(),
            "skills": [s.to_dict() for s in self.skills],
        }


# ---------------------------------------------------------------------------
# Task / Message / Artifact / Part
# ---------------------------------------------------------------------------


@dataclass
class Part:
    """A content part within a Message or Artifact."""

    text: str | None = None
    data: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.text is not None:
            d["text"] = self.text
        if self.data is not None:
            d["data"] = self.data
        if self.metadata:
            d["metadata"] = self.metadata
        return d


@dataclass
class Message:
    """An A2A message (user or agent)."""

    message_id: str
    role: str  # "user" | "agent"
    parts: list[Part] = field(default_factory=list)
    context_id: str | None = None
    task_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "messageId": self.message_id,
            "role": self.role,
            "parts": [p.to_dict() for p in self.parts],
        }
        if self.context_id:
            d["contextId"] = self.context_id
        if self.task_id:
            d["taskId"] = self.task_id
        return d


@dataclass
class Artifact:
    """Output artifact attached to a completed Task."""

    artifact_id: str
    name: str = ""
    description: str = ""
    parts: list[Part] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "artifactId": self.artifact_id,
            "parts": [p.to_dict() for p in self.parts],
        }
        if self.name:
            d["name"] = self.name
        if self.description:
            d["description"] = self.description
        return d


@dataclass
class TaskStatus:
    """Current state of a Task."""

    state: TaskState
    timestamp: str = ""
    message: Message | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"state": self.state.value}
        if self.timestamp:
            d["timestamp"] = self.timestamp
        if self.message:
            d["message"] = self.message.to_dict()
        return d


@dataclass
class Task:
    """A2A Task — the unit of work between agents."""

    id: str
    status: TaskStatus
    context_id: str | None = None
    artifacts: list[Artifact] = field(default_factory=list)
    history: list[Message] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "status": self.status.to_dict(),
        }
        if self.context_id:
            d["contextId"] = self.context_id
        if self.artifacts:
            d["artifacts"] = [a.to_dict() for a in self.artifacts]
        if self.history:
            d["history"] = [m.to_dict() for m in self.history]
        return d