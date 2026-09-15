"""Agent-to-Agent (A2A) interoperability layer.

Implements the Google A2A protocol for cross-agent communication:

* :class:`A2AServer` — exposes our Scout agent as an A2A service.
  Other agents can discover us via ``/.well-known/agent.json``
  and send tasks via JSON-RPC 2.0 ``message/send``.

* :class:`A2AClient` — discovers and calls remote A2A agents to
  delegate tasks our pipeline can't handle alone.

Protocol reference: https://a2a-protocol.org/latest/specification/
"""

from agent_system.a2a.client import A2AClient
from agent_system.a2a.server import A2AServer

__all__ = ["A2AServer", "A2AClient"]