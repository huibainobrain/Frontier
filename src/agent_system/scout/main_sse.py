import logging
import sys

import uvicorn
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Route

from agent_system.scout.mcp_core import server

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger(__name__)

# The SseServerTransport handles bridging HTTP requests to MCP data streams
sse = SseServerTransport("/messages")

async def handle_sse(request):
    """Endpoint for MCP clients to establish the SSE stream connection."""
    logger.info("New SSE client connection established.")
    async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
        await server.run(streams[0], streams[1], server.create_initialization_options())

async def handle_messages(request):
    """Endpoint for MCP clients to send JSON-RPC messages over HTTP POST."""
    logger.debug("Received client message payload.")
    await sse.handle_post_message(request.scope, request.receive, request._send)

app = Starlette(
    debug=True,
    routes=[
        Route("/sse", endpoint=handle_sse),
        Route("/messages", endpoint=handle_messages, methods=["POST"]),
    ],
)

if __name__ == "__main__":
    logger.info("Starting SSE MCP Server on port 8000...")
    uvicorn.run("agent_system.scout.main_sse:app", host="0.0.0.0", port=8000)