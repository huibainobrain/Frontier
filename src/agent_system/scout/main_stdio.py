import asyncio
import logging
import sys

from mcp.server.stdio import stdio_server

from agent_system.scout.mcp_core import server

# CRITICAL: For standard IO transport, all logging MUST go to stderr.
# Outputting to stdout will corrupt the JSON-RPC payload expected by MCP clients.
logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Start the MCP standard input/output server."""
    logger.info("Initializing Stdio MCP server for Scout Agent...")
    
    async with stdio_server() as (read_stream, write_stream):
        logger.info("Stdio transport ready. Listening for client JSON-RPC requests...")
        await server.run(read_stream, write_stream, server.create_initialization_options())

if __name__ == "__main__":
    # Under Windows environment, specific event loop policies might be required for Stdio,
    # but standard asyncio.run works for standard pipes out-of-the-box in modern Python.
    asyncio.run(main())