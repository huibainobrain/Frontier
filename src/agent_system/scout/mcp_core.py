import asyncio
import logging
from typing import Any

import mcp.types as types
from mcp.server import Server

from agent_system.scout.agent import Scout

logger = logging.getLogger(__name__)

# Initialize the core MCP Server instance
server = Server("scout-mcp-server")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """Register the literature search tool for the MCP client."""
    return [
        types.Tool(
            name="search_literature",
            description="Search for recent AI literature, blog posts, and papers across multiple sources (arXiv, Semantic Scholar, OpenAlex, etc.).",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query or topic to investigate (e.g., 'Agentic workflows', 'LLM reasoning')."
                    },
                    "time_window_days": {
                        "type": "integer",
                        "description": "Number of days back to search for recent literature. Defaults to 14."
                    }
                },
                "required": ["query"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict[str, Any] | None) -> list[types.TextContent]:
    """Handle incoming tool execution requests from the MCP client."""
    if name != "search_literature":
        raise ValueError(f"Unsupported tool requested: {name}")

    if not arguments or "query" not in arguments:
        raise ValueError("Missing required argument: 'query'")

    query = arguments["query"]
    time_window_days = arguments.get("time_window_days", 14)

    logger.info("Handling tool call 'search_literature' with query: '%s', time_window: %d", query, time_window_days)

    try:
        # Offload synchronous network operations to a thread to avoid blocking the MCP event loop
        result_text = await asyncio.to_thread(_run_scout_search, query, time_window_days)
        return [types.TextContent(type="text", text=result_text)]
    except Exception as exc:
        logger.exception("Failed to execute literature search.")
        return [types.TextContent(type="text", text=f"Search execution failed: {exc}")]

def _run_scout_search(query: str, time_window_days: int) -> str:
    """Synchronous execution wrapper for the Scout agent."""
    scout = Scout()

    # Scout expects a UserProfile, but logic in plan_sources currently bypasses it. Pass None safely.
    plan = scout.plan_sources(query, None) # type: ignore
    plan.time_window_days = time_window_days

    posts = scout.fetch(plan)
    if not posts:
        return "No relevant literature found for the given search query."

    formatted_results = []
    for idx, post in enumerate(posts, 1):
        authors_str = ", ".join(post.authors) if post.authors else "Unknown Authors"
        summary = post.content[:1000] + "..." if len(post.content) > 1000 else post.content
        
        formatted_results.append(
            f"Result [{idx}]\n"
            f"Title: {post.title}\n"
            f"Authors: {authors_str}\n"
            f"Published: {post.published_at} (Source: {post.source})\n"
            f"URL: {post.url}\n"
            f"Summary: {summary}\n"
        )

    return "\n---\n".join(formatted_results)