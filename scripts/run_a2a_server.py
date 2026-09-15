"""Start the A2A server for external agent interop.

Usage:
    python scripts/run_a2a_server.py              # default port 8080
    python scripts/run_a2a_server.py --port 9090  # custom port
"""

import argparse
import logging

from agent_system.a2a.server import A2AServer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="A2A server for FrontierLitAgent")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8080, help="Listen port")
    args = parser.parse_args()

    server = A2AServer(host=args.host, port=args.port)
    print(f"A2A AgentCard: http://{args.host}:{args.port}/.well-known/agent.json")
    print(f"A2A JSON-RPC:  http://{args.host}:{args.port}/")
    server.run()


if __name__ == "__main__":
    main()
