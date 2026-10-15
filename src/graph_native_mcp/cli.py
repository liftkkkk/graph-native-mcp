"""CLI entry point: `gnm <server>` starts one of the four MCP servers.

    gnm memory   # graph-memory  (Ch.6)
    gnm rag      # graph-rag     (Ch.5)
    gnm build    # graph-build   (Ch.7, needs GNM_LLM_*)
    gnm agent    # graph-agent   (Ch.8)

All servers share one SQLite store (GNM_DB, default ~/.gnm/graph.db).
"""

from __future__ import annotations

import argparse
import sys

SERVERS = {
    "memory": "graph_native_mcp.servers.memory",
    "rag": "graph_native_mcp.servers.rag",
    "build": "graph_native_mcp.servers.build",
    "agent": "graph_native_mcp.servers.agent",
}


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="gnm",
        description="graph-native-mcp: four graph-native MCP servers over one shared store")
    parser.add_argument("server", choices=sorted(SERVERS),
                        help="which MCP server to start (stdio transport)")
    parser.add_argument("--db", default=None,
                        help="override SQLite path (default: $GNM_DB or ~/.gnm/graph.db)")
    args = parser.parse_args(argv)

    if args.db:
        import os
        os.environ["GNM_DB"] = args.db

    import importlib
    mod = importlib.import_module(SERVERS[args.server])
    mod.run()


if __name__ == "__main__":
    main()
