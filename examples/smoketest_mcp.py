"""End-to-end MCP stdio smoke test.

Starts each server as a real subprocess and drives it through the MCP
client protocol: initialize handshake -> list tools -> call one tool per
server. graph-build's LLM tools are skipped unless GNM_LLM_* is configured
(graph_checkup needs no LLM and is always exercised).

Usage:
    python examples/smoketest_mcp.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

DB = os.path.join(tempfile.mkdtemp(prefix="gnm_smoke_"), "graph.db")
HERE = os.path.dirname(os.path.abspath(__file__))


async def drive(server_name: str, calls: list[tuple[str, dict]]):
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "graph_native_mcp.cli", server_name, "--db", DB],
        env={**os.environ, "GNM_DB": DB, "PYTHONPATH": os.path.abspath(src)},
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = [t.name for t in tools.tools]
            print(f"[{server_name}] handshake ok, {len(names)} tools: {', '.join(names)}")
            for tool_name, args in calls:
                res = await session.call_tool(tool_name, args)
                text = (res.content[0].text if res.content else "")[:120]
                print(f"  -> {tool_name}: {'ok' if not res.isError else 'ERROR'} {text}")


async def main():
    print(f"smoke db: {DB}\n")

    await drive("memory", [
        ("write_memory", {"content": "张三担任Acme公司CEO", "memory_type": "Fact",
                          "entities": ["张三", "Acme公司"], "source_ids": ["doc#1"]}),
        ("write_memory", {"content": "张三担任Acme公司CTO", "memory_type": "Fact",
                          "entities": ["张三", "Acme公司"], "source_ids": ["doc#2"]}),
        ("semantic_search", {"query": "张三的职务", "top_k": 3}),
        ("graph_stats", {}),
    ])

    await drive("rag", [
        ("search_local", {"query": "张三在Acme公司担任什么职务"}),
        ("build_communities", {"min_community": 1}),
        ("search_global", {"query": "材料主要讲什么"}),
        ("search_paths", {"query": "职务关系", "source_entity": "张三",
                          "target_entity": "Acme公司"}),
    ])

    await drive("build", [
        ("graph_checkup", {}),
    ])

    await drive("agent", [
        ("set_world_state", {"state": {"库存.苹果": 5, "资金": 100}}),
        ("register_skill", {"name": "采购苹果",
                            "precondition": [{"key": "资金", "op": "ge", "value": 50}],
                            "effect": [{"key": "库存.苹果", "op": "add", "value": 10},
                                       {"key": "资金", "op": "sub", "value": 50}]}),
        ("register_skill", {"name": "卖出苹果",
                            "precondition": [{"key": "库存.苹果", "op": "ge", "value": 10}],
                            "effect": [{"key": "库存.苹果", "op": "sub", "value": 10},
                                       {"key": "资金", "op": "add", "value": 80}]}),
        ("plan", {"goal": [{"key": "资金", "op": "ge", "value": 120}]}),
        ("replay", {}),
    ])

    print("\nAll servers responded over MCP stdio.")


if __name__ == "__main__":
    asyncio.run(main())
