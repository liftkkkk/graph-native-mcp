"""MCP SDK 兼容导入：同时支持 mcp 1.x 与 2.x。

mcp 2.0 起 FastMCP 更名为 MCPServer，且旧导入路径 mcp.server.fastmcp
被直接移除（不再有弃用别名）。装饰器 @mcp.tool() 与 run(transport=...)
的用法在两个版本间保持一致，因此只需处理导入差异。
本包的四个 server 统一从这里拿 FastMCP。
"""

try:
    from mcp.server.mcpserver import MCPServer as FastMCP  # mcp >= 2.0
except ImportError:  # mcp 1.x（fastmcp 尚未更名）
    from mcp.server.fastmcp import FastMCP

__all__ = ["FastMCP"]
