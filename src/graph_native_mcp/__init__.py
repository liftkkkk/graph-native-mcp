"""graph-native-mcp: graph-native agent capabilities as MCP servers.

Four servers, one shared store (gxt-core):
  graph_memory  - bi-temporal memory graph        (book Ch.6)
  graph_rag     - three retrieval modes, read-only (book Ch.5)
  graph_build   - text->graph / text->query / checkup (book Ch.7, needs LLM)
  graph_agent   - skill graph + A* planning + world model  (book Ch.8)
"""

__version__ = "0.1.0"
