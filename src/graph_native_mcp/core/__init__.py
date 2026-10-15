"""gxt-core: shared foundation for all graph-native MCP servers.

Implements the infrastructure layer described in the book:
  - SQLite property-graph store with bi-temporal timestamps (Ch.1 / Ch.6)
  - node/edge schema and the write gate (Ch.6 / Ch.7)
  - evidence serialization format with citation ids (Ch.5)
  - embedding interface with offline deterministic fallback (Ch.3-lite)
  - graph algorithms: k-hop, PPR, k-shortest paths, communities (Ch.2)
"""

from .schema import NODE_TYPES, EDGE_TYPES, validate_node, validate_edge
from .store import GraphStore, default_db_path
from .serialize import serialize_evidence, estimate_tokens
from .embeddings import get_embedding_provider
from .algorithms import (
    k_hop_subgraph,
    personalized_pagerank,
    top_k_paths,
    detect_communities,
)

__all__ = [
    "NODE_TYPES", "EDGE_TYPES", "validate_node", "validate_edge",
    "GraphStore", "default_db_path",
    "serialize_evidence", "estimate_tokens",
    "get_embedding_provider",
    "k_hop_subgraph", "personalized_pagerank", "top_k_paths", "detect_communities",
]
