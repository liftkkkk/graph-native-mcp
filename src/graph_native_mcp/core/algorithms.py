"""Classic graph algorithms over an in-memory networkx view (book Ch.2).

These are the deterministic workhorses behind the three retrieval modes
(Ch.5.3) and skill-graph planning (Ch.8.3):
  - k_hop_subgraph      -> mode 1 (local subgraph) with fan-out budgets
  - detect_communities  -> mode 2 (community summaries), Louvain as the
                           Leiden stand-in (connectivity guarantee noted in README)
  - top_k_paths         -> mode 3 (path reasoning), k shortest simple paths
  - personalized_pagerank -> HippoRAG-style structural voting
"""

from __future__ import annotations

import networkx as nx


def to_networkx(nodes: list[dict], edges: list[dict]) -> nx.DiGraph:
    g = nx.DiGraph()
    for n in nodes:
        g.add_node(n["node_id"], **{k: v for k, v in n.items() if k != "embedding"})
    for e in edges:
        g.add_edge(e["src"], e["dst"], **{"edge_type": e["edge_type"],
                                          "edge_id": e.get("edge_id"),
                                          "properties": e.get("properties", {})})
    return g


def k_hop_subgraph(g: nx.DiGraph, seeds: list[str], hops: int = 2,
                   max_nodes: int = 40, max_edges: int = 80) -> tuple[set, list]:
    """Budgeted BFS from seed nodes. Expansion stops at the first budget
    breach, so the cost model stays observable (Ch.5.3.1 枢纽节点防爆)."""
    frontier, seen, used_edges = set(seeds), set(seeds), []
    as_undirected = g.to_undirected(as_view=True)
    for _ in range(max(1, hops)):
        nxt, layer_edges = set(), []
        for nid in frontier:
            for nb in as_undirected.neighbors(nid):
                if nb not in seen:
                    layer_edges.append((nid, nb) if g.has_edge(nid, nb) else (nb, nid))
                    nxt.add(nb)
        if len(seen) + len(nxt) > max_nodes or len(used_edges) + len(layer_edges) > max_edges:
            room = max(0, max_nodes - len(seen))
            nxt = set(list(nxt)[:room])
            layer_edges = layer_edges[:max(0, max_edges - len(used_edges))]
        for a, b in layer_edges:
            if a in seen or a in nxt or b in seen or b in nxt:
                used_edges.append((a, b))
        seen |= nxt
        frontier = nxt
        if not frontier:
            break
    return seen, used_edges


def personalized_pagerank(g: nx.DiGraph, seeds: list[str],
                          damping: float = 0.85) -> dict:
    """PPR with uniform personalization over the seed set. Nodes relevant to
    the query 'float up' by structural importance (HippoRAG)."""
    if not seeds:
        return {}
    ug = g.to_undirected(as_view=True)
    if ug.number_of_nodes() == 0:
        return {}
    personalization = {s: 1.0 / len(seeds) for s in seeds if ug.has_node(s)}
    if not personalization:
        return {}
    return nx.pagerank(ug, alpha=damping, personalization=personalization)


def top_k_paths(g: nx.DiGraph, source: str, target: str, k: int = 3,
                max_len: int = 5) -> list[list[str]]:
    """k shortest simple paths up to length max_len. Each path is a reasoning
    chain candidate (Ch.5.3.3)."""
    if not (g.has_node(source) and g.has_node(target)):
        return []
    ug = g.to_undirected(as_view=True)
    try:
        paths = nx.shortest_simple_paths(ug, source, target)
        out = []
        for p in paths:
            if len(p) - 1 > max_len:
                break
            out.append(p)
            if len(out) >= k:
                break
        return out
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


def detect_communities(g: nx.Graph, seed: int = 42) -> list[set]:
    """Louvain communities (Leiden stand-in). Deterministic via seed."""
    if g.number_of_nodes() == 0:
        return []
    try:
        communities = nx.community.louvain_communities(
            g, seed=seed, resolution=1.0)
        return [c for c in communities if len(c) > 0]
    except Exception:
        return [set(g.nodes())]
