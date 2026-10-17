"""graph-rag MCP server (book Ch.5) — three retrieval modes, read-only.

The five-stage skeleton (seed -> expand -> filter/verify -> serialize ->
generate) is split across the MCP boundary exactly as the book argues:
this server does the deterministic first four stages and hands the client
LLM-ready evidence with citation ids; generation stays with the caller
(Ch.5.2.1 "检索管找证据，模型管下判断").

Tools map 1:1 to the three retrieval modes (Ch.5.3):
  search_local    mode 1: k-hop neighborhood subgraph
  search_global   mode 2: community summaries (needs build_communities first)
  search_paths    mode 3: k shortest paths / PPR surfacing
Every tool returns the audit triple: evidence text + citation closure +
cost report (nodes / edges / est_tokens).

This server never calls an LLM. Community summaries are deterministic
(representatives + frequent relations); LLM-polished summaries are roadmap.
"""

from __future__ import annotations

from collections import Counter

from graph_native_mcp._fastmcp_compat import FastMCP

from ..core import algorithms as alg
from ..core.serialize import serialize_evidence, cost_report
from ..core.store import GraphStore, default_db_path

mcp = FastMCP("graph-rag")
_store: GraphStore | None = None


def store() -> GraphStore:
    global _store
    if _store is None:
        _store = GraphStore(default_db_path())
    return _store


def _load_graph():
    s = store()
    nodes = [dict(n, embedding=None) for n in _all_active_nodes(s)]
    edges = _all_active_edges(s)
    return alg.to_networkx(nodes, edges), nodes, edges


def _all_active_nodes(s):
    return [s._row_to_node(r) for r in s.conn.execute(
        "SELECT * FROM nodes WHERE superseded_by IS NULL")]


def _all_active_edges(s):
    return [s._row_to_edge(r) for r in s.conn.execute(
        "SELECT * FROM edges WHERE valid_until IS NULL")]


def _seed_nodes(query: str, top_k: int) -> list[str]:
    hits = store().semantic_search(query, top_k=top_k)
    return [h["node_id"] for h in hits]


# ================================================================ mode 1
@mcp.tool()
def search_local(query: str, hops: int = 2, max_nodes: int = 40,
                 max_edges: int = 80, node_types: list = None) -> dict:
    """模式一：局部子图检索（Ch.5.3.1），适合事实型/关系型问题（谁是谁的什么）。
    语义检索定种子 → 预算内 k 跳扩展 → 过滤 → 序列化为带引用 id 的证据。
    - query: 自然语言问题
    - hops: 扩展跳数（默认 2；枢纽节点由 max_nodes/max_edges 预算兜底）
    - max_nodes / max_edges: 图扩展预算（代价模型的可观测落点）
    - node_types: 只保留这些类型的节点（如 ["Fact","Event","Entity"]）
    返回 {evidence, citations, cost, seeds}。证据里每行都带 node/edge id，
    生成端的引用必须落在 citations 闭包内。"""
    s = store()
    seeds = _seed_nodes(query, top_k=5)
    if not seeds:
        return {"evidence": "(no seed nodes found)", "citations": [],
                "cost": {"nodes": 0, "edges": 0, "est_tokens": 0}, "seeds": []}
    g, nodes, edges = _load_graph()
    node_ids, used_edges = alg.k_hop_subgraph(g, seeds, hops=hops,
                                              max_nodes=max_nodes, max_edges=max_edges)
    if node_types:
        keep = {n["node_id"] for n in nodes
                if n["node_id"] in node_ids and n["node_type"] in node_types}
        seeds_kept = keep & set(seeds)
        node_ids = keep | seeds_kept
    node_dicts = [n for n in nodes if n["node_id"] in node_ids]
    edge_dicts = [e for e in edges if e["src"] in node_ids and e["dst"] in node_ids]
    evidence = serialize_evidence(node_dicts, edge_dicts)
    s.log_event("retrieve", {"tool": "search_local", "query": query,
                             "seeds": seeds, **cost_report(node_dicts, edge_dicts, evidence)})
    return {"evidence": evidence,
            "citations": {"node_ids": sorted(node_ids),
                          "edge_ids": [e["edge_id"] for e in edge_dicts]},
            "cost": cost_report(node_dicts, edge_dicts, evidence),
            "seeds": seeds}


# ================================================================ mode 2
@mcp.tool()
def build_communities(min_community: int = 3) -> dict:
    """离线索引（模式二的前置）：Louvain 社区划分（Leiden 的工程替身）+
    每个社区的确定性摘要（代表节点 + 高频关系 + 关键词）。
    - min_community: 小于该规模的社区并入「杂项」
    写入的是派生索引（communities 表），不动事实数据。
    图更新后需要重跑（摘要重算成本，见 Ch.5.3.2）。"""
    s = store()
    g, nodes, edges = _load_graph()
    if g.number_of_nodes() == 0:
        return {"error": "graph is empty"}
    communities = alg.detect_communities(g)
    name_of = {n["node_id"]: n["name"] or n["content"][:30] for n in nodes}
    type_of = {n["node_id"]: n["node_type"] for n in nodes}
    kept, summaries = [], []
    for comm in communities:
        if len(comm) < min_community:
            continue
        members = list(comm)
        deg = Counter()
        rel_counter = Counter()
        for e in edges:
            if e["src"] in comm and e["dst"] in comm:
                rel_counter[e["edge_type"]] += 1
                deg[e["src"]] += 1
                deg[e["dst"]] += 1
        reps = sorted(members, key=lambda m: -deg[m])[:5]
        top_types = Counter(type_of[m] for m in members).most_common(3)
        summary = (f"社区 #{len(kept)+1}（{len(members)} 成员）"
                   f"代表: {', '.join(name_of[r] for r in reps)}；"
                   f"构成: {', '.join(f'{t}×{c}' for t, c in top_types)}；"
                   f"高频关系: {', '.join(f'{r}×{c}' for r, c in rel_counter.most_common(3))}")
        kept.append(members)
        summaries.append(summary)
    ids = s.save_communities(level=0, communities=kept, summaries=summaries)
    s.log_event("index", {"tool": "build_communities", "communities": len(kept)})
    return {"communities": len(kept), "summaries": summaries,
            "community_ids": ids,
            "note": "deterministic summaries; LLM-polished summaries on roadmap"}


@mcp.tool()
def search_global(query: str, top_communities: int = 3) -> dict:
    """模式二：社区摘要检索（Ch.5.3.2），适合全局型/主题型问题
    （这批文档主要论点是什么）。没有任何 k 跳子图能覆盖「全局」，
    这类问题只有社区摘要能答。
    先调用 build_communities 建索引。返回按相关度排序的社区摘要。"""
    s = store()
    comms = s.get_communities(level=0)
    if not comms:
        return {"error": "no community index; call build_communities first",
                "hint": "图更新后摘要需重算（Ch.5.3.2 摘要重算成本）"}
    scored = []
    for c in comms:
        member_text = " ".join(str(m) for m in c["member_ids"][:10])
        hits = s.semantic_search(query + " " + member_text, top_k=3,
                                 reinforce=False)
        score = sum(h["score"] for h in hits if h["node_id"] in set(c["member_ids"]))
        scored.append((score, c))
    scored.sort(key=lambda x: -x[0])
    chosen = [c for _, c in scored[:top_communities]]
    evidence = "\n\n".join(f"[社区摘要] {c['summary']}" for c in chosen)
    ids = [i for c in chosen for i in c["member_ids"]]
    s.log_event("retrieve", {"tool": "search_global", "query": query})
    return {"evidence": evidence,
            "citations": {"node_ids": sorted(set(ids)), "edge_ids": []},
            "cost": {"communities": len(chosen),
                     "est_tokens": int(len(evidence) * 0.6)},
            "retrieval_mode": "community-summaries"}


# ================================================================ mode 3
@mcp.tool()
def search_paths(query: str, source_entity: str = None,
                 target_entity: str = None, max_paths: int = 3,
                 max_len: int = 4) -> dict:
    """模式三：路径推理检索（Ch.5.3.3），适合多跳/因果问题（A 如何影响 B）。
    两种形态：
    - 给了 source_entity + target_entity：k 条最短路，每条序列化为
      带解释的推理链（路径即证据链，审计性最强）
    - 只给 query：个性化 PageRank 让相关节点按结构重要度自动浮现
      （HippoRAG 式「让图自己投票」）
    注意：路径不存在 ≠ 结论为假，只代表图上无证据——本工具会明确返回
    「图上无证据」而不是强行编造（Ch.5.5 案例二）。"""
    s = store()
    g, nodes, edges = _load_graph()
    name_of = {n["node_id"]: (n["name"] or n["content"][:30]) for n in nodes}
    if source_entity and target_entity:
        src = _resolve_entity(s, source_entity)
        dst = _resolve_entity(s, target_entity)
        if not src or not dst:
            return {"evidence": f"图上无证据：实体 '{source_entity if not src else target_entity}' 不在图中",
                    "citations": [], "cost": {"nodes": 0, "edges": 0, "est_tokens": 0},
                    "retrieval_mode": "path"}
        paths = alg.top_k_paths(g, src, dst, k=max_paths, max_len=max_len)
        if not paths:
            return {"evidence": "图上无证据：两实体间不存在满足长度约束的路径"
                                "（这不代表结论为假，只是图里没有这条证据）",
                    "citations": {"node_ids": [src, dst], "edge_ids": []},
                    "cost": {"nodes": 2, "edges": 0, "est_tokens": 0},
                    "retrieval_mode": "path"}
        lines = []
        edge_lookup = {(e["src"], e["dst"]): e for e in edges}
        edge_lookup.update({(e["dst"], e["src"]): e for e in edges})
        used_edges = []
        for i, p in enumerate(paths, 1):
            steps = []
            for a, b in zip(p, p[1:]):
                e = edge_lookup.get((a, b))
                if e:
                    used_edges.append(e)
                    steps.append(f"-[{e['edge_type']}]->")
                else:
                    steps.append("->")
            chain = " ".join(f"{name_of.get(n, n)}{step}"
                             for n, step in zip(p, steps + [""]))
            lines.append(f"[推理链 {i}] {chain}")
        evidence = "\n".join(lines)
        node_dicts = [n for n in nodes if n["node_id"] in set(sum(paths, []))]
        return {"evidence": evidence,
                "citations": {"node_ids": [n["node_id"] for n in node_dicts],
                              "edge_ids": [e["edge_id"] for e in used_edges]},
                "cost": cost_report(node_dicts, used_edges, evidence),
                "retrieval_mode": "path"}
    # ---- PPR mode ----
    seeds = _seed_nodes(query, top_k=3)
    if not seeds:
        return {"evidence": "(no seed nodes found)", "citations": [],
                "cost": {"nodes": 0, "edges": 0, "est_tokens": 0},
                "retrieval_mode": "ppr"}
    scores = alg.personalized_pagerank(g, seeds)
    top = sorted(scores.items(), key=lambda x: -x[1])[:15]
    node_dicts = [n for n in nodes if n["node_id"] in {i for i, _ in top}]
    sub_edges = [e for e in edges
                 if e["src"] in {i for i, _ in top} and e["dst"] in {i for i, _ in top}]
    lines = [serialize_evidence(node_dicts, sub_edges)]
    lines.append("[PPR排序] " + ", ".join(f"{name_of.get(i, i)}({sc:.3f})"
                                          for i, sc in top[:8]))
    evidence = "\n".join(lines)
    return {"evidence": evidence,
            "citations": {"node_ids": [n["node_id"] for n in node_dicts],
                          "edge_ids": [e["edge_id"] for e in sub_edges]},
            "cost": cost_report(node_dicts, sub_edges, evidence),
            "retrieval_mode": "ppr"}


def _resolve_entity(s: GraphStore, name: str) -> str | None:
    """Strict resolution over Entity/Event/Fact nodes: exact name match first,
    then substring. No fuzzy fallback - an unresolvable entity must surface as
    「图上无证据」instead of silently binding to the nearest node.
    Event/Fact nodes are legitimate path endpoints: a question like "发布事件
    与张三的关系" names an event, and refusing it would falsely report absence
    of evidence that exists. (Fuzzy resolution with real embedding backends,
    roadmap.)"""
    for node_type in ("Entity", "Event", "Fact"):
        rows = s.query_nodes(name_like=name, node_type=node_type)
        for r in rows:
            if r["name"] == name:
                return r["node_id"]
        if rows:
            return rows[0]["node_id"]
    return None


def run():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    run()
