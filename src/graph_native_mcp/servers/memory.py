"""graph-memory MCP server (book Ch.6) — the graph as long-term memory.

Successor of gx-memory: the tool interface (add_knowledge / semantic_search /
retrieve_context / ingest_file / ...) is carried over, the GX-engine + pickle
backend is replaced by the portable SQLite core, and the Ch.6 capabilities
gx-memory lacked are added:

  - write_memory with bi-temporal timestamps and conflict detection:
    contradictions become `contradicts` edges (kept, not overwritten);
    explicit corrections mark the old node `superseded_by` (kept, not deleted).
  - get_timeline: "what was true / what did the system know at time T".
  - consolidate: exponential time decay tiered by memory type; retrieval hits
    reinforce weights (使用投票).
"""

from __future__ import annotations

import os
import re

from mcp.server.fastmcp import FastMCP

from ..core.store import GraphStore, default_db_path

mcp = FastMCP("graph-memory")
_store: GraphStore | None = None


def store() -> GraphStore:
    global _store
    if _store is None:
        _store = GraphStore(default_db_path())
    return _store


# ================================================================ writes
@mcp.tool()
def add_knowledge(text: str, node_id: str = None, category: str = "knowledge",
                  link_to: list = None, relation: str = "related",
                  source_ids: list = None, attributes: dict = None) -> dict:
    """写入一条知识到图谱（自动计算语义向量）。
    - text: 知识正文（必填）
    - node_id: 可选自定义节点 ID
    - category: 类别标签
    - link_to: 需要建立关系的目标节点 ID 列表
    - relation: 关系标签（见 EDGE_TYPES）
    - source_ids: 来源引用列表（强烈建议提供，无来源的 Fact 会被写入门控拒绝）
    - attributes: 附加结构化属性
    返回新建节点的 ID。"""
    s = store()
    props = dict(attributes or {})
    props["category"] = category
    nid = s.add_node(name=text[:60], content=text, node_type="Knowledge",
                     properties=props, source_ids=source_ids or [f"inline:{text[:40]}"],
                     node_id=node_id)
    for target in (link_to or []):
        if target in s.all_node_ids():
            s.add_edge(nid, target, edge_type=relation if relation in
                       ("related", "supports", "involves", "causal", "temporal",
                        "derives_from", "contradicts") else "related",
                       source_ids=source_ids)
    s.log_event("write", {"tool": "add_knowledge", "node_id": nid})
    return {"node_id": nid, "status": "created"}


@mcp.tool()
def write_memory(content: str, memory_type: str = "Fact",
                 entities: list = None, valid_time: str = None,
                 source_ids: list = None, confidence: float = 1.0,
                 explicit_correction: bool = False) -> dict:
    """第6章核心写入：带双时间戳与冲突检测的记忆写入。
    - content: 记忆正文
    - memory_type: Fact / Event / Preference / Task
    - entities: 涉及的实体名列表（自动建实体节点并连 involves 边）
    - valid_time: 事实发生时间（ISO 格式；不填=与记录时间相同）。
      「事情发生的时间」与「系统知道的时间」分开记，乱序到达/事后更正/
      历史回溯三类查询都靠它。
    - source_ids: 来源引用（Fact 必填，无来源不入图）
    - confidence: 置信度 0~1
    - explicit_correction: 仅当用户明确说「之前记错了」时设为 True，
      此时旧记忆被标记 superseded（保留可回溯，不删除）
    返回冲突报告：与旧记忆矛盾时建 contradicts 边并保留双方。"""
    s = store()
    ent_ids = [s.upsert_entity(name) for name in (entities or [])]
    nid = s.add_node(name=content[:60], content=content, node_type=memory_type,
                     source_ids=source_ids, confidence=confidence,
                     valid_from=valid_time)
    for eid in ent_ids:
        s.add_edge(nid, eid, edge_type="involves", source_ids=source_ids)
    report = {"node_id": nid, "conflicts": [], "merged_into": None,
              "superseded": [], "status": "created"}
    # ---- conflict detection (Ch.6.3.2) ----
    for cand in s.find_conflict_candidates(content, set(ent_ids)):
        if cand["node_id"] == nid:
            continue
        verdict = _conflict_verdict(content, cand, memory_type)
        if verdict == "duplicate":
            merged_src = list(dict.fromkeys((source_ids or []) + cand["source_ids"]))
            s.conn.execute("UPDATE nodes SET source_ids=?, confidence=MIN(1.0, ?+0.1)"
                           " WHERE node_id=?",
                           (_json_dumps(merged_src),
                            min(1.0, cand["confidence"] + 0.1), cand["node_id"]))
            s.conn.commit()
            s.delete_node(nid)
            report["merged_into"] = cand["node_id"]
            report["status"] = "merged-duplicate"
            break
        if verdict is None:
            continue
        # verdict in ("numeric", "negation", "potential")
        if verdict != "potential" and explicit_correction:
            s.mark_superseded(cand["node_id"], nid)
            s.add_edge(nid, cand["node_id"], edge_type="supersedes",
                       source_ids=source_ids)
            report["superseded"].append(cand["node_id"])
            report["status"] = "corrected"
        else:
            # 保留式更新：矛盾不覆盖，建 contradicts 边留待检索时合并策略
            # 或人工裁决。potential = 确定性规则判不了（如 深色/浅色），
            # 只标注不裁定。
            s.add_edge(nid, cand["node_id"], edge_type="contradicts",
                       source_ids=source_ids,
                       properties={"verdict": verdict,
                                   "note": "conflict preserved, not overwritten"})
            report["conflicts"].append({"node_id": cand["node_id"],
                                        "verdict": verdict,
                                        "content": cand["content"][:100]})
            report["status"] = "conflict-preserved"
    s.log_event("write", {"tool": "write_memory", "node_id": nid,
                          "status": report["status"]})
    return report


def _json_dumps(x) -> str:
    import json
    return json.dumps(x, ensure_ascii=False)


def _strip_digits(text: str) -> str:
    import re as _re
    return _re.sub(r"\d+(?:\.\d+)?", "#", text)


def _norm(text: str) -> str:
    return re.sub(r"[\s，。、的了吗呢吧]", "", text)


def _frame_sim(new: str, old: str) -> float:
    """Similarity of the subject frame, with digits masked so that
    '持有股权5%' vs '持有股权10%' counts as the same frame."""
    return _sim(_strip_digits(new), _strip_digits(old))


def _conflict_verdict(new: str, old: dict, new_type: str = "") -> str | None:
    """Deterministic conflict judgment (rules only; LLM-assisted judgment
    belongs to graph-build, roadmap). Returns:
      "numeric"   same frame, different numbers -> hard conflict
      "negation"  same frame, different polarity -> hard conflict
      "potential" same frame, same type, cannot be decided by rules
                  (e.g. 深色主题 vs 浅色主题) -> preserved + flagged
      "duplicate" near-identical content
      None        no conflict signal
    Facts with different valid_from are temporal succession, not conflict."""
    old_content = old.get("content", "")
    old_type = old.get("node_type", "")
    # dated facts: a new fact of the same frame is presumed temporal
    # succession (bi-temporal handles it), not a contradiction. True
    # conflicts between dated facts need valid_time comparison or LLM
    # judgment (roadmap, graph-build).
    if old_type == "Fact" and new_type == "Fact" and (old.get("valid_from") or "").strip():
        return None
    frame = _frame_sim(new, old_content)
    if frame >= 0.92 and _sim(new, old_content) >= 0.92:
        return "duplicate"
    if frame < 0.40:
        return None
    n_new = re.findall(r"\d+(?:\.\d+)?", new)
    n_old = re.findall(r"\d+(?:\.\d+)?", old_content)
    if n_new and n_old and n_new[0] != n_old[0]:
        return "numeric"
    if ("不" in new) != ("不" in old_content):
        return "negation"
    if old_type == "Preference" or (old_type == new_type == "Fact"):
        return "potential"
    return None


def _sim(a: str, b: str) -> float:
    from ..core.embeddings import cosine, get_embedding_provider
    emb = get_embedding_provider()
    return cosine(emb.embed([a])[0], emb.embed([b])[0])


@mcp.tool()
def add_node(id: str, name: str, class_: str = "Knowledge", data: str = "",
             attributes: dict = None) -> dict:
    """添加结构化节点（不经过冲突检测，用于手工建图）。
    - id / name / class_: 节点 ID、名称、类型
    - data: 附加文本
    - attributes: 结构化属性"""
    nid = store().add_node(name=name, content=data, node_type=class_ or "Knowledge",
                           properties=attributes, node_id=id, gate=False)
    return {"node_id": nid, "status": "created"}


@mcp.tool()
def add_edge(source_id: str, target_id: str, label: str = "related",
             properties: dict = None) -> dict:
    """在两个已存在节点间建立有向边（关系）。
    - label: 关系类型，见 EDGE_TYPES（temporal/causal/involves/supports/
      contradicts/derives_from/enables/supersedes/related）"""
    eid = store().add_edge(source_id, target_id, edge_type=label,
                           properties=properties)
    return {"edge_id": eid, "status": "created"}


@mcp.tool()
def delete_node(node_id: str) -> dict:
    """删除节点及其相连边。"""
    return {"node_id": node_id, "deleted": store().delete_node(node_id)}


@mcp.tool()
def delete_edge(edge_id: int) -> dict:
    """按边 ID 删除一条边。"""
    return {"deleted": store().delete_edge(edge_id)}


@mcp.tool()
def ingest_file(path: str, category: str = "document",
                chunk_chars: int = 800) -> dict:
    """导入纯文本/Markdown/JSON 文件：建 hub 节点 + 按段切块的子节点，
    块间按原文顺序连 temporal 边（保留逻辑顺序）。
    注意：这是「切块记忆」不是「抽取图谱」——实体关系抽取请用
    graph-build server 的 ingest_document（四层管线）。"""
    s = store()
    if not os.path.isfile(path):
        return {"error": f"file not found: {path}"}
    text = open(path, encoding="utf-8", errors="ignore").read()
    seed_id = os.path.splitext(os.path.basename(path))[0]
    hub = s.add_node(name=seed_id, content=f"[hub] {seed_id}",
                     node_type="Knowledge",
                     properties={"category": category, "kind": "hub"},
                     source_ids=[f"file:{path}"])
    chunks = [c for c in re.split(r"\n\s*\n", text) if c.strip()]
    merged, buf = [], ""
    for c in chunks:
        if len(buf) + len(c) < chunk_chars:
            buf += "\n\n" + c
        else:
            merged.append(buf)
            buf = c
    if buf:
        merged.append(buf)
    prev = hub
    for i, c in enumerate(merged):
        cid = s.add_node(name=f"{seed_id}#{i}", content=c.strip(),
                         node_type="Knowledge",
                         properties={"category": category, "chunk": i},
                         source_ids=[f"file:{path}#chunk{i}"])
        s.add_edge(prev, cid, edge_type="temporal", source_ids=[f"file:{path}"])
        prev = cid
    s.log_event("write", {"tool": "ingest_file", "path": path, "hub": hub})
    return {"hub_id": hub, "chunks": len(merged), "path": path}


# ================================================================ retrieval
@mcp.tool()
def semantic_search(query: str, top_k: int = 5, threshold: float = 0.0,
                    node_type: str = None) -> list:
    """语义向量检索：与 query 最相似的节点。
    命中会自动强化记忆权重（使用投票，Ch.6.3.1）。
    返回 [{node_id, node_type, name, content, score}]。"""
    return store().semantic_search(query, top_k=top_k, threshold=threshold,
                                   node_type=node_type or "")


@mcp.tool()
def retrieve_context(query: str, max_nodes: int = 5, hops: int = 1) -> str:
    """GraphRAG 上下文检索：语义定位入口节点，沿图多跳取邻居，
    序列化为可直接喂给 LLM 的带引用 id 的证据文本。"""
    s = store()
    hits = s.semantic_search(query, top_k=max_nodes)
    if not hits:
        return "(no relevant memories found)"
    node_map, edges = {}, []
    for h in hits:
        node_map[h["node_id"]] = h
        for layer in range(1, hops + 1):
            nb = s.get_neighbors(h["node_id"], hops=1)
            for e in nb.get("hop1", {}).get("edges", []):
                edges.append(e)
                for end in (e["src"], e["dst"]):
                    n = s.get_node(end)
                    if n and end not in node_map:
                        node_map[end] = {k: n[k] for k in
                                         ("node_id", "node_type", "name", "content",
                                          "properties", "source_ids")}
    from ..core.serialize import serialize_evidence
    nodes = list(node_map.values())
    return serialize_evidence(nodes, edges)


@mcp.tool()
def get_neighbors(node_id: str, hops: int = 1, direction: str = "both") -> dict:
    """多跳邻居遍历，按跳数分层。direction: out/in/both。"""
    return store().get_neighbors(node_id, hops=hops, direction=direction)


@mcp.tool()
def get_timeline(entity: str, as_of: str = None) -> dict:
    """双时间戳时间线查询（Ch.6.2.2）。
    - entity: 实体名（模糊匹配）
    - as_of: ISO 时间。默认同时过滤「事实时间」和「系统知道的时间」，
      回答「当时系统知道什么」；加前缀 'valid:' 则只按事实时间过滤，
      回答「当时世界上发生了什么」。
    返回该实体相关的事件/事实按时间排序的列表。"""
    items = store().get_timeline(entity, as_of=as_of)
    return {"entity": entity, "as_of": as_of, "items": items,
            "count": len(items)}


@mcp.tool()
def query_nodes(name_like: str = "", class_: str = "",
                attribute_filters: dict = None) -> list:
    """按名称子串 / 类型 / 属性精确匹配过滤节点。"""
    return [{"node_id": n["node_id"], "node_type": n["node_type"], "name": n["name"],
             "content": n["content"][:100], "properties": n["properties"]}
            for n in store().query_nodes(name_like, class_, attribute_filters)]


# ================================================================ maintenance
@mcp.tool()
def consolidate() -> dict:
    """记忆整理（Ch.6.3.1）：指数时间衰减（按记忆类型分档 λ），
    冷记忆只报告不删除（保留式更新）。检索命中已在 semantic_search
    中自动强化权重。返回衰减报告与冷/热记忆清单。"""
    return store().consolidate()


@mcp.tool()
def graph_stats() -> dict:
    """图谱统计：节点/边/事件数、类型分布、矛盾边数、存储路径、
    嵌入后端。"""
    return store().stats()


@mcp.tool()
def clear_graph() -> dict:
    """清空当前图谱（不可恢复，请确认后调用）。"""
    s = store()
    with s.lock:
        s.conn.executescript("DELETE FROM nodes; DELETE FROM edges;"
                             " DELETE FROM communities; DELETE FROM events;")
        s.conn.commit()
    return s.stats()


def run():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    run()
