"""graph-build MCP server (book Ch.7) — agent-first, zero API keys by default.

Three pipelines, one per tool:

  ingest_document  Ch.7.2 four-layer extraction: candidates -> disambiguation
                   -> relation judgment -> write gate. Every error exits at a
                   named layer, so failures are attributable (召回不足查一层,
                   链接错配查二层, 关系误判查三层, 脏写入查四层).
  text_to_query    Ch.7.3 constrained generation + execution feedback loop.
  graph_checkup    Ch.7.4 five error classes + provenance audit.

Design decision — the calling agent IS the LLM. An MCP tool that demands its
own API key forces every user to buy a second model, which is both wasteful
and wrong: the host agent can do the generative steps (candidate extraction,
NL->DSL) itself, and the server does what it is uniquely good at — the
deterministic, verifiable part (linking, gating, validation, execution).

Default zero-key workflow (agent-in-the-loop):
  1. Call ingest_document without extraction_json -> get the extraction prompt.
  2. The agent (you) extracts entities/relations per the prompt.
  3. Call ingest_document again with extraction_json -> L2-L4 run locally.
Same for text_to_query: the agent writes the DSL plan, the server validates
and executes it, and execution errors feed back into the next agent turn —
which is exactly the Ch.7.3 feedback loop, with the host LLM driving it.

Optional offline/batch mode: set GNM_LLM_BASE_URL / GNM_LLM_API_KEY /
GNM_LLM_MODEL (any OpenAI-compatible endpoint) and the server will generate
candidates internally — useful for bulk imports where no agent is present.
"""

from __future__ import annotations

import json
import statistics

from graph_native_mcp._fastmcp_compat import FastMCP

from ..core import llm
from ..core.embeddings import cosine, get_embedding_provider
from ..core.schema import NODE_TYPES, EDGE_TYPES
from ..core.store import GraphStore, default_db_path

mcp = FastMCP("graph-build")
_store: GraphStore | None = None


def store() -> GraphStore:
    global _store
    if _store is None:
        _store = GraphStore(default_db_path())
    return _store


_EXTRACTION_SYSTEM = (
    "You are an information-extraction layer. Extract entities and relations "
    "from the text. Reply with a single JSON object: "
    '{{"entities": [{{"name": str, "type": "Entity"|"Fact"|"Event", "aliases": [str], '
    '"description": str}}], "relations": [{{"subject": str, "object": str, '
    '"predicate": str, "evidence": str, "confidence": float}}]}}. '
    "Allowed node types: {node_types}. "
    "Relation predicates will be mapped to: {edge_types}."
)

_QUERY_SYSTEM = (
    "You translate natural-language questions into a JSON graph query. "
    "Reply with a single JSON object with optional keys: "
    '{{"name_like": str (substring of node name), "node_type": str '
    "(one of {node_types}), 'hops': int (1-3), "
    '"edge_types": list (subset of {edge_types}). '
    "Choose the smallest hops that answers the question."
)


def _extraction_system_prompt() -> str:
    return _EXTRACTION_SYSTEM.format(node_types=sorted(NODE_TYPES),
                                     edge_types=sorted(EDGE_TYPES))


def _query_system_prompt() -> str:
    return _QUERY_SYSTEM.format(node_types=sorted(NODE_TYPES),
                                edge_types=sorted(EDGE_TYPES))


def _parse_extraction(extraction_json: str | dict) -> tuple[list, list, str | None]:
    """Parse agent-provided extraction. Accepts a JSON string or dict.
    Returns (entities, relations, error)."""
    if isinstance(extraction_json, dict):
        data = extraction_json
    else:
        try:
            data = json.loads(extraction_json)
        except (json.JSONDecodeError, TypeError) as exc:
            return [], [], f"extraction_json is not valid JSON: {exc}"
    if not isinstance(data, dict):
        return [], [], "extraction_json must be a JSON object"
    return data.get("entities", []) or [], data.get("relations", []) or [], None


# ================================================================ Ch.7.2
@mcp.tool()
def get_extraction_prompt(text: str, source_id: str, schema_hint: str = "") -> dict:
    """返回 L1 抽取提示词（零 API key 工作流第一步）。
    把返回的 prompt 交给当前对话中的 LLM（你自己）完成实体/关系抽取，
    再把抽取结果作为 extraction_json 传给 ingest_document 写入图谱。
    服务端只做确定性部分：消歧链接、schema 判定、写入门控。"""
    return {
        "system": _extraction_system_prompt(),
        "user": f"Schema hint: {schema_hint}\n\nSource: {source_id}\n\n{text[:8000]}",
        "next_step": ("让当前 LLM 按此 prompt 产出单个 JSON 对象，"
                      "然后调用 ingest_document(text=..., source_id=..., "
                      "extraction_json=<该 JSON>) 完成写入。"),
    }


@mcp.tool()
def ingest_document(text: str = "", source_id: str = "",
                    extraction_json: str = "",
                    confidence_threshold: float = 0.5,
                    schema_hint: str = "") -> dict:
    """四层管线从文本到图谱（Ch.7.2.1），默认零 API key：
      L1 候选生成 —— 两种方式：a) 不传 extraction_json 且未配 GNM_LLM_*，
         返回抽取 prompt 交给你（调用方 LLM）完成，再带 extraction_json 重调；
         b) 配置了 GNM_LLM_* 时服务端内部生成（批量导入场景）。
      L2 消歧与链接（别名表 + 嵌入相似，映射到既有实体或新建）—— 本地执行
      L3 关系判定（schema 约束，只允许已知边类型）—— 本地执行
      L4 写入门控（置信度阈值 + schema 校验 + 来源强制）—— 本地执行
    每层统计与错误按层归因返回 —— 不做端到端黑盒抽取，出错能定位。
    - text: 文档正文（返回 prompt 模式下用于拼 prompt；带 extraction_json 重调时可省略）
    - source_id: 来源标识（必填，引用闭包的锚点）
    - extraction_json: L1 抽取结果（JSON 字符串或对象，含 entities/relations）
    - confidence_threshold: L4 的置信度闸门（默认 0.5）
    - schema_hint: 领域提示（如「金融公告，关注持股/担保/人事变动」）"""
    s = store()
    report = {"layers": {}, "errors": [], "written_nodes": 0, "written_edges": 0}

    # ---------- L1: candidate generation ----------
    if extraction_json:
        ents, rels, err = _parse_extraction(extraction_json)
        if err:
            report["errors"].append({"layer": 1, "error": err,
                                     "hint": "可用 get_extraction_prompt 获取抽取规范后重试"})
            report["layers"]["L1_candidates"] = 0
            return report
        report["layer1_source"] = "agent"
    elif llm.llm_configured() and text:
        prompt = [
            {"role": "system", "content": _extraction_system_prompt()},
            {"role": "user", "content": (f"Schema hint: {schema_hint}\n\n"
                                         f"Source: {source_id}\n\n{text[:8000]}")},
        ]
        try:
            extraction = llm.chat_json(prompt)
        except Exception as exc:
            report["errors"].append({"layer": 1, "error": f"candidate generation failed: {exc}"})
            report["layers"]["L1_candidates"] = 0
            return report
        ents = extraction.get("entities", [])
        rels = extraction.get("relations", [])
        report["layer1_source"] = "server_llm"
    else:
        # 未配 key 也未带抽取结果：把 prompt 返还给调用方 LLM，零 key 工作流
        return {
            "needs_agent_extraction": True,
            "message": ("本服务不需要 API key：当前对话里的 LLM 就能做抽取。"
                        "按下面 prompt 产出单个 JSON 对象，再以 extraction_json 参数重调本工具。"),
            "prompt": _agent_extraction_prompt(text, source_id, schema_hint),
            "next_call": "ingest_document(text, source_id, extraction_json=<LLM产出的JSON>)",
        }
    report["layers"]["L1_candidates"] = {"entities": len(ents), "relations": len(rels)}

    _run_l2_l4(s, report, ents, rels, source_id, confidence_threshold)
    return report


def _agent_extraction_prompt(text: str, source_id: str, schema_hint: str) -> dict:
    return {
        "system": _extraction_system_prompt(),
        "user": f"Schema hint: {schema_hint}\n\nSource: {source_id}\n\n{text[:8000]}",
    }


def _run_l2_l4(s: GraphStore, report: dict, ents: list, rels: list,
               source_id: str, confidence_threshold: float) -> None:
    """L2 消歧链接 + L3 关系判定 + L4 写入门控，全部确定性本地执行。"""
    # ---------- L2: disambiguation & linking ----------
    emb = get_embedding_provider()
    id_of = {}
    new_nodes = 0
    for ent in ents:
        name = str(ent.get("name", "")).strip()
        if not name:
            continue
        existing = s.query_nodes(name_like=name, node_type="Entity")
        match = None
        if existing:
            match = existing[0]["node_id"]
        else:
            q = emb.embed([f"{name} {ent.get('description', '')}"])[0]
            best, best_s = None, 0.90
            for cand in s.query_nodes(node_type="Entity"):
                if not cand.get("embedding"):
                    cand_full = s.get_node(cand["node_id"])
                    ce = cand_full["embedding"] if cand_full else None
                else:
                    ce = cand.get("embedding")
                if ce:
                    sc = cosine(q, ce)
                    if sc > best_s:
                        best, best_s = cand["node_id"], sc
            match = best
        if match is None:
            ntype = ent.get("type") if ent.get("type") in NODE_TYPES else "Entity"
            match = s.add_node(name=name, content=ent.get("description", ""),
                               node_type=ntype,
                               properties={"aliases": ent.get("aliases", [])},
                               source_ids=[source_id], embed=True, gate=False)
            new_nodes += 1
            report.setdefault("created_entities", []).append(name)
        id_of[name] = match
    report["layers"]["L2_linking"] = {"linked": len(id_of) - new_nodes, "new": new_nodes}

    # ---------- L3 + L4: relation judgment + write gating ----------
    accepted, rejected_type, gated = 0, 0, 0
    for rel in rels:
        subj, obj = id_of.get(str(rel.get("subject", ""))), id_of.get(str(rel.get("object", "")))
        if not subj or not obj:
            report["errors"].append({"layer": 3, "item": rel,
                                     "reason": "endpoint entity not found after linking"})
            continue
        pred = str(rel.get("predicate", "related")).lower()
        etype = pred if pred in EDGE_TYPES else "related"
        if pred not in EDGE_TYPES:
            rejected_type += 1
        conf = float(rel.get("confidence", 0.8))
        # L4 valves: confidence + provenance
        if conf < confidence_threshold:
            gated += 1
            report["errors"].append({"layer": 4, "item": rel,
                                     "reason": f"confidence {conf} < threshold {confidence_threshold}"})
            continue
        s.add_edge(subj, obj, edge_type=etype,
                   properties={"evidence": rel.get("evidence", "")[:300]},
                   source_ids=[source_id], confidence=conf)
        accepted += 1
    report["layers"]["L3_relations"] = {"accepted": accepted,
                                        "untyped_downgraded": rejected_type}
    report["layers"]["L4_gate"] = {"gated": gated,
                                   "threshold": confidence_threshold}
    report["written_nodes"] = new_nodes
    report["written_edges"] = accepted
    s.log_event("build", {"tool": "ingest_document", "source": source_id,
                          "nodes": new_nodes, "edges": accepted})


# ================================================================ Ch.7.3
@mcp.tool()
def text_to_query(question: str = "", query_plan: str = "",
                  max_retries: int = 2) -> dict:
    """自然语言 → 结构化图查询（Ch.7.3 约束生成 + 执行反馈闭环），默认零 API key：
    - 带 query_plan（你自己产出的 JSON DSL）→ 服务端校验并执行，返回带引用 id
      的证据；执行报错会原样返回，你修正 DSL 后重调即为反馈闭环。
    - 不带 query_plan 且配置了 GNM_LLM_* → 服务端内部生成+重试（批量场景）。
    - 都没有 → 返回 DSL 格式说明，由你生成后重调。
    DSL 格式：
      {"name_like": str, "node_type": str, "hops": int, "edge_types": [str]}
    （Cypher 后端在路线图：Kùzu）"""
    s = store()

    if query_plan:
        try:
            plan = json.loads(query_plan) if isinstance(query_plan, str) else query_plan
        except json.JSONDecodeError as exc:
            return {"error": f"query_plan is not valid JSON: {exc}",
                    "dsl_format": {"name_like": "str", "node_type": "one of " + str(sorted(NODE_TYPES)),
                                   "hops": "1-3", "edge_types": "subset of " + str(sorted(EDGE_TYPES))}}
        result = _execute_dsl(s, plan)
        result["query_plan"] = plan
        if "error" in result:
            # 反馈闭环：错误回喂给调用方 LLM，由其修正后重调
            result["retry_hint"] = ("修正 query_plan 后重调本工具"
                                    "（检查 node_type 拼写、放宽 name_like、减小 hops）。")
        else:
            s.log_event("query", {"tool": "text_to_query", "question": question,
                                  "plan": plan, "mode": "agent"})
        return result

    if llm.llm_configured():
        messages = [{"role": "system", "content": _query_system_prompt()},
                    {"role": "user", "content": question}]
        feedback_log = []
        for attempt in range(max_retries + 1):
            try:
                plan = llm.chat_json(messages)
            except Exception as exc:
                return {"error": f"generation failed: {exc}", "feedback_log": feedback_log}
            result = _execute_dsl(s, plan)
            if "error" not in result:
                result["query_plan"] = plan
                result["feedback_log"] = feedback_log
                s.log_event("query", {"tool": "text_to_query", "question": question,
                                      "plan": plan, "mode": "server_llm"})
                return result
            feedback_log.append({"attempt": attempt + 1, "plan": plan,
                                 "error": result["error"]})
            messages.append({"role": "assistant", "content": json.dumps(plan, ensure_ascii=False)})
            messages.append({"role": "user", "content":
                             f"Execution failed: {result['error']}. "
                             "Adjust the query (check node_type spelling, relax name_like) and reply again."})
        return {"error": "all attempts failed", "feedback_log": feedback_log}

    # 零 key 工作流：把 DSL 规范交给调用方 LLM
    return {
        "needs_agent_plan": True,
        "message": ("本服务不需要 API key：当前对话里的 LLM 直接产出查询计划即可。"
                    "按 dsl_format 生成单个 JSON 对象，以 query_plan 参数重调本工具。"),
        "dsl_format": {"name_like": "节点名子串（必填其一即可命中）",
                       "node_type": f"可选，{sorted(NODE_TYPES)} 之一",
                       "hops": "可选，1-3，默认 1",
                       "edge_types": f"可选，{sorted(EDGE_TYPES)} 的子集"},
        "next_call": 'text_to_query(question, query_plan=\'{"name_like": "...", "hops": 1}\')',
    }


def _execute_dsl(s: GraphStore, plan: dict) -> dict:
    from ..core.serialize import serialize_evidence
    name_like = str(plan.get("name_like", ""))
    node_type = plan.get("node_type", "")
    hops = int(plan.get("hops", 1))
    if node_type and node_type not in NODE_TYPES:
        return {"error": f"unknown node_type '{node_type}'"}
    rows = s.query_nodes(name_like=name_like, node_type=node_type or "")
    if not rows:
        return {"error": f"no nodes match name_like='{name_like}'"
                         + (f" node_type='{node_type}'" if node_type else "")}
    if hops > 3:
        hops = 3
    node_map, edges = {r["node_id"]: {k: r[k] for k in
                                      ("node_id", "node_type", "name", "content",
                                       "properties", "source_ids")} for r in rows}, []
    for r in rows:
        nb = s.get_neighbors(r["node_id"], hops=hops)
        for h in range(1, hops + 1):
            for e in nb.get(f"hop{h}", {}).get("edges", []):
                edges.append(e)
                for end in (e["src"], e["dst"]):
                    if end not in node_map:
                        n = s.get_node(end)
                        if n:
                            node_map[end] = {k: n[k] for k in
                                             ("node_id", "node_type", "name", "content",
                                              "properties", "source_ids")}
    nodes = list(node_map.values())
    return {"evidence": serialize_evidence(nodes, edges),
            "matched": len(rows), "result_nodes": len(nodes), "result_edges": len(edges)}


# ================================================================ Ch.7.4
@mcp.tool()
def graph_checkup() -> dict:
    """图体检（Ch.7.4 五类图错误 + 溯源审计）：
      1. 孤立节点率（无任何边的节点）
      2. 度数异常（hub 节点：度 > 均值+3σ，扩展爆炸的风险点）
      3. 断链引用（边端点缺失——外键约束下应为 0，防御性检查）
      4. schema 违规（未知节点/边类型）
      5. 低置信边率 + 无来源节点率（无来源的 Fact 禁止进入检索上下文）
    返回体检报告与修复建议。纯本地计算，不需要 LLM。"""
    s = store()
    nodes = [s._row_to_node(r) for r in s.conn.execute("SELECT * FROM nodes")]
    edges = [s._row_to_edge(r) for r in s.conn.execute("SELECT * FROM edges")]
    ids = {n["node_id"] for n in nodes}
    degree = {i: 0 for i in ids}
    for e in edges:
        degree[e["src"]] = degree.get(e["src"], 0) + 1
        degree[e["dst"]] = degree.get(e["dst"], 0) + 1
    isolated = [i for i, d in degree.items() if d == 0]
    degs = [d for d in degree.values()] or [0]
    mean, sigma = statistics.mean(degs), statistics.pstdev(degs) or 1.0
    hubs = sorted([(i, d) for i, d in degree.items() if d > mean + 3 * sigma],
                  key=lambda x: -x[1])[:10]
    dangling = [e["edge_id"] for e in edges
                if e["src"] not in ids or e["dst"] not in ids]
    bad_ntypes = sorted({n["node_type"] for n in nodes} - set(NODE_TYPES))
    bad_etypes = sorted({e["edge_type"] for e in edges} - set(EDGE_TYPES))
    low_conf = [e["edge_id"] for e in edges if e["confidence"] < 0.5]
    unattributed = [n["node_id"] for n in nodes
                    if n["node_type"] == "Fact" and not n["source_ids"]]
    n = max(len(nodes), 1)
    report = {
        "summary": {"nodes": len(nodes), "edges": len(edges),
                    "isolated_rate": round(len(isolated) / n, 4),
                    "low_conf_edge_rate": round(len(low_conf) / max(len(edges), 1), 4),
                    "unattributed_fact_count": len(unattributed)},
        "hubs": [{"node_id": i, "degree": d} for i, d in hubs],
        "dangling_edge_refs": dangling,
        "schema_violations": {"node_types": bad_ntypes, "edge_types": bad_etypes},
        "unattributed_facts": unattributed[:20],
        "suggestions": [],
    }
    if isolated:
        report["suggestions"].append(
            f"{len(isolated)} 个孤立节点：确认是抽取漏连（L3 问题）还是实体消歧分裂（L2 问题）")
    if hubs:
        report["suggestions"].append(
            f"hub 节点 {hubs[0][0]}（度 {hubs[0][1]}）：局部检索时对该种子收紧 max_nodes 预算")
    if bad_ntypes or bad_etypes:
        report["suggestions"].append("存在 schema 外类型：检查写入方是否绕过了门控")
    if unattributed:
        report["suggestions"].append(
            f"{len(unattributed)} 个无来源 Fact 节点：它们已被检索层屏蔽，建议补 source_ids 或删除")
    s.log_event("checkup", {"tool": "graph_checkup", **report["summary"]})
    return report


def run():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    run()
