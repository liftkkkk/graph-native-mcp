"""Offline test suite for gxt-core + the four servers' logic.

Runs without pytest:  python tests/run_tests.py
Covers: store CRUD, write gate, bi-temporal timeline, conflict detection,
decay, serialization format, retrieval modes, planning, extract_skills,
constraints, replay.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from graph_native_mcp.core.store import GraphStore
from graph_native_mcp.core.serialize import serialize_evidence, estimate_tokens
from graph_native_mcp.core import algorithms as alg
from graph_native_mcp.core.schema import validate_node

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {detail}")


def test_store_and_gate():
    print("[store + write gate]")
    s = GraphStore(_tmpdb())
    ok, _ = validate_node("Fact", ["src1"], 0.9)
    check("fact with source passes gate", ok)
    ok, reason = validate_node("Fact", [], 0.9)
    check("fact without source rejected", not ok, reason)
    try:
        s.add_node(content="no source", node_type="Fact")
        check("store enforces fact gate", False)
    except ValueError:
        check("store enforces fact gate", True)
    nid = s.add_node(content="苹果公司发布新品", node_type="Fact",
                     source_ids=["news#1"])
    check("node created", bool(nid))
    check("stats counts", s.stats()["nodes"] == 1)
    s.close()


def test_bi_temporal_and_conflicts():
    print("[bi-temporal + conflict detection]")
    s = GraphStore(_tmpdb())
    e1 = s.upsert_entity("张三")
    f1 = s.add_node(name="张三职务", content="张三于2024年出任CEO", node_type="Fact",
                    source_ids=["doc#1"], valid_from="2024-01-01")
    s.add_edge(f1, e1, edge_type="involves", source_ids=["doc#1"])
    f2 = s.add_node(name="张三职务2", content="张三于2025年出任CTO", node_type="Fact",
                    source_ids=["doc#2"], valid_from="2025-01-01")
    s.add_edge(f2, e1, edge_type="involves", source_ids=["doc#2"])
    tl = s.get_timeline("张三")
    check("timeline has facts+edges", len(tl) >= 3, str(len(tl)))
    node_items = [x for x in tl if "node_id" in x and "edge_id" not in x]
    check("timeline includes fact nodes", len(node_items) == 2, str(len(node_items)))
    # bi-temporal: fact-time query sees the 2024 fact only
    tl_valid = s.get_timeline("张三", as_of="valid:2024-06-01")
    valid_nodes = [x for x in tl_valid if "node_id" in x and "edge_id" not in x]
    check("valid: as_of filters by fact time", len(valid_nodes) == 1, str(len(valid_nodes)))
    # system-time query: facts were recorded NOW, so system knew nothing in 2024
    tl_sys = s.get_timeline("张三", as_of="2024-06-01")
    sys_nodes = [x for x in tl_sys if "node_id" in x and "edge_id" not in x]
    check("plain as_of filters by system time", len(sys_nodes) == 0, str(len(sys_nodes)))
    # conflict candidate detection: new fact sharing entity, high similarity
    cands = s.find_conflict_candidates("张三于2024年出任CEO", {e1})
    check("conflict candidate found", any(c["node_id"] == f1 for c in cands))
    # conflict verdicts: numbers / negation / potential / duplicate
    from graph_native_mcp.servers.memory import _conflict_verdict
    check("number conflict detected",
          _conflict_verdict("张三持有股权5%", {"content": "张三持有股权10%"}) == "numeric")
    check("preference conflict flagged potential",
          _conflict_verdict("用户偏好深色主题",
                            {"content": "用户偏好浅色主题", "node_type": "Preference"},
                            "Preference") == "potential")
    check("temporal succession is not conflict",
          _conflict_verdict("张三于2025年出任CTO",
                            {"content": "张三于2024年出任CEO", "node_type": "Fact",
                             "valid_from": "2024-01-01"}, "Fact") is None)
    s.close()


def test_decay_and_reinforce():
    print("[decay / consolidate]")
    s = GraphStore(_tmpdb())
    s.add_node(content="一条事件记忆", node_type="Event", source_ids=["e#1"])
    rep = s.consolidate()
    check("consolidate reports decayed count", rep["decayed"] >= 1)
    s.close()


def test_serialization():
    print("[evidence serialization]")
    text = serialize_evidence(
        [{"node_id": "n123", "node_type": "Company", "name": "Acme Inc.",
          "content": "做硬件的公司", "properties": {"hq": "SZ"},
          "source_ids": ["doc#42"]}],
        [{"edge_id": 7, "src": "n123", "dst": "n456", "edge_type": "EXECUTIVE",
          "properties": {"since": "2024"}, "source_ids": ["doc#42"]}])
    check("node line has id", "[节点] n123: Company(Acme Inc.)" in text)
    check("edge line has ids", "n123 -[EXECUTIVE]-> n456" in text)
    check("source attached", "doc#42" in text)
    check("token estimate sane", estimate_tokens(text) > 0)
    print(text)


def test_retrieval_modes():
    print("[three retrieval modes]")
    s = GraphStore(_tmpdb())
    ids = {}
    for name in ["Acme公司", "张三", "李四"]:
        ids[name] = s.upsert_entity(name)
    fact = s.add_node(content="张三担任Acme公司CEO", node_type="Fact",
                      source_ids=["doc#1"])
    s.add_edge(ids["张三"], fact, edge_type="involves")
    s.add_edge(fact, ids["Acme公司"], edge_type="involves")
    s.add_edge(ids["张三"], ids["李四"], edge_type="related")
    nodes = [dict(n, embedding=None) for n in s.query_nodes()]
    edges = [s._row_to_edge(r) for r in s.conn.execute("SELECT * FROM edges")]
    g = alg.to_networkx(nodes, edges)
    # mode 1: k-hop
    sub, ue = alg.k_hop_subgraph(g, [ids["张三"]], hops=2, max_nodes=10, max_edges=10)
    check("k-hop includes fact+company", fact in sub and ids["Acme公司"] in sub)
    # mode 3: paths
    paths = alg.top_k_paths(g, ids["张三"], ids["Acme公司"], k=3, max_len=4)
    check("path found", len(paths) >= 1 and paths[0][0] == ids["张三"]
          and paths[0][-1] == ids["Acme公司"])
    # mode 3: PPR
    scores = alg.personalized_pagerank(g, [ids["张三"]])
    check("ppr ranks seed neighborhood", scores.get(ids["张三"], 0) > 0)
    # mode 2: communities
    comms = alg.detect_communities(g)
    check("communities detected", len(comms) >= 1)
    s.close()


def test_agent_planning():
    print("[skill graph + A* planning + constraints + replay]")
    from graph_native_mcp.servers import agent as ag
    os.environ["GNM_DB"] = _tmpdb()
    ag._store = None
    s = ag.store()
    ag.set_world_state({"库存.苹果": 5, "资金": 100})
    sk_buy = ag.register_skill(
        name="采购苹果",
        precondition=[{"key": "资金", "op": "ge", "value": 50}],
        effect=[{"key": "库存.苹果", "op": "add", "value": 10},
                {"key": "资金", "op": "sub", "value": 50}])[ "skill_id"]
    sk_sell = ag.register_skill(
        name="卖出苹果",
        precondition=[{"key": "库存.苹果", "op": "ge", "value": 10}],
        effect=[{"key": "库存.苹果", "op": "sub", "value": 10},
                {"key": "资金", "op": "add", "value": 80}])[ "skill_id"]
    res = ag.plan(goal=[{"key": "资金", "op": "ge", "value": 120}])
    check("plan found", res["status"] == "plan-found", str(res)[:200])
    if res["status"] == "plan-found":
        names = [st["name"] for st in res["plan"]]
        check("plan buys then sells", names == ["采购苹果", "卖出苹果"], str(names))
        check("plan predicts goal state",
              res["plan"][-1]["expected_state_after"]["资金"] >= 120)
    # precondition hard-check on apply_action
    bad = ag.apply_action(sk_sell, delta={})
    check("apply blocked when precondition fails", "error" in bad)
    good = ag.apply_action(sk_buy, delta={"库存.苹果": {"op": "add", "value": 10},
                                          "资金": {"op": "sub", "value": 50}})
    check("apply succeeds with delta", good.get("ok") is True)
    check("world updated", ag.get_world_state()["资金"] == 50)
    # report_result -> stats affect planning cost
    for _ in range(5):
        ag.report_result(sk_sell, success=False)
    skills = {sk_["name"]: sk_ for sk_ in ag.get_skills()}
    check("failure stats recorded", skills["卖出苹果"]["success_rate"] == 0.0)
    # constraints: forbid the buy skill, reset world so sell alone can't reach goal
    ag.set_world_state({"库存.苹果": 5, "资金": 100})
    ag.set_constraint(forbidden_skills=[sk_buy])
    res2 = ag.plan(goal=[{"key": "资金", "op": "ge", "value": 120}])
    check("forbidden skill blocks plan", res2["status"] == "no-plan-found", str(res2)[:200])
    # gates: mark sell skill as human gate
    ag.set_constraint(gate_skills=[sk_sell])
    ag.set_world_state({"库存.苹果": 50, "资金": 100})
    res3 = ag.plan(goal=[{"key": "资金", "op": "ge", "value": 150}])
    check("gated skill flagged in plan", res3.get("needs_human_gate") is True, str(res3)[:200])
    # replay
    events = ag.replay()["events"]
    check("replay has audit events", len(events) >= 5, str(len(events)))
    check("replay has world/action events",
          {"world", "action", "plan", "result", "constraint", "skill"} <=
          {e["event_type"] for e in events})


def test_extract_skills():
    print("[trajectory -> skill extraction]")
    from graph_native_mcp.servers import agent as ag
    os.environ["GNM_DB"] = _tmpdb()
    ag._store = None
    ag.set_world_state({"库存": 0, "资金": 0})
    traj = [
        {"action": "采购", "goal": "进货", "params": {"item": "apple"},
         "world_before": {"库存": 0, "资金": 100},
         "world_after": {"库存": 10, "资金": 50}, "success": True},
        {"action": "上架", "goal": "进货", "params": {},
         "world_before": {"库存": 10, "资金": 50},
         "world_after": {"库存": 10, "资金": 50, "上架": True}, "success": True},
    ]
    res = ag.extract_skills(trajectory=traj)
    check("skills created from trajectory", len(res["created"]) >= 1, str(res)[:200])
    again = ag.extract_skills(trajectory=traj[:1])
    check("repeat trajectory merges stats", len(again["merged"]) == 1, str(again)[:200])
    names = [sk["name"] for sk in ag.get_skills()]
    check("effect from world diff", "采购" in names, str(names))


def test_memory_server_tools():
    print("[graph-memory server tools]")
    from graph_native_mcp.servers import memory as mem
    os.environ["GNM_DB"] = _tmpdb()
    mem._store = None
    r = mem.write_memory(content="用户偏好深色主题", memory_type="Preference",
                         entities=["用户"], source_ids=["chat#1"])
    check("preference written", r["status"] == "created", str(r))
    r2 = mem.write_memory(content="用户偏好浅色主题", memory_type="Preference",
                          entities=["用户"], source_ids=["chat#2"])
    check("conflict preserved as edge", r2["status"] == "conflict-preserved"
          and len(r2["conflicts"]) == 1, str(r2))
    r3 = mem.write_memory(content="用户偏好浅色主题", memory_type="Preference",
                          entities=["用户"], source_ids=["chat#3"])
    check("duplicate merged", r3["status"] == "merged-duplicate", str(r3))
    r4 = mem.write_memory(content="用户偏好深色主题", memory_type="Preference",
                          entities=["用户"], source_ids=["chat#4"])
    check("re-assert original merges back", r4["status"] == "merged-duplicate", str(r4))
    # numeric fact + explicit correction -> supersede
    r5 = mem.write_memory(content="用户每天工作10小时", memory_type="Fact",
                          entities=["用户"], source_ids=["chat#5"])
    check("numeric fact written", r5["status"] == "created", str(r5))
    r6 = mem.write_memory(content="用户每天工作6小时", memory_type="Fact",
                          entities=["用户"], source_ids=["chat#6"],
                          explicit_correction=True)
    check("explicit correction supersedes", r6["status"] == "corrected"
          and len(r6["superseded"]) == 1, str(r6))
    hits = mem.semantic_search("主题偏好")
    check("semantic search returns hits", len(hits) >= 1, str(hits))
    ctx = mem.retrieve_context("用户偏好")
    check("retrieve_context yields evidence text", "用户" in ctx, ctx[:100])
    tl = mem.get_timeline("用户")
    check("timeline query works", tl["count"] >= 1, str(tl["count"]))
    st = mem.graph_stats()
    check("stats shows contradiction edge", st["contradiction_edges"] >= 1, str(st))


def test_rag_server_tools():
    print("[graph-rag server tools]")
    from graph_native_mcp.servers import rag
    os.environ["GNM_DB"] = _tmpdb()
    rag._store = None
    mem_setup()
    res = rag.search_local("张三在Acme公司担任什么职务")
    check("search_local returns evidence", "张三" in res["evidence"], str(res)[:200])
    check("citation closure present", len(res["citations"]["node_ids"]) >= 2)
    check("cost report present", res["cost"]["nodes"] >= 2)
    idx = rag.build_communities(min_community=1)
    check("communities built", idx.get("communities", 0) >= 1, str(idx)[:200])
    glob = rag.search_global("这些材料主要讲什么")
    check("search_global returns summaries", "社区" in glob.get("evidence", ""),
          str(glob)[:200])
    p = rag.search_paths("张三", source_entity="张三", target_entity="Acme公司")
    check("search_paths finds chain", "推理链" in p.get("evidence", ""), str(p)[:200])
    p2 = rag.search_paths("x", source_entity="张三", target_entity="不存在的实体")
    check("missing path reported honestly", "图上无证据" in p2.get("evidence", ""),
          str(p2)[:200])


def test_checkup():
    print("[graph_checkup]")
    from graph_native_mcp.servers import build as bd
    os.environ["GNM_DB"] = _tmpdb()
    bd._store = None
    mem_setup()
    rep = bd.graph_checkup()
    check("checkup returns summary", "summary" in rep and rep["summary"]["nodes"] >= 4,
          str(rep)[:200])
    check("suggestions listed", isinstance(rep["suggestions"], list))


def mem_setup():
    from graph_native_mcp.servers import memory as mem
    mem._store = None
    s = mem.store()
    zhang = s.upsert_entity("张三")
    acme = s.upsert_entity("Acme公司")
    li = s.upsert_entity("李四")
    fact = s.add_node(content="张三担任Acme公司CEO", node_type="Fact",
                      source_ids=["doc#1"])
    s.add_edge(zhang, fact, edge_type="involves")
    s.add_edge(fact, acme, edge_type="involves")
    s.add_edge(zhang, li, edge_type="related")


_TMP: list[str] = []


def _tmpdb() -> str:
    f = os.path.join(tempfile.mkdtemp(prefix="gnm_test_"), "graph.db")
    return f


def main():
    test_store_and_gate()
    test_bi_temporal_and_conflicts()
    test_decay_and_reinforce()
    test_serialization()
    test_retrieval_modes()
    test_agent_planning()
    test_extract_skills()
    test_memory_server_tools()
    test_rag_server_tools()
    test_checkup()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
