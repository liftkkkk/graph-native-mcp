"""graph-agent MCP server (book Ch.8) — the graph-native core.

This is the part the MCP ecosystem does not have yet. Four capabilities:

  - world model (8.1): state is an addressable assertion dict on a singleton
    WorldState node; every action produces a delta (dW); observation is a
    query; the delta log is the audit trail.
  - skill graph (8.2): skills carry machine-verifiable precondition/effect
    assertions (PDDL's grounded tradition, LLM-acquired), plus health
    statistics (cost, success_rate) updated from real outcomes.
  - graph planning (8.3): planning is path search (A*) over the skill graph;
    the learned cost (cost / success_rate) is the edge weight; replanning
    is local re-search, not a restart.
  - scaffolding (8.4): forbidden skills and human-in-the-loop gates are read
    BEFORE search, so constraints prune the plan space instead of punishing
    after execution.

Tools-as-graph-nodes note (8.2.4): the MCP tool declarations of all four
servers in this repo can themselves be registered as Tool nodes - see
register_tool_graph().

Deterministic by design: no LLM calls. The client LLM decides goals and
interprets plans; this server guarantees the verification machinery.
"""

from __future__ import annotations

import heapq
import itertools
import json

from mcp.server.fastmcp import FastMCP

from ..core.store import GraphStore, default_db_path

mcp = FastMCP("graph-agent")
_store: GraphStore | None = None

WORLD_ID = "world"
CONSTRAINT_ID = "constraints"

_OPS = {
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "gt": lambda a, b: _num(a) > _num(b),
    "ge": lambda a, b: _num(a) >= _num(b),
    "lt": lambda a, b: _num(a) < _num(b),
    "le": lambda a, b: _num(a) <= _num(b),
    "exists": lambda a, b: True,
}


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def store() -> GraphStore:
    global _store
    if _store is None:
        _store = GraphStore(default_db_path())
        _ensure_world()
    return _store


def _ensure_world():
    s = _store
    if not s.get_node(WORLD_ID):
        s.add_node(node_id=WORLD_ID, name="world", node_type="WorldState",
                   properties={}, gate=False, embed=False)


def _assertion_ok(state: dict, a: dict) -> bool:
    k, op = a.get("key"), a.get("op", "eq")
    if k not in state:
        return False
    fn = _OPS.get(op)
    return bool(fn and fn(state[k], a.get("value")))


def _apply_effects(state: dict, effects: list) -> dict:
    new = dict(state)
    for e in effects or []:
        k, op, v = e.get("key"), e.get("op", "set"), e.get("value")
        if k is None:
            continue
        if op == "set":
            new[k] = v
        elif op == "add":
            new[k] = _num(new.get(k, 0)) + _num(v)
        elif op == "sub":
            new[k] = _num(new.get(k, 0)) - _num(v)
        elif op == "delete":
            new.pop(k, None)
    return new


def _world_state(s: GraphStore) -> dict:
    n = s.get_node(WORLD_ID)
    return (n["properties"] if n else {}) or {}


def _set_world_state(s: GraphStore, state: dict):
    with s.lock:
        s.conn.execute("UPDATE nodes SET properties=? WHERE node_id=?",
                       (json.dumps(state, ensure_ascii=False), WORLD_ID))
        s.conn.commit()


def _constraints(s: GraphStore) -> dict:
    n = s.get_node(CONSTRAINT_ID)
    return (n["properties"] if n else {}) or {"forbidden_skills": [], "gate_skills": []}


# ================================================================ 8.1 world model
@mcp.tool()
def set_world_state(state: dict) -> dict:
    """初始化/整体替换世界模型（Ch.8.1）。state 是断言字典，
    如 {"inventory.apple": 10, "user.level": "vip"}。
    世界模型可寻址、可增量更新、可回放——这是它与「上下文塞入」的本质区别。"""
    s = store()
    old = _world_state(s)
    _set_world_state(s, dict(state))
    s.log_event("world", {"tool": "set_world_state", "before": old, "after": state})
    return {"ok": True, "keys": len(state)}


@mcp.tool()
def get_world_state(keys: list = None) -> dict:
    """查询当前世界状态（观察世界 = 一次查询）。可选只取指定 key。"""
    state = _world_state(store())
    if keys:
        state = {k: state[k] for k in keys if k in state}
    return state


@mcp.tool()
def apply_action(skill_id: str, params: dict = None,
                 delta: dict = None, actor: str = "agent") -> dict:
    """执行一个动作，产生 ΔW 增量更新（Ch.8.1）。
    - skill_id: 要执行的技能节点
    - params: 实例化参数（记录进执行图，审计用）
    - delta: 状态变更。支持三种算子：
      {"key": value}            → set（最常用，直接写 delta={"库存": 5}）
      {"key": {"op":"add","value":2}} → 累加
      {"key": {"op":"delete"}}  → 删除
    执行前自动校验该技能的前置断言；不满足则拒绝并返回缺口。
    每次执行写入事件日志（执行图），供 replay 回放。"""
    s = store()
    skill = s.get_node(skill_id)
    if not skill or skill["node_type"] != "Skill":
        return {"error": f"skill '{skill_id}' not found"}
    state = _world_state(s)
    preconds = skill["properties"].get("precondition", [])
    missing = [a for a in preconds if not _assertion_ok(state, a)]
    if missing:
        return {"error": "precondition not satisfied", "missing": missing,
                "hint": "前置断言是硬校验：先满足前置（或换 plan 重规划）再执行"}
    normalized = []
    for k, v in (delta or {}).items():
        if isinstance(v, dict) and "op" in v:
            normalized.append({"key": k, "op": v["op"], "value": v.get("value")})
        else:
            normalized.append({"key": k, "op": "set", "value": v})
    new_state = _apply_effects(state, normalized)
    _set_world_state(s, new_state)
    event_id = s.log_event("action", {
        "tool": "apply_action", "skill": skill_id, "params": params or {},
        "delta": normalized, "before": state, "after": new_state, "actor": actor})
    return {"ok": True, "event_id": event_id,
            "changed_keys": [d["key"] for d in normalized],
            "state_after": new_state}


# ================================================================ 8.2 skill graph
@mcp.tool()
def register_skill(name: str, precondition: list = None, effect: list = None,
                   params: dict = None, cost: float = 1.0,
                   enabled_by: list = None) -> dict:
    """注册一个技能节点（Ch.8.2.1）。技能不是自然语言描述，是可机器验证的：
    - precondition: 前置断言列表 [{"key": "库存.苹果", "op": ">=", "value": 10}]
      算子：eq/ne/gt/ge/lt/le/exists
    - effect: 效果断言 [{"key": "库存.苹果", "op": "sub", "value": 10},
                        {"key": "订单数", "op": "add", "value": 1}]
      算子：set/add/sub/delete
    - cost: 期望代价（时间/费用/风险，进入 A* 的边权）
    - enabled_by: 依赖的其他技能节点 ID 列表（建 enables 边）
    同名技能自动版本化：旧版本保留，统计延续。"""
    s = store()
    existing = s.query_nodes(name_like=name, node_type="Skill")
    version = 1
    for e in existing:
        version = max(version, (e["properties"].get("version", 1)) + 1)
    props = {"precondition": precondition or [], "effect": effect or [],
             "params": params or {}, "cost": float(cost), "version": version,
             "success_count": 0, "fail_count": 0,
             "success_rate": None}
    sid = s.add_node(name=name, node_type="Skill", properties=props, gate=False)
    for dep in (enabled_by or []):
        if dep in s.all_node_ids():
            s.add_edge(dep, sid, edge_type="enables")
    s.log_event("skill", {"tool": "register_skill", "skill": sid, "name": name})
    return {"skill_id": sid, "name": name, "version": version,
            "precondition": precondition or [], "effect": effect or []}


@mcp.tool()
def get_skills(include_stats: bool = True) -> list:
    """列出全部技能及其前置/效果断言（含健康度统计：成功率、调用次数）。"""
    out = []
    for n in store().query_nodes(node_type="Skill"):
        p = n["properties"]
        item = {"skill_id": n["node_id"], "name": n["name"],
                "precondition": p.get("precondition", []),
                "effect": p.get("effect", []), "version": p.get("version", 1)}
        if include_stats:
            sc, fc = p.get("success_count", 0), p.get("fail_count", 0)
            item.update({"cost": p.get("cost", 1.0),
                         "success_count": sc, "fail_count": fc,
                         "success_rate": (round(sc / (sc + fc), 3)
                                          if sc + fc else None)})
        out.append(item)
    return out


def _effect_overlap(effect: list, diff_keys: list) -> float:
    """Fraction of the new effect keys already covered by an existing skill.
    Merge when >= 0.5 (Ch.8.2.2 相似度>阈值: 合并并更新统计)."""
    if not diff_keys:
        return 0.0
    covered = {d.get("key") for d in effect}
    return len(covered & set(diff_keys)) / len(set(diff_keys))


@mcp.tool()
def extract_skills(trajectory: list) -> dict:
    """从执行轨迹自动提取技能（Ch.8.2.2 SKILL_EXTRACTION）。
    轨迹格式（JSON 数组，每步）：
      {"action": "下单", "goal": "补货", "params": {...},
       "world_before": {...}, "world_after": {...}, "success": true}
    - 按 goal 切分段
    - 效果 = world_before/after 的 diff（前后世界图差即效果）
    - 前置取 diff 涉及 key 的存在性断言（从轨迹学前置是开放问题，
      这里取保守的 exists 断言）
    - 与既有技能同名同效果域时合并并更新统计
    返回新建/合并的技能清单。"""
    s = store()
    created, merged = [], []
    segments, cur_goal, cur = [], None, []
    for step in trajectory or []:
        if step.get("goal") != cur_goal and cur:
            segments.append(cur)
            cur = []
        cur_goal = step.get("goal")
        cur.append(step)
    if cur:
        segments.append(cur)
    for seg in segments:
        if not seg or not all(st.get("success", True) for st in seg):
            continue
        name = seg[0].get("action") or seg[0].get("goal") or "unnamed-skill"
        before, after = seg[0].get("world_before") or {}, seg[-1].get("world_after") or {}
        diff_keys = [k for k in set(before) | set(after) if before.get(k) != after.get(k)]
        if not diff_keys:
            continue
        effect = []
        for k in diff_keys:
            bv, av = before.get(k), after.get(k)
            if isinstance(bv, (int, float)) and isinstance(av, (int, float)):
                effect.append({"key": k, "op": "add", "value": av - bv})
            else:
                effect.append({"key": k, "op": "set", "value": av})
        precondition = [{"key": k, "op": "exists"} for k in diff_keys if k in before]
        existing = [n for n in s.query_nodes(name_like=name, node_type="Skill")
                    if _effect_overlap(n["properties"].get("effect", []), diff_keys) >= 0.5]
        if existing:
            n = existing[0]
            p = n["properties"]
            p["success_count"] = p.get("success_count", 0) + 1
            with s.lock:
                s.conn.execute("UPDATE nodes SET properties=? WHERE node_id=?",
                               (json.dumps(p, ensure_ascii=False), n["node_id"]))
                s.conn.commit()
            merged.append({"skill_id": n["node_id"], "name": name})
        else:
            res = register_skill(name=name, precondition=precondition,
                                 effect=effect, params=seg[0].get("params"))
            created.append(res)
    s.log_event("skill", {"tool": "extract_skills",
                          "created": len(created), "merged": len(merged)})
    return {"created": created, "merged": merged,
            "segments": len(segments),
            "note": "preconditions from trajectories are conservative (exists); "
                    "tighten them via register_skill or LLM post-edit"}


@mcp.tool()
def report_result(skill_id: str, success: bool) -> dict:
    """回报技能执行结果，更新统计属性 θ（Ch.8.2/8.3）。
    同一技能连续失败会拉低成功率，下次 plan 自动绕开（代价=cost/成功率）。"""
    s = store()
    n = s.get_node(skill_id)
    if not n or n["node_type"] != "Skill":
        return {"error": f"skill '{skill_id}' not found"}
    p = n["properties"]
    p["success_count"] = p.get("success_count", 0) + (1 if success else 0)
    p["fail_count"] = p.get("fail_count", 0) + (0 if success else 1)
    with s.lock:
        s.conn.execute("UPDATE nodes SET properties=? WHERE node_id=?",
                       (json.dumps(p, ensure_ascii=False), skill_id))
        s.conn.commit()
    sc, fc = p["success_count"], p["fail_count"]
    s.log_event("result", {"skill": skill_id, "success": success})
    return {"skill_id": skill_id, "success_count": sc, "fail_count": fc,
            "success_rate": round(sc / (sc + fc), 3)}


@mcp.tool()
def validate_skill(skill_id: str) -> dict:
    """机器验证一个技能：前置断言当前是否满足、效果断言是否曾兑现
    （效果断言按『应用后世界应满足』语义校验当前世界）。"""
    s = store()
    n = s.get_node(skill_id)
    if not n or n["node_type"] != "Skill":
        return {"error": f"skill '{skill_id}' not found"}
    state = _world_state(s)
    pre = n["properties"].get("precondition", [])
    eff = n["properties"].get("effect", [])
    pre_results = [{"assertion": a, "satisfied": _assertion_ok(state, a)} for a in pre]
    simulated = _apply_effects(state, eff)
    eff_results = [{"assertion": e, "reachable": _assertion_ok(simulated, e)}
                   for e in eff if e.get("op") in ("set", "eq")]
    ok = all(r["satisfied"] for r in pre_results)
    return {"skill_id": skill_id, "preconditions_satisfied": ok,
            "details": pre_results, "effect_simulation": eff_results,
            "verdict": "ready" if ok else "blocked"}


# ================================================================ 8.3 planning
@mcp.tool()
def plan(goal: list, max_steps: int = 8, max_expansions: int = 500) -> dict:
    """图规划（Ch.8.3）：规划 = 技能图上的 A* 路径搜索。
    - goal: 目标断言列表（与 precondition 同格式）
    边权 = cost / max(success_rate, 0.05)：学到的代价当启发式经验，
    连续失败的技能自动被绕开。
    启发式 = 未满足目标断言数（可采纳的下界）。
    约束在搜索前生效（Ch.8.4）：forbidden 技能直接剪掉，
    gate 技能保留在计划里但标记 human_gate=true，执行前需人工确认。
    返回计划（技能序列 + 每步的世界状态预测 + 总代价）。"""
    s = store()
    state0 = _world_state(s)
    if all(_assertion_ok(state0, g) for g in goal):
        return {"status": "already-satisfied", "plan": [], "total_cost": 0.0}
    cons = _constraints(s)
    forbidden = set(cons.get("forbidden_skills", []))
    gates = set(cons.get("gate_skills", []))
    skills = []
    for n in s.query_nodes(node_type="Skill"):
        p = n["properties"]
        sc, fc = p.get("success_count", 0), p.get("fail_count", 0)
        rate = (sc / (sc + fc)) if (sc + fc) else 0.5  # 未验证技能按 0.5 起步
        skills.append({"skill_id": n["node_id"], "name": n["name"],
                       "precondition": p.get("precondition", []),
                       "effect": p.get("effect", []),
                       "cost": float(p.get("cost", 1.0)) / max(rate, 0.05),
                       "gated": n["node_id"] in gates})
    counter = itertools.count()
    open_heap = [(len(goal), next(counter), 0.0, state0, [])]
    seen = {frozenset(state0.items())}
    expansions = 0
    while open_heap and expansions < max_expansions:
        f, _, g_cost, state, seq = heapq.heappop(open_heap)
        if all(_assertion_ok(state, gl) for gl in goal):
            steps = []
            sim = state0
            for sk_id in seq:
                sk = next(x for x in skills if x["skill_id"] == sk_id)
                sim = _apply_effects(sim, sk["effect"])
                steps.append({"skill_id": sk_id, "name": sk["name"],
                              "cost": round(sk["cost"], 3),
                              "human_gate": sk["gated"],
                              "expected_state_after": sim})
            result = {"status": "plan-found", "plan": steps,
                      "total_cost": round(g_cost, 3), "expansions": expansions,
                      "needs_human_gate": any(st["human_gate"] for st in steps)}
            s.log_event("plan", {"goal": goal, "steps": [st["skill_id"] for st in steps],
                                 "cost": g_cost})
            return result
        expansions += 1
        for sk in skills:
            if sk["skill_id"] in forbidden or len(seq) >= max_steps:
                continue
            if not all(_assertion_ok(state, a) for a in sk["precondition"]):
                continue
            nxt = _apply_effects(state, sk["effect"])
            key = frozenset(nxt.items())
            if key in seen:
                continue
            seen.add(key)
            h = sum(0 if _assertion_ok(nxt, gl) else 1 for gl in goal)
            heapq.heappush(open_heap, (g_cost + sk["cost"] + h, next(counter),
                                       g_cost + sk["cost"], nxt, seq + [sk["skill_id"]]))
    return {"status": "no-plan-found", "reason":
            "目标在当前技能图上不可达（或超出 max_steps/max_expansions 预算）",
            "hint": "注册新技能、放宽目标断言，或检查前置是否过紧",
            "expansions": expansions}


# ================================================================ 8.4 scaffolding
@mcp.tool()
def set_constraint(forbidden_skills: list = None,
                   gate_skills: list = None) -> dict:
    """约束图（Ch.8.4）：业务红线在搜索前生效，不是执行后追责。
    - forbidden_skills: 绝对禁止的技能 ID（规划器直接剪掉）
    - gate_skills: 人工确认点（高风险/低成功率/不可逆动作）——
      计划保留这些技能但标记 human_gate，执行前必须人工放行
    文本规则约束不住 Agent（Poise/ToolPrivBench/CKA-Agent 的教训），
    强制门控才是防线。"""
    s = store()
    props = {"forbidden_skills": forbidden_skills or [],
             "gate_skills": gate_skills or []}
    with s.lock:
        existing = s.get_node(CONSTRAINT_ID)
        if existing:
            s.conn.execute("UPDATE nodes SET properties=? WHERE node_id=?",
                           (json.dumps(props, ensure_ascii=False), CONSTRAINT_ID))
        else:
            s.add_node(node_id=CONSTRAINT_ID, name="constraints",
                       node_type="Constraint", properties=props,
                       gate=False, embed=False)
        s.conn.commit()
    s.log_event("constraint", {"forbidden": forbidden_skills or [],
                               "gates": gate_skills or []})
    return {"ok": True, "forbidden_skills": forbidden_skills or [],
            "gate_skills": gate_skills or []}


@mcp.tool()
def replay(after_event_id: int = 0, event_type: str = "") -> dict:
    """审计回放（Ch.8.1 轨迹可回放 / 第9章事件溯源）：
    返回事件日志（世界变更/规划/执行/结果/约束），带时间戳。
    ΔW 序列本身就是完整的审计日志。"""
    events = store().get_events(after_event_id, event_type)
    return {"events": events, "count": len(events)}


def run():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    run()
