"""Evidence serialization: subgraph -> text with citation ids (book Ch.5.3.1).

The format is the hidden key of local-subgraph retrieval - node/edge ids stay
attached to every line so the LLM's citations can be checked against the
citation closure:

    [节点] n123: Company(Acme Inc.) — 摘要: ...
    [边]   n123 -[EXECUTIVE]-> n456 (since: 2024, 来源: doc#42)
"""

from __future__ import annotations


def serialize_evidence(nodes: list[dict], edges: list[dict]) -> str:
    lines = []
    for n in nodes:
        head = f"[节点] {n['node_id']}: {n.get('node_type', '')}({n.get('name', '')})"
        parts = []
        if n.get("content"):
            parts.append(f"摘要: {n['content'][:200]}")
        props = n.get("properties") or {}
        if props:
            kv = ", ".join(f"{k}={v}" for k, v in list(props.items())[:4])
            parts.append(kv)
        if n.get("source_ids"):
            parts.append("来源: " + ";".join(str(s) for s in n["source_ids"][:3]))
        lines.append(head + (" — " + " | ".join(parts) if parts else ""))
    for e in edges:
        props = e.get("properties") or {}
        kv = ", ".join(f"{k}: {v}" for k, v in list(props.items())[:3])
        src = ""
        if e.get("source_ids"):
            src = ", 来源: " + ";".join(str(s) for s in e["source_ids"][:2])
        line = (f"[边]   {e['src']} -[{e['edge_type']}]-> {e['dst']}"
                f" ({kv}{src})" if kv else
                f"[边]   {e['src']} -[{e['edge_type']}]-> {e['dst']}{src}")
        lines.append(line)
    return "\n".join(lines)


def estimate_tokens(text: str) -> int:
    """Rough token estimate: CJK-heavy text ~1 token/char, latin ~0.3/char."""
    cjk = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
    other = len(text) - cjk
    return int(cjk + other * 0.3)


def cost_report(nodes: list[dict], edges: list[dict], evidence: str) -> dict:
    """The audit triple: subgraph size + serialized token estimate (Ch.5.2.1
    cost model Cost = a|V| + b|E| + g*Tokens)."""
    return {"nodes": len(nodes), "edges": len(edges),
            "est_tokens": estimate_tokens(evidence),
            "evidence_chars": len(evidence)}
