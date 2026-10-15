"""Node/edge schema and the write gate (book Ch.6.2 / Ch.7.2.4).

Design decisions carried over from the book:
  - Every node carries source_ids + confidence; nodes without sources are
    refused entry into retrieval contexts (Ch.6.2.3 "字段即防线").
  - contradicts edges preserve conflicts instead of overwriting them.
  - The write gate is multi-valve: confidence threshold, schema validation,
    structural sanity. (Ch.7.2.4)
"""

from __future__ import annotations

# ---------------------------------------------------------------- node types
# Ch.6.2.1 memory-graph schema, extended with Ch.8 agent state types.
NODE_TYPES = {
    "Fact":        "long-lived true statement (content, confidence, source_ids)",
    "Event":       "timestamped occurrence (content, valid_time)",
    "Entity":      "person / org / object (name, type, aliases)",
    "Preference":  "user preference (content, validity)",
    "Task":        "task or goal state (status, deadline)",
    # --- Ch.8 graph-native extensions ---
    "Skill":       "reusable capability node (precondition/effect assertions)",
    "WorldState":  "singleton world-model node holding the assertion dict",
    "Constraint":  "policy node: forbidden skills / human gates",
    # --- generic fallback (gx-memory compatibility) ---
    "Knowledge":   "free-form knowledge chunk with embedding",
}

# ---------------------------------------------------------------- edge types
EDGE_TYPES = {
    # Ch.6.2.1
    "temporal":      "Event -> Event temporal order",
    "causal":        "Event -> Event causality",
    "involves":      "Event/Fact -> Entity participation",
    "supports":      "Fact -> Fact evidence support",
    "contradicts":   "Fact <-> Fact conflict, preserved not overwritten",
    "derives_from":  "Fact -> original Event (provenance)",
    # Ch.8
    "enables":       "Skill -> Skill dependency (precondition needs effect)",
    "supersedes":    "newer fact replaces older one (bi-temporal)",
    # generic
    "related":       "untyped association",
}

# write-gate defaults
DEFAULT_CONFIDENCE_THRESHOLD = 0.5


def validate_node(node_type: str | None, source_ids: list | None,
                  confidence: float | None) -> tuple[bool, str]:
    """First valve of the write gate. Returns (ok, reason)."""
    node_type = node_type or "Knowledge"
    if node_type not in NODE_TYPES:
        return False, f"unknown node_type '{node_type}', expected one of {sorted(NODE_TYPES)}"
    source_ids = source_ids or []
    if not isinstance(source_ids, list):
        return False, "source_ids must be a list"
    # Ch.6.2.3: a Fact without provenance must not enter the graph at all.
    if node_type == "Fact" and not source_ids:
        return False, "Fact nodes require non-empty source_ids (no unattributed facts)"
    confidence = 1.0 if confidence is None else float(confidence)
    if not (0.0 <= confidence <= 1.0):
        return False, f"confidence must be within [0,1], got {confidence}"
    return True, "ok"


def validate_edge(edge_type: str | None, src: str, dst: str,
                  existing_node_ids: set) -> tuple[bool, str]:
    """Structural valve: typed edge between existing nodes."""
    edge_type = edge_type or "related"
    if edge_type not in EDGE_TYPES:
        return False, f"unknown edge_type '{edge_type}', expected one of {sorted(EDGE_TYPES)}"
    if src not in existing_node_ids:
        return False, f"source node '{src}' does not exist"
    if dst not in existing_node_ids:
        return False, f"target node '{dst}' does not exist"
    return True, "ok"
