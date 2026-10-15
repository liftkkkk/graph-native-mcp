"""SQLite property-graph store with bi-temporal timestamps.

Replaces gx-memory's GX-engine + pickle backend with a portable single-file
SQLite database, and adds the Ch.6 capabilities gx-memory lacked:

  - bi-temporal timestamps: valid_time (when the fact happened) is separated
    from transaction time (when the system recorded it). Supports
    out-of-order arrival, retroactive correction, and "what did the system
    know at time T" queries. (Ch.6.2.2)
  - conflict handling: contradicts edges preserve conflicts; superseded
    nodes are marked, never deleted. (Ch.6.3.2, 保留式更新)
  - memory weight with decay / reinforcement hooks. (Ch.6.3.1)
  - an append-only event log for the agent server's audit replay. (Ch.8/9)

Concurrency note: SQLite with check_same_thread=False + a process-wide lock.
MCP servers are single-process stdio services; this is sufficient.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone

from .embeddings import cosine, get_embedding_provider
from .schema import validate_node, validate_edge, DEFAULT_CONFIDENCE_THRESHOLD

_SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    node_id      TEXT PRIMARY KEY,
    node_type    TEXT NOT NULL DEFAULT 'Knowledge',
    name         TEXT NOT NULL DEFAULT '',
    content      TEXT NOT NULL DEFAULT '',
    properties   TEXT NOT NULL DEFAULT '{}',
    embedding    TEXT,
    source_ids   TEXT NOT NULL DEFAULT '[]',
    confidence   REAL NOT NULL DEFAULT 1.0,
    valid_from   TEXT,
    valid_until  TEXT,
    created_at   TEXT NOT NULL,
    superseded_by TEXT,
    weight       REAL NOT NULL DEFAULT 1.0,
    last_accessed TEXT
);
CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type);
CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes(name);

CREATE TABLE IF NOT EXISTS edges (
    edge_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    src         TEXT NOT NULL REFERENCES nodes(node_id),
    dst         TEXT NOT NULL REFERENCES nodes(node_id),
    edge_type   TEXT NOT NULL DEFAULT 'related',
    properties  TEXT NOT NULL DEFAULT '{}',
    source_ids  TEXT NOT NULL DEFAULT '[]',
    confidence  REAL NOT NULL DEFAULT 1.0,
    valid_from  TEXT,
    valid_until TEXT,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst);
CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(edge_type);

CREATE TABLE IF NOT EXISTS communities (
    community_id INTEGER PRIMARY KEY,
    level        INTEGER NOT NULL DEFAULT 0,
    parent_id    INTEGER,
    member_ids   TEXT NOT NULL,
    summary      TEXT NOT NULL DEFAULT '',
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    event_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload    TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_db_path() -> str:
    p = os.environ.get("GNM_DB")
    if p:
        return p
    home = os.path.expanduser("~")
    d = os.path.join(home, ".gnm")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "graph.db")


class GraphStore:
    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or default_db_path()
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)) or ".", exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(_SCHEMA)
        self.conn.commit()
        self.lock = threading.RLock()
        self.embedder = get_embedding_provider()

    # ------------------------------------------------------------ helpers
    def _row_to_node(self, r) -> dict:
        return {
            "node_id": r["node_id"], "node_type": r["node_type"],
            "name": r["name"], "content": r["content"],
            "properties": json.loads(r["properties"]),
            "embedding": json.loads(r["embedding"]) if r["embedding"] else None,
            "source_ids": json.loads(r["source_ids"]),
            "confidence": r["confidence"],
            "valid_from": r["valid_from"], "valid_until": r["valid_until"],
            "created_at": r["created_at"], "superseded_by": r["superseded_by"],
            "weight": r["weight"], "last_accessed": r["last_accessed"],
        }

    def _row_to_edge(self, r) -> dict:
        return {
            "edge_id": r["edge_id"], "src": r["src"], "dst": r["dst"],
            "edge_type": r["edge_type"],
            "properties": json.loads(r["properties"]),
            "source_ids": json.loads(r["source_ids"]),
            "confidence": r["confidence"],
            "valid_from": r["valid_from"], "valid_until": r["valid_until"],
            "created_at": r["created_at"],
        }

    # ------------------------------------------------------------ writes
    def add_node(self, name: str = "", content: str = "", node_type: str = "Knowledge",
                 properties: dict | None = None, source_ids: list | None = None,
                 confidence: float | None = None, valid_from: str | None = None,
                 node_id: str | None = None, embed: bool = True,
                 gate: bool = True) -> str:
        """Insert a node. Gate = False skips schema validation (internal use)."""
        if gate:
            ok, reason = validate_node(node_type, source_ids, confidence)
            if not ok:
                raise ValueError(reason)
        node_id = node_id or f"n{uuid.uuid4().hex[:10]}"
        embedding = None
        if embed and (content or name):
            try:
                embedding = self.embedder.embed([f"{name} {content}".strip()])[0]
            except Exception:
                embedding = None
        with self.lock:
            self.conn.execute(
                "INSERT INTO nodes (node_id, node_type, name, content, properties,"
                " embedding, source_ids, confidence, valid_from, created_at, last_accessed)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (node_id, node_type or "Knowledge", name, content,
                 json.dumps(properties or {}, ensure_ascii=False),
                 json.dumps(embedding) if embedding else None,
                 json.dumps(source_ids or [], ensure_ascii=False),
                 1.0 if confidence is None else float(confidence),
                 valid_from, _now(), _now()))
            self.conn.commit()
        return node_id

    def add_edge(self, src: str, dst: str, edge_type: str = "related",
                 properties: dict | None = None, source_ids: list | None = None,
                 confidence: float | None = None, valid_from: str | None = None,
                 gate: bool = True) -> int:
        if gate:
            ids = {r["node_id"] for r in self.conn.execute("SELECT node_id FROM nodes")}
            ok, reason = validate_edge(edge_type, src, dst, ids)
            if not ok:
                raise ValueError(reason)
        with self.lock:
            cur = self.conn.execute(
                "INSERT INTO edges (src, dst, edge_type, properties, source_ids,"
                " confidence, valid_from, created_at) VALUES (?,?,?,?,?,?,?,?)",
                (src, dst, edge_type or "related",
                 json.dumps(properties or {}, ensure_ascii=False),
                 json.dumps(source_ids or [], ensure_ascii=False),
                 1.0 if confidence is None else float(confidence),
                 valid_from, _now()))
            self.conn.commit()
            return cur.lastrowid

    def upsert_entity(self, name: str, aliases: list | None = None,
                      properties: dict | None = None) -> str:
        """Entity resolution helper: reuse an existing Entity with the same
        name/alias, otherwise create one. (Ch.7.2 disambiguation layer.)"""
        with self.lock:
            row = self.conn.execute(
                "SELECT node_id FROM nodes WHERE node_type='Entity' AND name=?",
                (name,)).fetchone()
            if row:
                return row["node_id"]
            props = dict(properties or {})
            if aliases:
                props["aliases"] = aliases
            return self.add_node(name=name, node_type="Entity", properties=props,
                                 embed=False, gate=False)

    def delete_node(self, node_id: str) -> bool:
        with self.lock:
            # edges first: FK constraints reference nodes
            self.conn.execute("DELETE FROM edges WHERE src=? OR dst=?", (node_id, node_id))
            cur = self.conn.execute("DELETE FROM nodes WHERE node_id=?", (node_id,))
            self.conn.commit()
            return cur.rowcount > 0

    def delete_edge(self, edge_id: int) -> bool:
        with self.lock:
            cur = self.conn.execute("DELETE FROM edges WHERE edge_id=?", (edge_id,))
            self.conn.commit()
            return cur.rowcount > 0

    # ------------------------------------------------------------ reads
    def get_node(self, node_id: str) -> dict | None:
        r = self.conn.execute("SELECT * FROM nodes WHERE node_id=?", (node_id,)).fetchone()
        return self._row_to_node(r) if r else None

    def get_edge(self, edge_id: int) -> dict | None:
        r = self.conn.execute("SELECT * FROM edges WHERE edge_id=?", (edge_id,)).fetchone()
        return self._row_to_edge(r) if r else None

    def all_node_ids(self) -> set:
        return {r["node_id"] for r in self.conn.execute("SELECT node_id FROM nodes")}

    def query_nodes(self, name_like: str = "", node_type: str = "",
                    attribute_filters: dict | None = None,
                    active_only: bool = True) -> list[dict]:
        sql, args = "SELECT * FROM nodes WHERE 1=1", []
        if name_like:
            sql += " AND name LIKE ?"
            args.append(f"%{name_like}%")
        if node_type:
            sql += " AND node_type=?"
            args.append(node_type)
        if active_only:
            sql += " AND superseded_by IS NULL"
        out = []
        for r in self.conn.execute(sql, args):
            n = self._row_to_node(r)
            if attribute_filters:
                props = n["properties"]
                if any(props.get(k) != v for k, v in attribute_filters.items()):
                    continue
            out.append(n)
        return out

    def semantic_search(self, query: str, top_k: int = 5, threshold: float = 0.0,
                        node_type: str = "", reinforce: bool = True) -> list[dict]:
        """Cosine search over stored embeddings. Retrieval hits reinforce the
        memory weight (Ch.6.3.1 使用投票)."""
        q = self.embedder.embed([query])[0]
        scored = []
        for r in self.conn.execute("SELECT * FROM nodes WHERE embedding IS NOT NULL"
                                   " AND superseded_by IS NULL"):
            n = self._row_to_node(r)
            if node_type and n["node_type"] != node_type:
                continue
            # unattributed nodes are barred from retrieval contexts (Ch.6.2.3)
            if n["node_type"] == "Fact" and not n["source_ids"]:
                continue
            score = cosine(q, n["embedding"])
            if score >= threshold:
                scored.append((score, n))
        scored.sort(key=lambda x: -x[0])
        top = [n for _, n in scored[:top_k]]
        if reinforce and top:
            now = _now()
            with self.lock:
                for n in top:
                    self.conn.execute(
                        "UPDATE nodes SET weight = weight * 1.05, last_accessed=?"
                        " WHERE node_id=?", (now, n["node_id"]))
                self.conn.commit()
        return [{"node_id": n["node_id"], "node_type": n["node_type"],
                 "name": n["name"], "content": n["content"],
                 "score": round(s, 4)}
                for s, n in scored[:top_k]]

    def get_neighbors(self, node_id: str, hops: int = 1, direction: str = "both",
                      edge_types: list | None = None, active_only: bool = True) -> dict:
        """Layered multi-hop neighborhood. Returns {hop: {nodes, edges}}."""
        frontier = {node_id}
        seen = {node_id}
        result = {}
        for h in range(1, max(1, hops) + 1):
            next_frontier, layer_edges = set(), []
            for nid in frontier:
                for r in self._incident(nid, direction, active_only):
                    e = self._row_to_edge(r)
                    if edge_types and e["edge_type"] not in edge_types:
                        continue
                    other = e["dst"] if e["src"] == nid else e["src"]
                    layer_edges.append(e)
                    if other not in seen:
                        next_frontier.add(other)
            nodes = []
            for nid in next_frontier:
                n = self.get_node(nid)
                if n:
                    seen.add(nid)
                    nodes.append(self._public(n))
            result[f"hop{h}"] = {"nodes": nodes, "edges": layer_edges}
            frontier = next_frontier
            if not frontier:
                break
        return result

    def _incident(self, node_id: str, direction: str = "both", active_only: bool = True):
        if direction == "out":
            sql, args = "SELECT * FROM edges WHERE src=?", [node_id]
        elif direction == "in":
            sql, args = "SELECT * FROM edges WHERE dst=?", [node_id]
        else:
            sql, args = "SELECT * FROM edges WHERE (src=? OR dst=?)", [node_id, node_id]
        if active_only:
            sql += " AND valid_until IS NULL"
        return self.conn.execute(sql, args)

    @staticmethod
    def _public(n: dict) -> dict:
        return {k: n[k] for k in ("node_id", "node_type", "name", "content",
                                  "properties", "source_ids", "confidence",
                                  "valid_from", "valid_until", "created_at", "weight")}

    def edges_between(self, node_ids: set) -> list[dict]:
        ids = list(node_ids)
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        rows = self.conn.execute(
            "SELECT * FROM edges WHERE valid_until IS NULL"
            " AND src IN (%s) AND dst IN (%s)" % (placeholders, placeholders),
            ids + ids)
        return [self._row_to_edge(r) for r in rows]

    # ---------------------------------------------------- bi-temporal / conflicts
    def get_timeline(self, entity: str, as_of: str | None = None) -> list[dict]:
        """Bi-temporal query: everything involving the entity that was
        (a) valid and (b) already known to the system at time `as_of`.
        as_of filters BOTH valid_from (fact time) and created_at (system time)
        unless as_of starts with 'valid:' which filters fact time only."""
        ent_rows = self.query_nodes(name_like=entity, node_type="Entity")
        ent_ids = {r["node_id"] for r in ent_rows}
        items, seen_ids = [], set()
        for nid in ent_ids:
            for raw in self._incident(nid, "both", active_only=False):
                e = self._row_to_edge(raw)
                key = ("e", e["edge_id"])
                if key not in seen_ids:
                    seen_ids.add(key)
                    items.append(e)
                # collect the fact/event nodes on the other endpoint
                for end in (e["src"], e["dst"]):
                    if end in ent_ids:
                        continue
                    nkey = ("n", end)
                    if nkey in seen_ids:
                        continue
                    n = self.get_node(end)
                    if n and n["node_type"] in ("Fact", "Event", "Preference", "Task"):
                        seen_ids.add(nkey)
                        items.append(n)
        if as_of:
            valid_only = as_of.startswith("valid:")
            t = as_of[6:] if valid_only else as_of
            def _known(x):
                vf = x.get("valid_from")
                ca = x.get("created_at")
                ok_v = (not vf) or vf <= t
                ok_t = valid_only or (not ca) or ca <= t
                return ok_v and ok_t
            items = [x for x in items if _known(x)]
        items.sort(key=lambda x: x.get("valid_from") or x.get("created_at") or "")
        return items

    def find_conflict_candidates(self, content: str, entity_ids: set) -> list[dict]:
        """Candidates for conflict detection: semantically similar AND sharing
        an entity (Ch.6.3.2)."""
        if not entity_ids:
            return []
        q = self.embedder.embed([content])[0]
        cands = []
        for r in self.conn.execute(
                "SELECT * FROM nodes WHERE superseded_by IS NULL"
                " AND node_type IN ('Fact','Event','Preference','Task')"):
            n = self._row_to_node(r)
            shares = any(
                e["src"] in entity_ids or e["dst"] in entity_ids
                for e in self._incident(n["node_id"], "both", active_only=False))
            if shares and n["embedding"]:
                cands.append((cosine(q, n["embedding"]), n))
        cands.sort(key=lambda x: -x[0])
        return [n for s, n in cands if s >= 0.40][:10]

    def mark_superseded(self, old_id: str, new_id: str):
        """Preserve-style update: mark, never delete (Ch.6.3.2)."""
        with self.lock:
            self.conn.execute(
                "UPDATE nodes SET superseded_by=? WHERE node_id=?", (new_id, old_id))
            self.conn.commit()

    # ---------------------------------------------------- decay / consolidate
    def consolidate(self, decay_lambdas: dict | None = None) -> dict:
        """Exponential decay w(t) = w0 * exp(-lambda * days_idle), lambda
        tiered by memory type (Ch.6.3.1). Report-only: nothing is destroyed."""
        lambdas = {"Fact": 0.01, "Event": 0.05, "Preference": 0.005,
                   "Task": 0.03, "Knowledge": 0.02, "Entity": 0.001}
        lambdas.update(decay_lambdas or {})
        now = datetime.now(timezone.utc)
        report = {"decayed": 0, "cold_candidates": [], "hot": []}
        with self.lock:
            for r in self.conn.execute(
                    "SELECT node_id, node_type, weight, last_accessed FROM nodes"
                    " WHERE superseded_by IS NULL"):
                ref = r["last_accessed"]
                try:
                    days = (now - datetime.fromisoformat(ref)).days if ref else 30
                except ValueError:
                    days = 30
                lam = lambdas.get(r["node_type"], 0.02)
                import math
                new_w = r["weight"] * math.exp(-lam * days)
                self.conn.execute("UPDATE nodes SET weight=? WHERE node_id=?",
                                  (max(new_w, 0.01), r["node_id"]))
                if new_w < 0.1:
                    report["cold_candidates"].append(r["node_id"])
                elif new_w > 1.0:
                    report["hot"].append(r["node_id"])
                report["decayed"] += 1
            self.conn.commit()
        report["cold_candidates"] = report["cold_candidates"][:20]
        report["hot"] = report["hot"][:20]
        report["note"] = ("cold candidates are reported only; removal requires an "
                          "explicit human decision (保留式更新, Ch.6.3.2)")
        return report

    # ---------------------------------------------------- communities
    def save_communities(self, level: int, communities: list[list[str]],
                         summaries: list[str], parent_ids: list | None = None):
        with self.lock:
            self.conn.execute("DELETE FROM communities WHERE level=?", (level,))
            ids = []
            for i, (members, summary) in enumerate(zip(communities, summaries)):
                cur = self.conn.execute(
                    "INSERT INTO communities (level, parent_id, member_ids, summary,"
                    " created_at) VALUES (?,?,?,?,?)",
                    (level, parent_ids[i] if parent_ids else None,
                     json.dumps(members), summary, _now()))
                ids.append(cur.lastrowid)
            self.conn.commit()
            return ids

    def get_communities(self, level: int = 0) -> list[dict]:
        return [{"community_id": r["community_id"], "level": r["level"],
                 "member_ids": json.loads(r["member_ids"]), "summary": r["summary"]}
                for r in self.conn.execute(
                    "SELECT * FROM communities WHERE level=? ORDER BY community_id", (level,))]

    # ---------------------------------------------------- event log (audit)
    def log_event(self, event_type: str, payload: dict) -> int:
        with self.lock:
            cur = self.conn.execute(
                "INSERT INTO events (ts, event_type, payload) VALUES (?,?,?)",
                (_now(), event_type, json.dumps(payload, ensure_ascii=False, default=str)))
            self.conn.commit()
            return cur.lastrowid

    def get_events(self, after_event_id: int = 0, event_type: str = "") -> list[dict]:
        sql = "SELECT * FROM events WHERE event_id>? "
        args: list = [after_event_id]
        if event_type:
            sql += "AND event_type=? "
            args.append(event_type)
        sql += "ORDER BY event_id"
        return [{"event_id": r["event_id"], "ts": r["ts"],
                 "event_type": r["event_type"], "payload": json.loads(r["payload"])}
                for r in self.conn.execute(sql, args)]

    # ---------------------------------------------------- stats
    def stats(self) -> dict:
        n_nodes = self.conn.execute("SELECT COUNT(*) c FROM nodes").fetchone()["c"]
        n_edges = self.conn.execute("SELECT COUNT(*) c FROM edges").fetchone()["c"]
        n_events = self.conn.execute("SELECT COUNT(*) c FROM events").fetchone()["c"]
        by_type = {r["node_type"]: r["c"] for r in self.conn.execute(
            "SELECT node_type, COUNT(*) c FROM nodes GROUP BY node_type")}
        n_contra = self.conn.execute(
            "SELECT COUNT(*) c FROM edges WHERE edge_type='contradicts'").fetchone()["c"]
        return {"nodes": n_nodes, "edges": n_edges, "events": n_events,
                "nodes_by_type": by_type, "contradiction_edges": n_contra,
                "db_path": self.db_path, "embedding_backend": self.embedder.name,
                "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD}

    def close(self):
        self.conn.close()
