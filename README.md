# graph-native-mcp

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[中文说明](README.zh-CN.md)

**Four MCP servers for graph-native agents**: the core capabilities of chapters 5–8 of the book *Graph-Native Agents (Graph × LLM × Agent)*, packaged as tools that any MCP client (zcode / Claude Code / Cursor / any MCP host) can call directly.

## Why this project exists

This project was built to **plug into zcode**. zcode + GLM-5.3 is a strong coding-and-reasoning combo, but it has no graph-native capabilities natively — no cross-session memory graph, no auditable graph retrieval, no assertion-backed skill library. Mount these four MCP servers, and GLM-5.3 gains that layer:

- **GLM-5.3 does the judging**: setting goals, reading evidence, drawing conclusions, interpreting plans — the probabilistic reasoning models are good at;
- **These servers do the deterministic part**: bi-temporal memory, retrieval with citation ids, precondition/effect assertion checks, A* planning, audit replay — testable and verifiable offline.

This division is exactly the book's thesis: *retrieval finds the evidence, the model makes the call*. Once configured in zcode (see below), every memory write, retrieval, plan and execution by GLM-5.3 goes through these four graphs, with the evidence chain and token cost observable end to end.

```
zcode (GLM-5.3)
        │ MCP protocol
        ├──────────────┬──────────────┬──────────────┐
   graph-memory    graph-rag      graph-build    graph-agent
   Ch.6 memory     Ch.5 three     Ch.7 build     Ch.8 graph-native core
   bi-temporal ·   retrieval      pipeline       skill graph · A* planning ·
   conflict detect local/global/  extraction ·   world model
                   paths          NL2query · checkup
        └──────────────┴──────┬───────┴──────────────┘
                     gxt-core shared Python package
   SQLite graph store · bi-temporal schema · evidence serialization · embeddings · graph algorithms
```

The four servers **share one SQLite store and never call each other** — mount two now, add the rest later.

## Installation

```bash
pip install graph-native-mcp        # once published on PyPI
pip install -e .                    # or from source
```

Only two dependencies: `mcp` and `networkx`. No GPU, no database service, zero network by default.

## Quick start (zcode + GLM-5.3)

Copy the servers you need from [mcp.json.example](mcp.json.example) into your zcode MCP config, and GLM-5.3 gains graph-native capabilities. Minimal setup (works offline, zero API keys):

```json
{
  "mcpServers": {
    "graph-memory": {"command": "gnm", "args": ["memory"]},
    "graph-rag":    {"command": "gnm", "args": ["rag"]}
  }
}
```

A three-minute walkthrough lives in [examples/quickstart.md](examples/quickstart.md).

## The four servers, mapped to the book

| Server | Chapter | Tools | In one line |
|---|---|---|---|
| **graph-memory** | Ch. 6 Graph as long-term memory | `write_memory` `get_timeline` `consolidate` `semantic_search` `retrieve_context` `add_knowledge` `add_node/edge` `ingest_file` … (15) | Conflicts preserved, history never deleted, memory temperature observable |
| **graph-rag** | Ch. 5 Graph-augmented retrieval | `search_local` `search_global` `search_paths` `build_communities` | Three retrieval modes, read-only, no LLM, evidence with citation ids |
| **graph-build** | Ch. 7 LLM-augmented graphs | `ingest_document` `text_to_query` `graph_checkup` | Four-layer extraction with attributable errors; the only server needing an LLM key |
| **graph-agent** | Ch. 8 Graph-native agents | `register_skill` `extract_skills` `plan` `apply_action` `report_result` `set_constraint` `replay` … (11) | Skills with precondition/effect assertions, planning = A* on the skill graph, constraints enforced before search |

### graph-memory: preserve-style updates

`write_memory` records two timestamps per fact — when it *happened* vs. when the system *learned it*. A new memory that conflicts with an old one is never overwritten: a `contradicts` edge preserves both sides (merged at retrieval time or adjudicated by a human). Only an explicit correction (`explicit_correction=True`) marks the old node superseded — it stays traceable, never physically deleted. `get_timeline("Alice", as_of="2024-06-01")` answers *what did the system know back then*. `consolidate()` applies exponential time decay (λ tiered by memory type); cold memories are reported, not destroyed.

### graph-rag: retrieval finds evidence, the model makes the call

Three retrieval modes for three question types: `search_local` (who is related to whom), `search_global` (what is this corpus about), `search_paths` (how does A affect B — the path *is* the reasoning chain). Every call returns the **audit triple**: evidence text with `[node] n123` / `[edge] n123 -[involves]-> n456` citation ids + the citation closure + a cost report (nodes / edges / estimated tokens). When no path exists, it says so honestly — "no evidence on the graph" — instead of fabricating one.

### graph-build: an extraction pipeline with attributable errors

`ingest_document` runs a four-layer pipeline (candidate generation → disambiguation & linking → relation judgment → write gate) and reports per-layer statistics, so every failure has a layer to blame. `text_to_query` is constrained generation with an execution-feedback loop: generate → run → feed the error back → retry. `graph_checkup` screens five classes of graph errors (isolated nodes / hub degree anomalies / dangling refs / schema violations / low-confidence and unattributed).

### graph-agent: thinking and acting on graphs

```python
set_world_state({"inventory.apple": 5, "cash": 100})       # addressable world model
register_skill("buy_apples",
    precondition=[{"key": "cash", "op": "ge", "value": 50}],    # machine-verifiable
    effect=[{"key": "inventory.apple", "op": "add", "value": 10},
            {"key": "cash", "op": "sub", "value": 50}])
plan(goal=[{"key": "cash", "op": "ge", "value": 120}])
# → buy_apples → sell_apples; edge weight = cost / success_rate,
#   skills that keep failing get routed around automatically
set_constraint(forbidden_skills=[...], gate_skills=[...])   # red lines apply before search
apply_action(skill_id, delta={...})                         # ΔW incremental update
replay()                                                    # audit replay
```

`extract_skills` distills skills from execution trajectories automatically (effect = the diff between world states before and after). This server is fully deterministic with zero LLM calls — the client model sets goals and interprets plans; this side guarantees the verification machinery.

## Design principles

1. **One server = one exclusive state.** Memory graph / read-only index / build pipeline / skill graph & world model: four states sharing only the underlying store. Zero calls, zero dependencies between servers — mount on demand.
2. **Read/write separation + LLM isolation.** rag is read-only, LLM-free, millisecond-fast; build is the only server that needs an LLM key (extraction needs a model); agent writes at runtime but only to its own state graph. Permission granularity, failure domains and dependency weight never mix.
3. **No probabilistic computation at runtime.** Retrieval, planning and verification are deterministic graph computations — testable, verifiable offline. Generation and judgment stay with the client: that is the correct cut for the MCP boundary.
4. **Fields are the defense line.** A Fact without `source_ids` is rejected by the write gate and screened out of retrieval; every piece of evidence traces back to its source via ids.

## Relation to gx-memory

graph-memory is the successor of gx-memory: the tool interface (`add_knowledge` / `semantic_search` / `retrieve_context` / `ingest_file` …) carries over, the local GX-engine + pickle backend is replaced by portable SQLite, and the three pieces gx-memory lacked — bi-temporal timestamps, conflict-detection edges, and time decay — are added.

## Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `GNM_DB` | SQLite store path, shared by all four servers | `~/.gnm/graph.db` |
| `GNM_EMBED_BACKEND` | `hash` (offline, deterministic) / `openai` | `hash` |
| `GNM_EMBED_BASE_URL` / `_API_KEY` / `_MODEL` | OpenAI-compatible embedding endpoint | — |
| `GNM_LLM_BASE_URL` / `_API_KEY` / `_MODEL` | LLM used by graph-build (OpenAI-compatible) | — |

The default embedding backend is deterministic feature hashing: zero dependencies, zero network, but it matches surface terms rather than meaning — fine for tests and demos; switch to a real embedding backend for production.

## Tests

```bash
python tests/run_tests.py          # 55 offline assertions (zero network, zero API keys)
python examples/smoketest_mcp.py   # end-to-end over real MCP stdio, all four servers
```

## Roadmap

- [ ] Kùzu backend: real Cypher for `text_to_query` (currently a JSON DSL)
- [ ] Replace Louvain with Leiden (connectivity guarantee)
- [ ] LLM-assisted conflict judgment and community summaries (inside graph-build)
- [ ] sqlite-vec vector index (currently brute-force cosine, fine up to ~10k nodes)
- [ ] Tool registry graph: register the MCP tool declarations of all four servers as graph nodes (book §8.2.4)
- [ ] PyPI release

## License

MIT

---

## Buy me a coffee

graph-native-mcp is an open-source project I develop and maintain in my spare time, free forever.

If it gave your AI assistant a long-term memory, auditable retrieval, or skill planning — consider buying me a coffee (9.9 CNY is enough). Every bit of support goes straight into new features and bug fixes.

<p align="center">
  <img src="donate.jpg" alt="Buy me a coffee" width="280">
</p>

<p align="center"><i>Leave a note with your donation about the feature you want most - I prioritize those ;)</i></p>

**Can't donate?** Starring the repo, opening an Issue, or sharing it with someone who needs it helps just as much!
