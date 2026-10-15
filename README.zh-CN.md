# graph-native-mcp

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[English](README.md)

**图原生智能体的四个 MCP server**：把《图原生智能体（Graph × LLM × Agent 技术详解）》第 5–8 章的核心能力，封装成任何 MCP 客户端（zcode / Claude Code / Cursor / 任意 MCP Host）都能直接调用的工具。

## 为什么做这个项目

这个项目是为**接入 zcode** 而开发的。zcode + GLM-5.3 是很强的编码与推理组合，但它本身没有图原生能力——没有跨会话的长期记忆图、没有可审计的图检索、没有带断言的技能库。挂上这四个 MCP server，GLM-5.3 就补上了这一层：

- **GLM-5.3 负责判断**：定目标、读证据、下结论、解计划——模型擅长的概率推理；
- **本项目的 server 负责确定性部分**：双时间戳记忆、带引用 id 的图检索、前置/效果断言校验、A* 规划、审计回放——可测试、可离线验证。

这个分工正是书里的主张：「检索管找证据，模型管下判断」。在 zcode 里配好 MCP（见下），GLM-5.3 的每一次记忆写入、检索、规划、执行都走这四张图，证据链与词元开销全程可观测。

```
zcode（GLM-5.3）
        │ MCP 协议
        ├──────────────┬──────────────┬──────────────┐
   graph-memory    graph-rag      graph-build    graph-agent
   第6章 记忆图      第5章 三种检索   第7章 构建管线   第8章 图原生核心
   双时间戳·冲突检测  局部/全局/路径   抽取·NL2查询·体检  技能图·A*规划·世界模型
        └──────────────┴──────┬───────┴──────────────┘
                        gxt-core 共享 Python 包
        SQLite 图存储 · bi-temporal schema · 证据序列化 · 嵌入接口 · 图算法
```

四个 server **共享同一个 SQLite 存储，彼此零调用**——你可以只挂两个，另外两个以后再加。

## 安装

```bash
pip install graph-native-mcp        # 发布 PyPI 后
pip install -e .                    # 或从源码
```

依赖只有 `mcp` 和 `networkx`。不需要 GPU、不需要数据库服务、默认零网络。

## 快速开始（zcode + GLM-5.3）

把 [mcp.json.example](mcp.json.example) 里你需要的 server 复制进 zcode 的 MCP 配置，GLM-5.3 即获得图原生能力。最小组合（离线可用、零 API key）：

```json
{
  "mcpServers": {
    "graph-memory": {"command": "gnm", "args": ["memory"]},
    "graph-rag":    {"command": "gnm", "args": ["rag"]}
  }
}
```

三分钟上手见 [examples/quickstart.md](examples/quickstart.md)。

## 四个 server 与书稿章节的对应

| Server | 章节 | 工具 | 一句话 |
|---|---|---|---|
| **graph-memory** | 第 6 章 图作为长期记忆 | `write_memory` `get_timeline` `consolidate` `semantic_search` `retrieve_context` `add_knowledge` `add_node/edge` `ingest_file` …（15 个） | 矛盾不覆盖、历史不删除、冷热可观测 |
| **graph-rag** | 第 5 章 图增强检索 | `search_local` `search_global` `search_paths` `build_communities` | 三种检索模式，只读无 LLM，证据带引用 id |
| **graph-build** | 第 7 章 大模型增强图谱 | `ingest_document` `text_to_query` `graph_checkup` | 四层抽取错误可归因；唯一需要 LLM key 的 server |
| **graph-agent** | 第 8 章 图原生智能体 | `register_skill` `extract_skills` `plan` `apply_action` `report_result` `set_constraint` `replay` …（11 个） | 技能带前置/效果断言，规划=图上 A*，约束在搜索前生效 |

### graph-memory：记忆是保留式更新

`write_memory` 双时间戳分开记「事实发生时间」与「系统知道的时间」；新记忆与旧记忆冲突时不覆盖，建 `contradicts` 边保留双方（检索时合并策略或人工裁决）；只有用户明确说「之前记错了」（`explicit_correction=True`）才标记 superseded——旧节点保留可回溯，从不物理删除。`get_timeline("张三", as_of="2024-06-01")` 回答「当时系统知道什么」。`consolidate()` 做指数时间衰减（按记忆类型分档 λ），冷记忆只报告不删除。

### graph-rag：检索管找证据，模型管下判断

三种检索模式对应三类问题：`search_local`（谁是谁的什么）、`search_global`（全局在讲什么）、`search_paths`（A 如何影响 B，路径即推理链）。每次检索返回**审计三件套**：带 `[节点] n123` / `[边] n123 -[involves]-> n456` 引用 id 的证据文本 + 引用闭包 + 代价报告（节点数/边数/词元估计）。路径不存在时明确返回「图上无证据」，而不是强行编造。

### graph-build：错误可归因的抽取管线

`ingest_document` 四层管线（候选生成 → 消歧与链接 → 关系判定 → 写入门控）逐层返回统计，出错能定位到层。`text_to_query` 是约束生成 + 执行反馈闭环：生成 → 试跑 → 报错回喂重试。`graph_checkup` 体检五类图错误（孤立节点 / hub 度数异常 / 断链 / schema 违规 / 低置信与无来源）。

### graph-agent：图上思考与行动

```python
set_world_state({"库存.苹果": 5, "资金": 100})          # 世界模型可寻址
register_skill("采购苹果",
    precondition=[{"key": "资金", "op": "ge", "value": 50}],   # 机器可验证
    effect=[{"key": "库存.苹果", "op": "add", "value": 10},
            {"key": "资金", "op": "sub", "value": 50}])
plan(goal=[{"key": "资金", "op": "ge", "value": 120}])
# → 采购苹果 → 卖出苹果；边权 = cost / 成功率，连续失败的技能自动被绕开
set_constraint(forbidden_skills=[...], gate_skills=[...])  # 红线在搜索前生效
apply_action(skill_id, delta={...})                        # ΔW 增量更新
replay()                                                   # 审计回放
```

`extract_skills` 从执行轨迹自动沉淀技能（效果 = 前后世界状态的 diff）。本 server 全程确定性、无 LLM 调用——客户端模型负责定目标和解计划，这里保证验证机制。

## 设计原则

1. **一个 server = 一份独占状态。** 记忆图 / 只读索引 / 构建管线 / 技能图与世界模型，四份状态共享的只有底层存储。server 之间零调用、零依赖，按需挂载。
2. **读写分离 + LLM 隔离。** rag 只读、无 LLM、毫秒级；build 是唯一需要 LLM key 的（抽取离不开模型）；agent 运行时写但只写自己的状态图。权限粒度、故障域、依赖重量不混。
3. **运行时不做概率计算。** 检索、规划、验证都是确定性图计算，可测试、可离线验证。生成与判断留给客户端——这是 MCP 边界的正确切法。
4. **字段即防线。** 无 `source_ids` 的 Fact 被写入门控直接拒绝、被检索层屏蔽；每条证据可沿 id 追溯到来源。

## 与 gx-memory 的关系

graph-memory 是 [gx-memory](https://github.com/) 的继任：工具接口（`add_knowledge` / `semantic_search` / `retrieve_context` / `ingest_file` …）延续自 gx-memory，本地 GX 引擎 + pickle 后端换成可移植的 SQLite，并补上双时间戳、冲突检测边、时间衰减三件 gx-memory 缺失的能力。

## 环境变量

| 变量 | 作用 | 默认 |
|---|---|---|
| `GNM_DB` | SQLite 存储路径，四个 server 共用 | `~/.gnm/graph.db` |
| `GNM_EMBED_BACKEND` | `hash`（离线确定性）/ `openai` | `hash` |
| `GNM_EMBED_BASE_URL` / `_API_KEY` / `_MODEL` | OpenAI 兼容嵌入端点 | — |
| `GNM_LLM_BASE_URL` / `_API_KEY` / `_MODEL` | graph-build 用的 LLM（OpenAI 兼容） | — |

默认嵌入后端是确定性特征哈希：零依赖零网络，但只匹配表面词不匹配语义——评测与演示够用，生产请换真嵌入。

## 测试

```bash
python tests/run_tests.py          # 55 项离线断言（零网络、零 API key）
python examples/smoketest_mcp.py   # 四个 server 真实 stdio 协议端到端
```

## 路线图

- [ ] Kùzu 后端：`text_to_cypher` 真 Cypher（当前是 JSON DSL）
- [ ] Leiden 算法替换 Louvain（连通性保证）
- [ ] LLM 辅助冲突判定与社区摘要（graph-build 内）
- [ ] sqlite-vec 向量索引（当前暴力余弦，万级节点内够用）
- [ ] 工具注册图：把四个 server 的 MCP 工具声明注册为图节点（书 8.2.4）
- [ ] PyPI 发布

## License

MIT

---

## 请我喝杯咖啡

graph-native-mcp 是我在业余时间开发和维护的开源项目，永久免费。

如果它帮你的 AI 助手补上了长期记忆、可审计检索或者技能规划——可以考虑请我喝杯咖啡（9.9 元就够）。每一份支持都会直接变成新功能和 bug 修复。

<p align="center">
  <img src="donate.jpg" alt="打赏支持" width="280">
</p>

<p align="center"><i>打赏时留言你最想要的功能，我会优先做 ;)</i></p>

**不方便打赏？** 点个 Star、提个 Issue、或者分享给需要的人，同样是很好的支持！
