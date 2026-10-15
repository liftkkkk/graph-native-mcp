# Quickstart

## 1. 安装

```bash
pip install graph-native-mcp          # from PyPI (once published)
# 或从源码:
pip install -e .
```

依赖只有 `mcp` 和 `networkx`。不需要 GPU、不需要数据库服务、默认零网络。

## 2. 挂载到 zcode（最小配置：memory + rag，离线可用）

本项目为接入 zcode 而开发：把 `mcp.json.example` 里你需要的 server 复制进 zcode 的 MCP 配置，GLM-5.3 即补上图原生能力（长期记忆图、可审计图检索、技能图规划）。最小组合：

```json
{
  "mcpServers": {
    "graph-memory": {"command": "gnm", "args": ["memory"]},
    "graph-rag":    {"command": "gnm", "args": ["rag"]}
  }
}
```

四个 server 通过同一个 `GNM_DB`（默认 `~/.gnm/graph.db`）共享数据，彼此零调用——你可以只挂两个，另外两个以后再加。

## 3. 三分钟走一遍四层能力

### 记忆层（Ch.6）

对 graph-memory 说：

> 帮我记住：张三 2024 年 1 月起担任 Acme 公司 CEO（来源：acme 公告）

再故意制造一次矛盾：

> 帮我记住：张三 2025 年起担任 Acme 公司 CTO（来源：另一份公告）

`write_memory` 不会覆盖旧记忆，而是建一条 `contradicts` 边保留冲突。
然后问「2024 年 6 月时系统知道张三的职务是什么」——`get_timeline`
按双时间戳回答「当时知道什么」。

### 检索层（Ch.5）

```python
# graph-rag: search_local
search_local("张三在 Acme 公司担任什么职务")
# 返回带 [节点] n123 / [边] n123 -[involves]-> n456 引用 id 的证据文本 + 代价报告

build_communities()      # 离线建社区索引（Louvain）
search_global("这批材料主要讲什么")   # 模式二：全局型问题
search_paths(source_entity="张三", target_entity="红杉资本")  # 模式三：多跳推理链
```

### 构建层（Ch.7，需要 LLM key）

设置 `GNM_LLM_BASE_URL / GNM_LLM_API_KEY / GNM_LLM_MODEL` 后：

```python
ingest_document(text=open("examples/data/acme_announcement.md").read(),
                source_id="acme-announcement")
# 四层管线逐层返回统计：候选 → 消歧 → 关系 → 门控，错误按层归因

graph_checkup()   # 五类图错误体检 + 无来源节点审计
```

### 智能体层（Ch.8，确定性，无 LLM）

```python
set_world_state({"库存.苹果": 5, "资金": 100})

register_skill(name="采购苹果",
    precondition=[{"key": "资金", "op": "ge", "value": 50}],
    effect=[{"key": "库存.苹果", "op": "add", "value": 10},
            {"key": "资金", "op": "sub", "value": 50}])

register_skill(name="卖出苹果",
    precondition=[{"key": "库存.苹果", "op": "ge", "value": 10}],
    effect=[{"key": "库存.苹果", "op": "sub", "value": 10},
            {"key": "资金", "op": "add", "value": 80}])

plan(goal=[{"key": "资金", "op": "ge", "value": 120}])
# → 计划: 采购苹果 → 卖出苹果, 总代价 2.0, 附每步的世界状态预测

set_constraint(forbidden_skills=["<skill_id>"])   # 红线在搜索前生效
apply_action(skill_id="<skill_id>", delta={"资金": {"op": "sub", "value": 50}})
replay()   # ΔW 序列 = 完整审计日志
```

轨迹自动沉淀技能（`examples/data/trajectory.json`）：

```python
extract_skills(trajectory=json.load(open("examples/data/trajectory.json")))
```

## 4. 跑测试

```bash
python tests/run_tests.py   # 40+ 项离线断言，零依赖零网络
```

## 5. 换上真嵌入（可选）

默认嵌入后端是确定性 hash（离线可用，语义有限）。要真语义检索：

```json
"env": {
  "GNM_EMBED_BACKEND": "openai",
  "GNM_EMBED_BASE_URL": "http://localhost:1234/v1",
  "GNM_EMBED_API_KEY": "any",
  "GNM_EMBED_MODEL": "text-embedding-3-small"
}
```

任何 OpenAI 兼容端点都行（LM Studio / vLLM / Ollama 的 /v1）。
