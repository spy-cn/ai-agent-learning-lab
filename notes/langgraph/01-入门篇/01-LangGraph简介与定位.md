# LangGraph 简介与定位

## 什么是 LangGraph

LangGraph 是 LangChain 团队推出的**基于图的状态机框架**，专为构建可靠、可控的大语言模型（LLM）智能体应用而设计。

### 一句话定义

> LangGraph = **有状态的、可控循环的、可流式输出的** LLM 应用编排框架

### 核心定位

```
传统 LLM 应用的问题                LangGraph 的解决思路
┌──────────────────────┐         ┌──────────────────────────┐
│ • 单次调用，无记忆     │         │ • 状态持久化，跨轮次记忆    │
│ • 线性链式，无法循环   │   →     │ • 图结构，支持循环与分支    │
│ • 不可控的黑盒        │         │ • 可中断、可审批、可修正    │
│ • 难以调试与观测      │         │ • 与 LangSmith 深度集成     │
│ • 无法多智能体协作    │         │ • 原生支持多智能体架构      │
└──────────────────────┘         └──────────────────────────┘
```

---

## 为什么需要 LangGraph

### 1. 从「链」到「图」的范式跃迁

传统 LangChain 的 Chain 是**有向无环图（DAG）**——数据只能从上一步流到下一步，不能回头：

```
[输入] → [Prompt] → [LLM] → [解析] → [输出]
```

但真实的智能体应用需要**循环**：
- 工具调用后根据结果继续推理
- 答案不满意时重新检索
- 多轮对话中反复迭代

LangGraph 将应用建模为**有向有环图**：

```
       ┌─────────────────────┐
       │                     │
       ▼                     │
  ┌─────────┐         ┌──────────┐
  │  Agent   │ ──────► │ 工具调用  │
  │  推理    │ ◄────── │ 结果返回  │
  └─────────┘         └──────────┘
       │
       ▼
  ┌─────────┐
  │ 最终输出 │
  └─────────┘
```

### 2. 可控性（Controllability）

| 特性 | LangChain Chain | LangGraph |
|------|----------------|-----------|
| 执行流程 | 固定线性 | 可编程路由 |
| 中断能力 | ❌ | ✅ interrupt |
| 循环控制 | ❌ | ✅ recursion_limit |
| 状态持久化 | ❌ | ✅ Checkpoint |
| 人机回路 | ❌ | ✅ Human-in-the-Loop |

### 3. 流式与可观测

```python
# 流式输出每一个 token
for event in graph.stream(input, stream_mode="updates"):
    print(event)

# 在 LangSmith 中追踪每一步执行
# 包括：节点调用、状态变化、LLM请求/响应、工具调用
```

---

## LangGraph vs 其他框架

### 与 LangChain 的关系

```
LangChain（生态）
├── langchain-core    # 核心抽象（Runnable, Message等）
├── langchain         # 链式集成
├── LangGraph         # ← 图式编排（本教程）
├── LangSmith         # 追踪、评估、监控
└── LangServe         # API部署
```

**关键区别**：

| 维度 | LangChain Chain | LangGraph |
|------|----------------|-----------|
| 拓扑结构 | 线性链 / DAG | 有环有向图 |
| 状态管理 | 隐式传递 | 显式状态对象 |
| 循环 | 不支持 | 原生支持 |
| 适用场景 | 简单流程 | 复杂智能体 |
| 学习曲线 | 低 | 中等 |
| 生产成熟度 | 一般 | 高 |

### 与 CrewAI / AutoGen 对比

| 框架 | 设计哲学 | 控制粒度 | 状态管理 | 适合场景 |
|------|----------|----------|----------|----------|
| **LangGraph** | 图结构、显式控制 | ⭐⭐⭐⭐⭐ | 强类型+持久化 | 生产级可控Agent |
| CrewAI | 角色扮演、任务驱动 | ⭐⭐⭐ | 弱 | 快速原型多Agent |
| AutoGen | 对话驱动、自发协作 | ⭐⭐ | 弱 | 研究探索多Agent |
| LlamaIndex | 数据驱动、RAG优先 | ⭐⭐⭐ | 中等 | 检索增强应用 |

**选择建议**：
- 需要**精确控制**执行流程 → LangGraph
- 需要**快速搭建**多Agent原型 → CrewAI
- 需要**最强RAG**能力 → LlamaIndex + LangGraph

---

## 核心设计理念

### 1. 状态驱动（State-Driven）

所有数据通过一个**中心化的 State 对象**在节点间流转：

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # 对话历史
    retrieved_docs: list                      # 检索到的文档
    current_step: str                         # 当前步骤
    tool_results: dict                        # 工具结果
```

### 2. 图即代码（Graph as Code）

图的拓扑结构用 Python 代码显式定义，所见即所得：

```python
graph = StateGraph(AgentState)
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)
graph.add_edge(START, "retrieve")
graph.add_conditional_edges("retrieve", route_function)
graph.add_edge("generate", END)
```

### 3. 可组合与可复用

子图可以像函数一样被组合：

```python
# RAG 子图
rag_subgraph = build_rag_graph()

# 代码生成子图
code_subgraph = build_code_graph()

# 主图组合两者
main_graph.add_node("rag", rag_subgraph)
main_graph.add_node("code", code_subgraph)
```

---

## 典型应用场景

| 场景 | 说明 | 复杂度 |
|------|------|--------|
| **多轮对话机器人** | 有记忆、有上下文的聊天 | ⭐⭐ |
| **工具调用Agent** | LLM自主决定调用哪个工具 | ⭐⭐⭐ |
| **RAG系统** | 检索增强生成，支持自适应路由 | ⭐⭐⭐ |
| **多智能体协作** | 多个Agent分工合作完成任务 | ⭐⭐⭐⭐ |
| **代码生成与执行** | 生成代码→执行→根据结果修正 | ⭐⭐⭐⭐ |
| **数据分析Agent** | 自动查询数据库、生成图表 | ⭐⭐⭐⭐ |
| **自主研究Agent** | 搜索→阅读→总结→写作 | ⭐⭐⭐⭐⭐ |
| **Human-in-the-Loop** | 需要人工审批的工作流 | ⭐⭐⭐ |

---

## 生态系统全景

```
                    LangGraph 生态系统
                          │
           ┌──────────────┼──────────────┐
           │              │              │
     开发与调试        部署与运行      监控与优化
           │              │              │
     ┌─────┴─────┐   ┌───┴───┐    ┌────┴────┐
     │           │   │       │    │         │
  LangGraph  LangSmith  LangGraph  LangSmith  性能
   Studio    追踪调试   Cloud/     评估      优化
                        自部署
```

### 关键组件

1. **LangGraph 库**：核心 Python/JS 库，定义图、节点、边
2. **LangGraph Studio**：可视化 IDE，实时查看图结构与状态
3. **LangGraph Cloud/Platform**：托管部署平台（或自托管）
4. **LangSmith**：追踪、评估、监控、数据集管理
5. **Checkpoint 库**：状态持久化（内存/Redis/PostgreSQL）

---

## 小结

| 要点 | 说明 |
|------|------|
| LangGraph 是什么 | 基于图的、有状态的 LLM 应用编排框架 |
| 解决什么问题 | 复杂Agent的循环、分支、状态管理、人机回路 |
| 核心价值 | 可控性 + 可观测性 + 生产就绪 |
| 与LangChain关系 | LangChain生态的一部分，专注于复杂Agent编排 |
| 适合谁 | 需要构建可控、生产级LLM应用的开发者 |

---

## 下一篇

➡️ [环境搭建与安装](./02-环境搭建与安装.md)
