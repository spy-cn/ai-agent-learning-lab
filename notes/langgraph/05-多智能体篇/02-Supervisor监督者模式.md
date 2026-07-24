# Supervisor 监督者模式

Supervisor 是最实用的多智能体架构——一个中心调度器协调多个专业 Agent。

---

## Supervisor 的核心思想

```
用户请求 → Supervisor 分析 → 分配给最合适的 Agent
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
               ┌────────┐     ┌────────┐     ┌────────┐
               │ Agent A │     │ Agent B │     │ Agent C │
               │ (研究)  │     │ (编码)  │     │ (分析)  │
               └────┬───┘     └────┬───┘     └────┬───┘
                    │               │               │
                    └───────────────┼───────────────┘
                                    │
                                    ▼
                              Supervisor
                              汇总结果
                                    │
                                    ▼
                                 返回用户
```

---

## 实现 Supervisor

### 状态定义

```python
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
import operator

class TeamState(TypedDict):
    """团队共享状态"""
    messages: Annotated[list, add_messages]
    team_members: list[str]           # 可用 Agent 列表
    next_agent: str                    # Supervisor 决定的下一个 Agent
    task: str                          # 当前任务
    context: Annotated[list[str], operator.add]  # 累积的上下文
```

### Supervisor 节点

```python
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

class RouteDecision(BaseModel):
    """路由决策"""
    next: Literal["researcher", "coder", "analyst", "FINISH"]

def supervisor(state: TeamState) -> dict:
    """Supervisor: 分析任务，决定下一个执行的 Agent"""
    system_prompt = f"""
    你是团队监督者，负责分配任务给合适的团队成员。

    可用成员:
    - researcher: 信息搜索、资料收集、事实核查
    - coder: 编写代码、调试、技术实现
    - analyst: 数据分析、统计分析、可视化

    根据当前任务和已有结果，决定下一步分配给谁。
    如果任务已完成，返回 "FINISH"。

    当前任务: {state['task']}
    已有上下文: {state.get('context', [])}
    """

    # 用结构化输出做决策
    decision_llm = llm.with_structured_output(RouteDecision)
    decision = decision_llm.invoke([
        ("system", system_prompt),
        ("user", str(state["messages"][-1].content))
    ])

    return {"next_agent": decision.next}
```

### 路由函数

```python
def route_from_supervisor(state: TeamState) -> str:
    """根据 Supervisor 的决策路由"""
    next_agent = state["next_agent"]
    if next_agent == "FINISH":
        return END
    return next_agent
```

### 专业 Agent 节点

```python
def researcher(state: TeamState) -> dict:
    """研究 Agent"""
    response = llm.invoke(
        f"你是研究专家。任务: {state['task']}\n"
        f"已有信息: {state.get('context', [])}\n"
        f"请搜索相关信息并提供研究结果。"
    )
    return {
        "messages": [response],
        "context": [f"[研究] {response.content}"]  # 追加到上下文
    }

def coder(state: TeamState) -> dict:
    """编码 Agent"""
    response = llm.invoke(
        f"你是编程专家。任务: {state['task']}\n"
        f"已有信息: {state.get('context', [])}\n"
        f"请编写代码解决方案。"
    )
    return {
        "messages": [response],
        "context": [f"[代码] {response.content}"]
    }

def analyst(state: TeamState) -> dict:
    """分析 Agent"""
    response = llm.invoke(
        f"你是数据分析专家。任务: {state['task']}\n"
        f"已有信息: {state.get('context', [])}\n"
        f"请分析数据并给出结论。"
    )
    return {
        "messages": [response],
        "context": [f"[分析] {response.content}"]
    }
```

### 构建图

```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(TeamState)

# 添加所有节点
builder.add_node("supervisor", supervisor)
builder.add_node("researcher", researcher)
builder.add_node("coder", coder)
builder.add_node("analyst", analyst)

# 入口 → Supervisor
builder.add_edge(START, "supervisor")

# Supervisor → 根据决策路由
builder.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    {
        "researcher": "researcher",
        "coder": "coder",
        "analyst": "analyst",
        END: END,
    }
)

# 每个 Agent 完成后回到 Supervisor
builder.add_edge("researcher", "supervisor")
builder.add_edge("coder", "supervisor")
builder.add_edge("analyst", "supervisor")

graph = builder.compile()
```

### 图结构

```
                ┌──► researcher ──┐
                │                  │
START → supervisor ──┼──► coder ──────┼──► supervisor → ... → END
                │                  │
                └──► analyst ─────┘
                  (循环直到 FINISH)
```

### 运行

```python
result = graph.invoke({
    "messages": [("user", "研究AI最新趋势，写一个数据可视化脚本，分析市场规模")],
    "team_members": ["researcher", "coder", "analyst"],
    "task": "AI市场综合分析",
    "context": [],
    "next_agent": ""
})
```

---

## Supervisor 的执行流程

```
1. 用户请求 → Supervisor 分析
2. Supervisor: "需要先研究" → 分配给 researcher
3. researcher 返回研究结果 → 回到 Supervisor
4. Supervisor: "需要编码" → 分配给 coder
5. coder 返回代码 → 回到 Supervisor
6. Supervisor: "需要分析" → 分配给 analyst
7. analyst 返回分析 → 回到 Supervisor
8. Supervisor: "任务完成" → FINISH
```

---

## 使用预构建的 Supervisor

LangGraph 提供了 `langgraph-supervisor` 库：

```bash
pip install langgraph-supervisor
```

```python
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent

# 创建专业 Agent
research_agent = create_react_agent(
    llm,
    tools=[search_tool],
    name="researcher",
    prompt="你是研究专家，擅长搜索信息。"
)

code_agent = create_react_agent(
    llm,
    tools=[python_tool],
    name="coder",
    prompt="你是编程专家，擅长写代码。"
)

# 创建 Supervisor 团队
team = create_supervisor(
    llm=llm,
    agents=[research_agent, code_agent],
    prompt="你是团队监督者，负责分配任务。"
).compile()

# 运行
result = team.invoke({
    "messages": [("user", "研究LangGraph并写一个示例代码")]
})
```

---

## Supervisor 的最佳实践

### 1. 合理划分 Agent 职责

```python
# ✅ 好：职责清晰不重叠
researcher = "负责信息检索"
coder = "负责代码实现"
reviewer = "负责质量审核"

# ❌ 不好：职责重叠
agent_a = "负责搜索和编码"   # 太多职责
agent_b = "负责编码和搜索"   # 与 A 重叠
```

### 2. 控制循环次数

```python
def supervisor_with_limit(state: TeamState) -> dict:
    """带步数限制的 Supervisor"""
    step = len(state.get("context", []))

    if step >= 10:
        return {"next_agent": "FINISH"}  # 强制结束

    # 正常决策...
```

### 3. 提供清晰的上下文

```python
def supervisor(state: TeamState) -> dict:
    context_summary = "\n".join(state.get("context", []))

    prompt = f"""
    任务: {state['task']}

    已完成的步骤:
    {context_summary}

    决定下一步分配给谁...
    """
```

### 4. 成本控制

```python
# Supervisor 本身也消耗 token
# 如果 Agent 数量多，Supervisor 的决策成本会累积
# 建议:
# - Agent 数量控制在 3-7 个
# - Supervisor 用较小的模型（如 gpt-4o-mini）
```

---

## 小结

| 要点 | 说明 |
|------|------|
| Supervisor | 中心调度，分配任务 |
| 路由决策 | LLM 结构化输出 |
| 循环 | Agent → Supervisor → Agent |
| 共享状态 | 所有 Agent 通过 State 通信 |
| FINISH | Supervisor 判断任务完成 |
| 预构建库 | langgraph-supervisor |

---

## 下一篇

➡️ [Hierarchical层级模式](./03-Hierarchical层级模式.md)
