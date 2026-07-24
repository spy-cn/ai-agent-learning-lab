# Swarm 群智协作模式

Swarm 模式是去中心化的多 Agent 架构——Agent 之间直接传递任务，无需中央调度。

---

## Swarm vs Supervisor

```
Supervisor (中心化):              Swarm (去中心化):
     ┌─────────┐
     │Supervisor│                  A → B → C → D
     └────┬────┘                  (接力传递)
    ┌─┬─┬─┘
    A B C                        每个 Agent 决定下一个
    (全部经过中心)
```

---

## Handoff 机制

Swarm 的核心是 **Handoff**——一个 Agent 完成后将任务"交接"给下一个。

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from typing import Annotated

class SwarmState(TypedDict):
    messages: Annotated[list, add_messages]
    current_agent: str

# 定义 Agent 和它可以交接给谁
def agent_a(state: SwarmState) -> dict:
    """Agent A: 接待用户"""
    from langchain_core.messages import AIMessage

    response = llm.invoke(state["messages"])

    # 决定交接给谁
    if "技术" in response.content:
        next_agent = "agent_b"  # 交给技术专家
    elif "账单" in response.content:
        next_agent = "agent_c"  # 交给财务
    else:
        next_agent = "END"

    return {
        "messages": [response],
        "current_agent": next_agent
    }

def agent_b(state: SwarmState) -> dict:
    """Agent B: 技术专家"""
    response = llm.invoke(state["messages"])
    return {"messages": [response], "current_agent": "END"}

def agent_c(state: SwarmState) -> dict:
    """Agent C: 财务"""
    response = llm.invoke(state["messages"])
    return {"messages": [response], "current_agent": "END"}

# 路由
def route_swarm(state: SwarmState) -> str:
    next_agent = state.get("current_agent", "END")
    if next_agent == "END":
        return END
    return next_agent

# 构建
builder = StateGraph(SwarmState)
builder.add_node("agent_a", agent_a)
builder.add_node("agent_b", agent_b)
builder.add_node("agent_c", agent_c)

builder.add_edge(START, "agent_a")
builder.add_conditional_edges("agent_a", route_swarm)
builder.add_conditional_edges("agent_b", route_swarm)
builder.add_conditional_edges("agent_c", route_swarm)

graph = builder.compile()
```

## 使用 langgraph-swarm 库

```bash
pip install langgraph-swarm
```

```python
from langgraph_swarm import create_swarm, create_handoff_tool

# 创建交接工具
handoff_to_b = create_handoff_tool(agent_name="agent_b")
handoff_to_c = create_handoff_tool(agent_name="agent_c")

# 创建带交接能力的 Agent
agent_a = create_react_agent(
    llm, tools=[handoff_to_b, handoff_to_c], name="agent_a"
)
agent_b = create_react_agent(llm, tools=[], name="agent_b")
agent_c = create_react_agent(llm, tools=[], name="agent_c")

# 构建 Swarm
swarm = create_swarm(
    agents=[agent_a, agent_b, agent_c],
    default_active_agent="agent_a"
).compile()
```

---

## 小结

| 特性 | Supervisor | Swarm |
|------|-----------|-------|
| 控制 | 中心化 | 去中心化 |
| 灵活性 | 中 | 高 |
| 可预测性 | 高 | 中 |
| 适合 | 结构化任务 | 探索性任务 |

---

## 下一篇

➡️ [智能体通信协议](./05-智能体通信协议.md)
