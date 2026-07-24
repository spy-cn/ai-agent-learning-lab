# StateGraph 图结构

StateGraph 是 LangGraph 的核心抽象——所有应用都以图的形式组织。本篇深入讲解图的创建、配置、编译全流程。

---

## 创建 StateGraph

### 基本语法

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages

# 1. 定义状态类型
class State(TypedDict):
    messages: Annotated[list, add_messages]
    step_count: int

# 2. 创建图构建器
graph_builder = StateGraph(State)
```

### START 和 END

LangGraph 使用两个特殊节点标记入口和出口：

```python
from langgraph.graph import START, END

# START: 虚拟入口节点，图的起点
# END:   虚拟出口节点，图的终点

graph.add_edge(START, "first_node")  # 入口 → 第一个节点
graph.add_edge("last_node", END)     # 最后节点 → 出口
```

---

## 图的组成部分

```
        ┌───────┐
        │ START │ ← 虚拟入口
        └───┬───┘
            │
            ▼
     ┌────────────┐
     │   Node A   │ ← 真实节点
     └─────┬──────┘
           │ │
     ┌─────┘ └─────┐
     │             │  ← 条件边
     ▼             ▼
┌─────────┐  ┌─────────┐
│ Node B  │  │ Node C  │
└────┬────┘  └────┬────┘
     │             │
     └──────┬──────┘
            │
            ▼
     ┌────────────┐
     │   Node D   │
     └─────┬──────┘
           │
           ▼
        ┌───┐
        │END│ ← 虚拟出口
        └───┘
```

---

## 添加节点

### 方式一：函数节点

```python
def my_node(state: State) -> dict:
    # 读取状态
    messages = state["messages"]
    # 执行逻辑
    response = llm.invoke(messages)
    # 返回更新
    return {"messages": [response], "step_count": state.get("step_count", 0) + 1}

graph_builder.add_node("my_node", my_node)
```

### 方式二：Runnable 节点（Chain/LLM）

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

# 直接将 LLM 作为节点
# LangGraph 会自动适配其输入输出
graph_builder.add_node("llm", llm)
```

### 方式三：完整子图作为节点

```python
# 先构建并编译子图
subgraph_builder = StateGraph(SubState)
subgraph_builder.add_node("a", node_a)
subgraph_builder.add_node("b", node_b)
subgraph_builder.add_edge(START, "a")
subgraph_builder.add_edge("a", "b")
subgraph_builder.add_edge("b", END)
subgraph = subgraph_builder.compile()

# 将子图作为主图的节点
main_builder.add_node("sub_process", subgraph)
```

### 节点命名规则

- 必须是字符串
- 不能与 `START`、`END`、`"__root__"` 冲突
- 建议用小写+下划线：`"retrieve_docs"`, `"generate_answer"`

---

## 添加边

### 普通边

```python
# A 执行完后必然执行 B
graph_builder.add_edge("node_a", "node_b")
```

### 入口边

```python
# 图启动时第一个执行的节点
graph_builder.add_edge(START, "entry_node")
```

### 出口边

```python
# 执行完后结束
graph_builder.add_edge("final_node", END)
```

### 多条出边（并行扇出）

```python
# B 和 C 会并行执行
graph_builder.add_edge("node_a", "node_b")
graph_builder.add_edge("node_a", "node_c")
```

图解：

```
         ┌──► [Node B] ──┐
[Node A] ─┤               ├──► [Node D]
         └──► [Node C] ──┘
```

---

## 条件边

条件边是 LangGraph 最强大的特性之一——允许根据当前状态**动态决定下一个节点**。

### 基本用法

```python
def route_based_on_intent(state: State) -> str:
    """根据用户意图路由"""
    last_message = state["messages"][-1].content.lower()

    if "天气" in last_message:
        return "weather_agent"
    elif "代码" in last_message:
        return "code_agent"
    else:
        return "general_chat"

graph_builder.add_conditional_edges(
    "router",                      # 源节点
    route_based_on_intent,         # 路由函数
    {
        "weather_agent": "weather_node",
        "code_agent": "code_node",
        "general_chat": "chat_node",
    }
)
```

### 条件边图解

```
                     ┌──► "weather"   → [Weather Node]
[User Input] → [Router] ──┼──► "code"      → [Code Node]
                     └──► "chat"       → [Chat Node]
```

### 条件边 + 并行

```python
def route(state: State) -> list[str]:
    """返回列表表示并行执行多个节点"""
    return ["node_b", "node_c"]  # 同时执行 B 和 C

graph_builder.add_conditional_edges("node_a", route)
```

---

## 图的属性查看

```python
# 查看图中所有节点
print(graph.nodes)
# {'node_a': ..., 'node_b': ...}

# 查看边的结构
print(graph.edges)

# 获取图的 Mermaid 可视化
print(graph.get_graph().draw_mermaid())

# 输出示例:
# ```mermaid
# graph TD
#     __start__([__start__]):::first
#     node_a(node_a)
#     node_b(node_b)
#     __end__([__end__]):::last
#     __start__ --> node_a
#     node_a --> node_b
#     node_b --> __end__
# ```
```

---

## 常见图结构模式

### 1. 线性流程

```python
graph.add_edge(START, "step_1")
graph.add_edge("step_1", "step_2")
graph.add_edge("step_2", "step_3")
graph.add_edge("step_3", END)
```

```
START → step_1 → step_2 → step_3 → END
```

### 2. 分支流程

```python
graph.add_edge(START, "classify")
graph.add_conditional_edges("classify", classify_fn, {
    "category_a": "process_a",
    "category_b": "process_b",
})
graph.add_edge("process_a", END)
graph.add_edge("process_b", END)
```

```
              ┌──► process_a → END
START → classify
              └──► process_b → END
```

### 3. 循环流程（Agent）

```python
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {
    "tools": "tools",
    END: END
})
graph.add_edge("tools", "agent")  # 工具执行后回到 agent
```

```
START → agent ←──── tools
            │         ↑
            └─────────┘
              (循环)
```

### 4. Map-Reduce 流程

```python
from langgraph.constants import Send

def fan_out(state: State) -> list[Send]:
    """将任务分发给多个并行节点"""
    return [
        Send("process", {"item": item})
        for item in state["items"]
    ]

graph.add_conditional_edges("split", fan_out)
graph.add_edge("process", "aggregate")
```

```
            ┌──► process(item_1) ──┐
split ──────┼──► process(item_2) ──┼──► aggregate → END
            └──► process(item_3) ──┘
```

---

## 编译选项

`compile()` 方法接受多个参数，控制图的行为：

```python
graph = graph_builder.compile(
    # 状态持久化
    checkpointer=MemorySaver(),

    # 中断点（在指定节点前/后暂停）
    interrupt_before=["human_review"],
    interrupt_after=["draft_generation"],

    # 最大循环步数（默认25）
    # recursion_limit 在 invoke 时设置
)
```

### interrupt_before / interrupt_after

```python
graph = builder.compile(
    interrupt_before=["approval_node"]
)

# 执行到 approval_node 前会暂停
result = graph.invoke(input, config)

# 此时可以检查状态
current_state = graph.get_state(config)
print(current_state.values)

# 手动恢复执行
result = graph.invoke(None, config)  # 传 None 表示继续
```

---

## 完整示例：多层路由图

```python
"""
多层路由的复杂图示例
"""
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]
    intent: str          # 识别出的意图
    sub_intent: str      # 子意图
    response: str        # 最终回复


def intent_classifier(state: State):
    """识别用户意图"""
    msg = state["messages"][-1].content
    if "买" in msg or "推荐" in msg:
        intent = "shopping"
    elif "天气" in msg:
        intent = "weather"
    else:
        intent = "chat"
    return {"intent": intent}


def route_intent(state: State) -> str:
    return state["intent"]


def shopping_agent(state: State):
    return {"response": f"推荐商品: {state['messages'][-1].content}"}

def weather_agent(state: State):
    return {"response": "今天天气晴朗，25度"}

def chat_agent(state: State):
    return {"response": "好的，我来陪你聊天"}


# 构建图
builder = StateGraph(State)
builder.add_node("classify", intent_classifier)
builder.add_node("shopping", shopping_agent)
builder.add_node("weather", weather_agent)
builder.add_node("chat", chat_agent)

builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", route_intent, {
    "shopping": "shopping",
    "weather": "weather",
    "chat": "chat",
})
builder.add_edge("shopping", END)
builder.add_edge("weather", END)
builder.add_edge("chat", END)

graph = builder.compile()

# 运行
result = graph.invoke({"messages": [("user", "推荐一台笔记本电脑")]})
print(f"意图: {result['intent']}")
print(f"回复: {result['response']}")
```

### 图结构

```
                ┌──► shopping → END
START → classify ──┼──► weather  → END
                └──► chat     → END
```

---

## 小结

| 要点 | 说明 |
|------|------|
| 创建图 | `StateGraph(State)` |
| 添加节点 | `add_node(name, function)` |
| 添加普通边 | `add_edge(source, target)` |
| 添加条件边 | `add_conditional_edges(source, router_fn, mapping)` |
| 并行执行 | 一个节点有多条出边 |
| 编译 | `compile(checkpointer=..., interrupt_before=...)` |
| 查看 | `graph.nodes`, `graph.get_graph().draw_mermaid()` |

---

## 下一篇

➡️ [Node节点详解](./02-Node节点详解.md)
