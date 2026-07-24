# Edge 边与路由

边（Edge）定义了图的**控制流**——节点之间的执行顺序和路由规则。LangGraph 提供了强大的边系统，支持固定路由、条件路由、并行扇出等多种模式。

---

## 边的三种类型

```
┌─────────────────────────────────────────────────────┐
│                Edge 类型总览                          │
├─────────────┬───────────────────────────────────────┤
│  普通边      │ 固定连接：A → B                       │
│  (Edge)     │ A 执行完必然执行 B                     │
├─────────────┼───────────────────────────────────────┤
│  条件边      │ 动态路由：A → f(state)                │
│  (CondEdge) │ 根据 state 决定下一个节点              │
├─────────────┼───────────────────────────────────────┤
│  入口/出口   │ START → first_node / last → END       │
│  (Entry)    │ 特殊的虚拟边                           │
└─────────────┴───────────────────���───────────────────┘
```

---

## 普通边（Normal Edge）

```python
# 语法
graph.add_edge(source, target)

# 示例
graph.add_edge("node_a", "node_b")
```

语义：`node_a` 执行完后，**必然**执行 `node_b`。

```
[node_a] ──────► [node_b]
```

### 多条普通边（并行）

```python
# 一个节点可以有多条出边
graph.add_edge("node_a", "node_b")
graph.add_edge("node_a", "node_c")
```

语义：`node_a` 执行完后，`node_b` 和 `node_c` **同时**执行（并行）。

```
              ┌──► [node_b]
[node_a] ─────┤
              └──► [node_c]
```

### 多条入边（汇合）

```python
# 多个节点指向同一个节点
graph.add_edge("node_b", "node_d")
graph.add_edge("node_c", "node_d")
```

语义：`node_d` 会等待 `node_b` 和 `node_c` **都完成后**再执行（扇入）。

```
[node_b] ─────┐
              ├──► [node_d]
[node_c] ─────┘
```

---

## 条件边（Conditional Edge）

条件边是 LangGraph 最强大的控制流特性。

### 基本语法

```python
graph.add_conditional_edges(
    source,                # 源节点
    router_function,       # 路由函数
    path_map               # 路径映射（可选）
)
```

### 路由函数

```python
def router(state: State) -> str:
    """
    路由函数
    - 接收 state 作为参数
    - 返回字符串：下一个节点的名称
    """
    if state["needs_search"]:
        return "search"
    elif state["needs_code"]:
        return "code"
    else:
        return "end"
```

### path_map 映射

```python
graph.add_conditional_edges(
    "router_node",
    router,
    {
        "search": "search_node",     # 返回 "search" → 走 search_node
        "code": "code_node",         # 返回 "code" → 走 code_node
        "end": END                   # 返回 "end" → 结束
    }
)
```

### 完整示例

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class State(TypedDict):
    query: str
    intent: str
    response: str

def classify(state: State) -> dict:
    query = state["query"]
    if "天气" in query:
        intent = "weather"
    elif "新闻" in query:
        intent = "news"
    else:
        intent = "general"
    return {"intent": intent}

def route(state: State) -> str:
    return state["intent"]

def weather_node(state: State) -> dict:
    return {"response": "今天晴天"}

def news_node(state: State) -> dict:
    return {"response": "今日新闻..."}

def general_node(state: State) -> dict:
    return {"response": "我能帮你什么？"}

builder = StateGraph(State)
builder.add_node("classify", classify)
builder.add_node("weather", weather_node)
builder.add_node("news", news_node)
builder.add_node("general", general_node)

builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", route, {
    "weather": "weather",
    "news": "news",
    "general": "general",
})
builder.add_edge("weather", END)
builder.add_edge("news", END)
builder.add_edge("general", END)

graph = builder.compile()
```

图结构：

```
                ┌──► weather ──► END
START → classify ──┼──► news    ──► END
                └──► general  ──► END
```

---

## 条件边返回列表（并行路由）

路由函数可以返回**列表**，表示并行执行多个节点：

```python
def parallel_router(state: State) -> list[str]:
    """返回多个目标节点"""
    targets = []
    if state.get("need_search"):
        targets.append("search")
    if state.get("need_translate"):
        targets.append("translate")
    if state.get("need_summarize"):
        targets.append("summarize")
    return targets if targets else [END]
```

图解：

```
                 ┌──► search     ──┐
[classifier] ────┼──► translate  ──┼──► [aggregate]
                 └──► summarize  ──┘
```

---

## Send API：动态扇出

当需要**为每个并行任务传递不同参数**时，使用 `Send`：

```python
from langgraph.constants import Send

def distribute_tasks(state: State) -> list[Send]:
    """为每个子任务创建一个并行执行"""
    return [
        Send("worker", {"task_id": i, "task": task})
        for i, task in enumerate(state["tasks"])
    ]

builder.add_conditional_edges("splitter", distribute_tasks)
```

### Send vs 普通并行

```
普通并行: 所有目标节点共享同一个 state
    A → [B, C, D]  (B、C、D 读取相同的 state)

Send: 每个目标节点接收独立的输入
    A → Send(B, input_1), Send(C, input_2), Send(D, input_3)
    （每个节点接收不同的参数）
```

### 完整 Map-Reduce 示例

```python
from typing import Annotated
from typing_extensions import TypedDict
import operator

class State(TypedDict):
    topics: list[str]
    summaries: Annotated[list[str], operator.add]  # 追加合并

class WorkerState(TypedDict):
    topic: str

def split(state: State):
    """分发任务"""
    return [Send("summarize", {"topic": t}) for t in state["topics"]]

def summarize(worker_state: WorkerState) -> dict:
    """单个任务处理"""
    topic = worker_state["topic"]
    summary = f"{topic}的摘要内容"
    return {"summaries": [summary]}

def aggregate(state: State) -> dict:
    """聚合结果"""
    return {"final_report": "\n".join(state["summaries"])}

builder = StateGraph(State)
builder.add_node("summarize", summarize)
builder.add_node("aggregate", aggregate)

builder.add_conditional_edges(START, split)
builder.add_edge("summarize", "aggregate")
builder.add_edge("aggregate", END)

graph = builder.compile()

result = graph.invoke({"topics": ["AI", "区块链", "量子计算"]})
print(result["final_report"])
```

---

## 路由策略详解

### 策略一：基于意图

```python
def route_by_intent(state: State) -> str:
    last_msg = state["messages"][-1].content.lower()
    if any(kw in last_msg for kw in ["买", "推荐", "价格"]):
        return "shopping"
    elif any(kw in last_msg for kw in ["天气", "温度"]):
        return "weather"
    elif any(kw in last_msg for kw in ["帮助", "怎么"]):
        return "help"
    return "chat"
```

### 策略二：基于 LLM 决策

```python
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

class RouteDecision(BaseModel):
    destination: str

def llm_router(state: State) -> str:
    structured_llm = ChatOpenAI(model="gpt-4o").with_structured_output(RouteDecision)
    decision = structured_llm.invoke(
        f"分析用户意图，选择处理节点。用户消息: {state['messages'][-1].content}\n"
        f"可选节点: search, calculate, chat"
    )
    return decision.destination
```

### 策略三：基于状态标志

```python
def route_by_flag(state: State) -> str:
    if not state.get("retrieved_docs"):
        return "retrieve"       # 还没检索，先检索
    elif state.get("need_review"):
        return "review"         # 需要审查
    elif state.get("draft_ready"):
        return "generate"       # 可以生成了
    else:
        return END
```

### 策略四：基于评分

```python
def route_by_score(state: State) -> str:
    confidence = state.get("confidence", 0)
    if confidence > 0.8:
        return "output"         # 高置信度，直接输出
    elif confidence > 0.5:
        return "refine"         # 中等，需要精炼
    else:
        return "fallback"       # 低置信度，回退
```

---

## 边的执行顺序

当一个节点有多条出边时，LangGraph 的执行顺序：

```
[node_a] 执行完成
    │
    ├── 边1: add_edge("node_a", "node_b")  ───► node_b 加入队列
    │
    └── 边2: add_edge("node_a", "node_c")  ───► node_c 加入队列
    │
    ▼
[node_b] 和 [node_c] 在同一个 superstep 中并行执行
    │
    ▼
都完成后，继续后续节点
```

---

## 循环与递归控制

### 基本循环

```python
# Agent 循环
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, {
    "tools": "tools",
    END: END
})
builder.add_edge("tools", "agent")  # 回到 agent → 形成循环
```

### 递归上限

```python
# 默认最多 25 步
# 可以调整
result = graph.invoke(
    input,
    config={"recursion_limit": 50}  # 最多 50 步
)
```

> ⚠️ 超过递归上限会抛出 `GraphRecursionError`。

---

## 特殊路由场景

### 场景一：跳过节点

```python
def route(state: State) -> str:
    if state.get("skip_processing"):
        return END  # 直接结束
    return "process"
```

### 场景二：回到起点

```python
def route(state: State) -> str:
    if state.get("retry"):
        return "start"  # 回到第一个节点
    return "continue"
```

### 场景三：条件循环

```python
def route(state: State) -> str:
    attempts = state.get("attempts", 0)
    max_attempts = 3

    if state.get("success"):
        return "output"
    elif attempts < max_attempts:
        return "retry"  # 继续尝试
    else:
        return "give_up"
```

---

## 小结

| 边类型 | 语法 | 说明 |
|--------|------|------|
| 普通边 | `add_edge(A, B)` | A→B 固定连接 |
| 并行边 | 多条 `add_edge(A, *)` | A 完成后并行执行多个 |
| 条件边 | `add_conditional_edges(A, fn, map)` | 根据 state 动态路由 |
| Send | `Send(node, input)` | 动态扇出，每任务独立输入 |
| 循环 | 边指回前面的节点 | 需设 recursion_limit |
| 入口边 | `add_edge(START, A)` | 图的起点 |
| 出口边 | `add_edge(B, END)` | 图的终点 |

---

## 下一篇

➡️ [Reducer与状态合并](./04-Reducer与状态合并.md)
