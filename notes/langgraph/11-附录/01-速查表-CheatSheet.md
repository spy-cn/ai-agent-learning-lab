# 速查表 CheatSheet

LangGraph 常用 API 和模式快速参考。

---

## 核心 API

### 创建图

```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(StateType)
builder.add_node("name", function)
builder.add_edge("A", "B")
builder.add_edge(START, "first")
builder.add_edge("last", END)
builder.add_conditional_edges("source", router_fn, {"path": "node", END: END})
graph = builder.compile(checkpointer=..., interrupt_before=[...])
```

### 运行

```python
# 同步
result = graph.invoke(input, config)
# 流式
for event in graph.stream(input, config, stream_mode="updates"): ...
# 异步
result = await graph.ainvoke(input, config)
async for event in graph.astream(input, config): ...
```

### 状态管理

```python
state = graph.get_state(config)
graph.update_state(config, {"key": value})
graph.get_state_history(config)
```

---

## 状态定义模板

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
import operator

class State(TypedDict):
    messages: Annotated[list, add_messages]  # 消息追加
    docs: Annotated[list, operator.add]      # 列表追加
    counter: Annotated[int, operator.add]    # 数字累加
    answer: str                               # 默认覆盖
```

---

## 常用模式

### Tool-Calling Agent

```python
from langgraph.prebuilt import create_react_agent
agent = create_react_agent(llm, tools)
```

### 条件路由

```python
def router(state) -> str:
    if state["condition"]:
        return "node_a"
    return "node_b"

builder.add_conditional_edges("source", router)
```

### 并行 Map-Reduce

```python
from langgraph.constants import Send

def fan_out(state):
    return [Send("worker", {"item": i}) for i in state["items"]]
```

### HITL 中断

```python
graph = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["approval"]
)
# 暂停后恢复
graph.invoke(None, config)
```

---

## 工具定义

```python
@tool
def my_tool(param: str) -> str:
    """工具描述"""
    return result

# 绑定
llm_with_tools = llm.bind_tools([tool1, tool2])
```

---

## Checkpointers

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.postgres import PostgresSaver

memory = MemorySaver()
sqlite = SqliteSaver.from_conn_string("db.sqlite")
postgres = PostgresSaver.from_conn_string(DB_URI)
```

---

## 常用导入

```python
# 图
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.constants import Send
from langgraph.checkpoint.memory import MemorySaver

# 预构建
from langgraph.prebuilt import create_react_agent, ToolNode

# LangChain
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
```
