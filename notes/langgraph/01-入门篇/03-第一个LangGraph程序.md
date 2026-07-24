# 第一个 LangGraph 程序

本篇带你从零开始，一步步构建一个完整的 LangGraph 对话机器人，涵盖**状态定义、节点创建、图组装、编译运行**全流程。

---

## 目标：构建一个对话机器人

我们要实现的流程：

```
用户输入 → [chatbot节点] → LLM回复 → 判断是否继续
                ↑                      │
                └──────────────────────┘
                   （循环对话）
```

---

## 第一步：定义状态

状态（State）是 LangGraph 的核心——所有数据都通过它在节点间流转。

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    # messages 字段使用 add_messages reducer
    # 这样新消息会追加到列表，而非覆盖
    messages: Annotated[list, add_messages]
```

### 什么是 Reducer？

Reducer 定义了**节点返回值如何与当前状态合并**：

```python
# 没有 reducer（默认行为：覆盖）
state["messages"] = node_return["messages"]

# 有 add_messages reducer（追加）
state["messages"] = state["messages"] + node_return["messages"]
```

### 图解

```
当前状态:                          节点返回:
messages: [                        messages: [
  HumanMessage("你好"),              AIMessage("你好！我是机器人")
]                                   ]
         │                              │
         └──────────┬───────────────────┘
                    │ add_messages reducer
                    ▼
合并后状态:
messages: [
  HumanMessage("你好"),
  AIMessage("你好！我是机器人")     ← 追加
]
```

---

## 第二步：创建节点

节点是执行计算的函数，接收状态，返回状态更新。

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

def chatbot(state: State):
    """对话机器人节点：调用LLM生成回复"""
    response = llm.invoke(state["messages"])
    # 只需返回要更新的字段（会通过reducer合并）
    return {"messages": [response]}
```

### 节点函数签名

```python
def node_name(state: State) -> dict:
    """
    参数: state - 当前图的完整状态
    返回: dict - 只包含需要更新的字段
    """
    # 读取状态
    messages = state["messages"]

    # 执行逻辑...
    result = do_something(messages)

    # 返回状态更新（不需要返回完整状态）
    return {"field_to_update": result}
```

---

## 第三步：组装图

```python
from langgraph.graph import StateGraph, START, END

# 1. 创建图构建器，传入状态类型
graph_builder = StateGraph(State)

# 2. 添加节点
graph_builder.add_node("chatbot", chatbot)

# 3. 添加边（定义执行顺序）
graph_builder.add_edge(START, "chatbot")  # 开始 → chatbot
graph_builder.add_edge("chatbot", END)    # chatbot → 结束

# 4. 编译图
graph = graph_builder.compile()
```

### 图结构可视化

```
┌─────────┐     ┌───────────┐     ┌─────────┐
│  START  │ ──► │  chatbot  │ ──► │   END   │
└─────────┘     └───────────┘     └─────────┘
```

---

## 第四步：运行图

### 基本调用

```python
# invoke: 同步执行，返回最终结果
result = graph.invoke({
    "messages": [
        ("user", "你好！请介绍一下你自己。")
    ]
})

print(result["messages"][-1].content)
# "你好！我是一个AI助手..."
```

### 流式输出

```python
# stream: 逐步骤输出中间结果
for event in graph.stream({
    "messages": [("user", "写一首关于秋天的诗")]
}):
    for node_name, node_output in event.items():
        print(f"[{node_name}]")
        if "messages" in node_output:
            print(node_output["messages"][-1].content)
```

### 流式 Token

```python
# 逐 token 流式输出
for msg, metadata in graph.stream(
    {"messages": [("user", "解释一下量子计算")]},
    stream_mode="messages"
):
    if msg.content:
        print(msg.content, end="", flush=True)
```

---

## 完整代码

```python
"""
第一个 LangGraph 程序：简单对话机器人
"""

import os
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


# ========== 1. 定义状态 ==========
class State(TypedDict):
    messages: Annotated[list, add_messages]


# ========== 2. 初始化 LLM ==========
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)


# ========== 3. 定义节点 ==========
def chatbot(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# ========== 4. 构建图 ==========
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()


# ========== 5. 运行 ==========
if __name__ == "__main__":
    # 单轮对话
    print("=" * 50)
    result = graph.invoke({
        "messages": [("user", "你好！你是谁？")]
    })
    print("AI:", result["messages"][-1].content)

    # 多轮对话（需要传入完整历史）
    print("\n" + "=" * 50)
    history = [("user", "我叫小明")]
    result1 = graph.invoke({"messages": history})
    ai_reply = result1["messages"][-1]

    history.extend([result1["messages"][-2], ai_reply])
    history.append(("user", "我叫什么名字？"))
    result2 = graph.invoke({"messages": history})
    print("AI:", result2["messages"][-1].content)
    # AI 应该能回答出"小明"（因为有完整历史）
```

---

## 加入工具调用：更进阶的例子

让我们构建一个能**调用工具**的 Agent：

```python
"""
带工具调用的 Agent
"""
from langchain_core.tools import tool

@tool
def search_weather(city: str) -> str:
    """查询指定城市的天气"""
    # 模拟数据
    weathers = {"北京": "晴 25°C", "上海": "多云 28°C", "广州": "雷阵雨 30°C"}
    return weathers.get(city, f"暂无{city}的天气信息")

@tool
def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression)  # 生产环境请用安全的表达式解析器
        return f"{expression} = {result}"
    except:
        return "无法计算"

# 绑定工具到 LLM
llm_with_tools = llm.bind_tools([search_weather, calculate])


# === 状态定义 ===
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# === 节点定义 ===
def agent(state: AgentState):
    """LLM 推理节点"""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def tool_executor(state: AgentState):
    """工具执行节点"""
    last_message = state["messages"][-1]
    results = []
    for tool_call in last_message.tool_calls:
        # 执行工具
        if tool_call["name"] == "search_weather":
            result = search_weather.invoke(tool_call["args"])
        elif tool_call["name"] == "calculate":
            result = calculate.invoke(tool_call["args"])
        else:
            result = "未知工具"

        from langchain_core.messages import ToolMessage
        results.append(ToolMessage(
            content=result,
            tool_call_id=tool_call["id"]
        ))
    return {"messages": results}


# === 路由函数 ===
def should_use_tools(state: AgentState) -> str:
    """判断是否需要调用工具"""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


# === 构建图 ===
graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", agent)
graph_builder.add_node("tools", tool_executor)

graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", should_use_tools, {
    "tools": "tools",
    END: END
})
graph_builder.add_edge("tools", "agent")  # 工具执行后回到 agent

graph = graph_builder.compile()


# === 运行 ===
if __name__ == "__main__":
    result = graph.invoke({
        "messages": [("user", "北京和上海今天天气怎么样？23+45等于几？")]
    })

    print("最终回复:", result["messages"][-1].content)
```

### 图结构

```
┌─────────┐     ┌────────┐     ┌───────┐
│  START  │ ──► │ agent  │ ──► │ 路由判断│
└─────────┘     └────────┘     └───┬───┘
                   ▲               │
                   │         ┌─────┴─────┐
                   │         │           │
                   │      有工具调用   无工具调用
                   │         │           │
              ┌────┴─────┐    │           │
              │  tools   │    │           │
              │ (执行工具)│    │           │
              └──────────┘    │           │
                              ▼           ▼
                         [继续循环]     END
```

---

## 添加记忆：对话持久化

默认情况下，每次 `invoke` 都是独立的。使用 Checkpoint 可以让对话**持久化**：

```python
from langgraph.checkpoint.memory import MemorySaver

# 创建记忆存储
memory = MemorySaver()

# 编译时传入
graph = graph_builder.compile(checkpointer=memory)

# 使用 thread_id 标识对话
config = {"configurable": {"thread_id": "conversation-1"}}

# 第一轮
graph.invoke({"messages": [("user", "我叫小红")]}, config=config)

# 第二轮（同一个 thread_id，会记住之前的对话）
result = graph.invoke({"messages": [("user", "我叫什么？")]}, config=config)
print(result["messages"][-1].content)  # "你叫小红"
```

---

## 小结

| 概念 | 说明 | 代码 |
|------|------|------|
| **State** | 数据容器 | `TypedDict` + `Annotated` |
| **Node** | 计算单元 | `def node(state) -> dict` |
| **Edge** | 执行顺序 | `add_edge(A, B)` |
| **Conditional Edge** | 条件路由 | `add_conditional_edges` |
| **Compile** | 编译图 | `graph_builder.compile()` |
| **Invoke** | 执行图 | `graph.invoke(input)` |
| **Stream** | 流式执行 | `graph.stream(input)` |
| **Checkpoint** | 状态持久化 | `compile(checkpointer=memory)` |

### 构建流程总结

```
1. 定义 State（TypedDict + Annotated reducer）
       ↓
2. 定义 Node 函数（读 state → 计算逻辑 → 返回更新）
       ↓
3. 创建 StateGraph（传入 State 类型）
       ↓
4. 添加节点（add_node）
       ↓
5. 添加边（add_edge / add_conditional_edges）
       ↓
6. 编译图（compile）
       ↓
7. 运行（invoke / stream）
```

---

## 下一篇

➡️ [架构总览与设计哲学](./04-架构总览与设计哲学.md)
