# Tool Calling 工作流

本篇深入讲解工具调用的完整工作流——从 LLM 决策到工具执行再到结果回注。

---

## 工具调用完整流程

```
步骤1: 用户提问
  "帮我查一下北京天气，然后算23+45"

步骤2: LLM 分析（第一轮）
  → 识别需要两个工具: search_weather + calculate
  → 返回 tool_calls

步骤3: 执行工具（并行）
  → search_weather("北京") → "晴 25°C"
  → calculate("23+45") → "68"

步骤4: LLM 总结（第二轮）
  → 基于工具结果生成最终回答
  → "北京今天晴天25度，23+45=68"

步骤5: 返回用户
```

---

## 单工具调用

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

@tool
def get_weather(city: str) -> str:
    """获取城市天气"""
    weathers = {"北京": "晴 25°C", "上海": "多云 28°C", "广州": "雨 30°C"}
    return weathers.get(city, "未知")

tools = [get_weather]
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools)

class State(TypedDict):
    messages: Annotated[list, add_messages]

def agent(state: State) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: State) -> str:
    if state["messages"][-1].tool_calls:
        return "tools"
    return END

# 构建
builder = StateGraph(State)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
builder.add_edge("tools", "agent")

graph = builder.compile()

# 运行
result = graph.invoke({"messages": [("user", "北京天气怎么样？")]})
print(result["messages"][-1].content)
```

---

## 多工具并行调用

LLM 可以在一次响应中请求调用多个工具：

```python
@tool
def get_weather(city: str) -> str:
    """获取天气"""
    return f"{city}: 晴 25°C"

@tool
def get_time(timezone: str) -> str:
    """获取时间"""
    return f"{timezone}: 14:30"

@tool
def calculate(expression: str) -> str:
    """计算"""
    return str(eval(expression))

tools = [get_weather, get_time, calculate]
llm_with_tools = llm.bind_tools(tools)

# 用户请求多个信息
result = graph.invoke({
    "messages": [("user", "北京天气、东京时间、123*456分别是什么？")]
})

# LLM 会同时请求调用三个工具
# ToolNode 会并行执行它们
```

---

## 工具调用循环详解

```python
"""
完整的工具调用循环示例（带调试输出）
"""
def agent_with_debug(state: State) -> dict:
    """带调试信息的 Agent 节点"""
    print(f"\n[Agent] 分析 {len(state['messages'])} 条消息")

    response = llm_with_tools.invoke(state["messages"])

    if response.tool_calls:
        print(f"[Agent] 决定调用 {len(response.tool_calls)} 个工具:")
        for tc in response.tool_calls:
            print(f"  → {tc['name']}({tc['args']})")
    else:
        print(f"[Agent] 生成最终回答: {response.content[:50]}...")

    return {"messages": [response]}


def tool_executor_with_debug(state: State) -> dict:
    """带调试信息的工具执行"""
    last_msg = state["messages"][-1]
    results = []

    for tc in last_msg.tool_calls:
        print(f"\n[Tool] 执行 {tc['name']}({tc['args']})")

        # 执行工具
        for t in tools:
            if t.name == tc["name"]:
                result = t.invoke(tc["args"])
                print(f"[Tool] 结果: {result[:50]}...")
                results.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tc["id"]
                ))
                break

    return {"messages": results}
```

### 执行过程

```
用户: 北京天气怎么样？123*456=?

[Agent] 分析 1 条消息
[Agent] 决定调用 2 个工具:
  → get_weather({'city': '北京'})
  → calculate({'expression': '123*456'})

[Tool] 执行 get_weather({'city': '北京'})
[Tool] 结果: 北京: 晴 25°C

[Tool] 执行 calculate({'expression': '123*456'})
[Tool] 结果: 56088

[Agent] 分析 4 条消息
[Agent] 生成最终回答: 北京今天晴天25度，123*456=56088...
```

---

## 工具调用的高级模式

### 模式一：链式工具调用

```python
@tool
def search_person(name: str) -> str:
    """搜索人物信息"""
    return f"{name}是XYZ公司的CEO"

@tool
def search_company(company: str) -> str:
    """搜索公司信息"""
    return f"{company}成立于2010年，有500人"

# LLM 会先搜索人物，获取公司名，再搜索公司
# 用户: "介绍一下张三和他的公司"
# → search_person("张三") → "张三是XYZ公司的CEO"
# → search_company("XYZ") → "XYZ成立于2010年"
```

### 模式二：条件工具选择

```python
@tool
def search_web(query: str) -> str:
    """搜索网络（时效性信息）"""
    ...

@tool
def search_knowledge_base(query: str) -> str:
    """搜索知识库（内部知识）"""
    ...

# LLM 根据问题性质选择不同工具
# "今天的新闻" → search_web
# "公司请假制度" → search_knowledge_base
```

### 模式三：工具 + RAG

```python
@tool
def retrieve_docs(query: str) -> str:
    """从知识库检索文档"""
    docs = vector_store.similarity_search(query)
    return "\n".join(d.page_content for d in docs)

@tool
def search_web(query: str) -> str:
    """搜索互联网"""
    ...

@tool
def calculator(expression: str) -> str:
    """数学计算"""
    ...

# Agent 可以组合使用 RAG + 搜索 + 计算
```

---

## 错误处理

### 工具执行错误

```python
def safe_tool_executor(state: State) -> dict:
    """带错误处理的工具执行"""
    last_msg = state["messages"][-1]
    results = []

    for tc in last_msg.tool_calls:
        try:
            # 执行工具
            for t in tools:
                if t.name == tc["name"]:
                    result = t.invoke(tc["args"])
                    results.append(ToolMessage(
                        content=str(result),
                        tool_call_id=tc["id"]
                    ))
                    break
            else:
                # 工具不存在
                results.append(ToolMessage(
                    content=f"错误: 工具 '{tc['name']}' 不存在",
                    tool_call_id=tc["id"]
                ))

        except Exception as e:
            # 执行出错
            results.append(ToolMessage(
                content=f"工具执行出错: {str(e)}",
                tool_call_id=tc["id"]
            ))

    return {"messages": results}
```

### 工具重试

```python
@tool
def unreliable_api(query: str) -> str:
    """可能失败的API"""
    import random
    if random.random() < 0.3:  # 30%概率失败
        raise ConnectionError("API超时")
    return f"结果: {query}"

def tool_executor_with_retry(state: State, max_retries=2) -> dict:
    last_msg = state["messages"][-1]
    results = []

    for tc in last_msg.tool_calls:
        for attempt in range(max_retries + 1):
            try:
                result = execute_tool(tc)
                results.append(ToolMessage(
                    content=result,
                    tool_call_id=tc["id"]
                ))
                break
            except Exception as e:
                if attempt < max_retries:
                    continue
                results.append(ToolMessage(
                    content=f"重试{max_retries}次后仍失败: {e}",
                    tool_call_id=tc["id"]
                ))
    return {"messages": results}
```

---

## 使用预构建 Agent

LangGraph 提供了预构建的 ReAct Agent：

```python
from langgraph.prebuilt import create_react_agent

# 一行创建 Agent
agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4o"),
    tools=[get_weather, calculate, search_web]
)

# 直接使用
result = agent.invoke({
    "messages": [("user", "北京天气怎么样？23+45=?")]
})
```

### create_react_agent 的图结构

```
START → agent ←─── tools
          │            │
          │  (有tool_calls)│
          └────────────┘
          │
          │ (无tool_calls)
          ▼
         END
```

---

## 流式输出工具调用

```python
# 流式查看工具调用过程
for event in graph.stream(
    {"messages": [("user", "查天气和算数学")]},
    stream_mode="updates"
):
    for node, update in event.items():
        if node == "agent":
            msg = update["messages"][-1]
            if msg.tool_calls:
                print(f"Agent 想调用: {[tc['name'] for tc in msg.tool_calls]}")
            else:
                print(f"Agent 回答: {msg.content[:100]}")

        elif node == "tools":
            for msg in update["messages"]:
                print(f"工具结果: {msg.content[:50]}")
```

---

## 小结

| 要点 | 说明 |
|------|------|
| 工具调用流程 | LLM决策 → 执行工具 → 结果回注 → LLM总结 |
| tool_calls | AIMessage 中的工具调用请求 |
| ToolMessage | 工具执行结果 |
| ToolNode | 预构建的工具执行节点 |
| create_react_agent | 快速创建工具Agent |
| 并行调用 | LLM 可同时请求多个工具 |
| 链式调用 | 前一个工具的结果影响后续 |
| 错误处理 | try-except + 重试机制 |

---

## 下一篇

➡️ [预构建工具生态](./03-预构建工具生态.md)
