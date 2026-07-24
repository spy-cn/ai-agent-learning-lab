# Node 节点详解

节点（Node）是 LangGraph 图中执行实际计算的单元。本篇深入讲解节点的定义方式、函数签名、最佳实践。

---

## 节点函数签名

### 标准签名

```python
def my_node(state: State) -> dict:
    """
    参数:
        state: 当前图的完整状态（State 对象）
    返回:
        dict: 需要更新的字段（不需要返回完整状态）
    """
    # 读取状态
    messages = state["messages"]

    # 执行逻辑
    result = do_something(messages)

    # 返回部分更新
    return {"field_to_update": result}
```

### 核心规则

```
┌──────────────────────────────────────────────────┐
│ 节点函数规则                                      │
│                                                  │
│ 1. 接收完整的 State 作为参数                       │
│ 2. 返回 dict，只包含需要更新的字段                  │
│ 3. 未返回的字段保持不变                            │
│ 4. 带 Reducer 的字段按 Reducer 策略合并            │
│ 5. 返回空 dict {} 表示不更新任何字段                │
└──────────────────────────────────────────────────┘
```

---

## 节点的多种形态

### 形态一：纯函数节点

最常见的形式——一个 Python 函数：

```python
def retrieve_docs(state: RAGState) -> dict:
    """检索文档"""
    question = state["question"]
    docs = vector_store.similarity_search(question, k=3)
    return {"retrieved_docs": docs}
```

### 形态二：异步节点

```python
async def async_search(state: State) -> dict:
    """异步执行搜索"""
    results = await async_search_api(state["query"])
    return {"search_results": results}

# LangGraph 会自动识别 async 函数并异步执行
graph.add_node("search", async_search)
```

### 形态三：类节点（有状态）

```python
class LLMAgent:
    def __init__(self, model_name: str):
        self.llm = ChatOpenAI(model=model_name)
        self.call_count = 0

    def __call__(self, state: State) -> dict:
        self.call_count += 1
        response = self.llm.invoke(state["messages"])
        return {"messages": [response]}

agent = LLMAgent("gpt-4o")
graph.add_node("agent", agent)
```

### 形态四：Runnable 节点

LangChain 的 Runnable（如 LLM、Chain）可以直接作为节点：

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o")

# 直接用 LLM 作为节点
graph.add_node("llm", llm)

# 用 Chain 作为节点
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是助手"),
    ("human", "{input}")
])
chain = prompt | llm
graph.add_node("chain", chain)
```

> ⚠️ Runnable 节点要求 State 的字段名与 Runnable 的输入字段匹配。

### 形态五：子图节点

```python
# 编译一个子图
subgraph = subgraph_builder.compile()

# 作为主图的节点
main_builder.add_node("sub_process", subgraph)
```

---

## 节点输入与输出

### 读取状态

```python
def my_node(state: State) -> dict:
    # 读取单个字段
    messages = state["messages"]
    question = state["question"]

    # 安全读取（防止 KeyError）
    step = state.get("step", 0)

    # 读取最后一个消息
    last_msg = state["messages"][-1]

    # 检查字段是否存在
    if "retrieved_docs" in state:
        docs = state["retrieved_docs"]
```

### 返回状态更新

```python
def my_node(state: State) -> dict:
    # 方式一：返回完整字段值
    return {"answer": "这是答案"}

    # 方式二：带 Annotated reducer 的字段追加
    return {"messages": [new_message]}  # 会追加到现有列表

    # 方式三：返回多个字段
    return {
        "answer": "这是答案",
        "confidence": 0.95,
        "sources": ["doc1", "doc2"]
    }

    # 方式四：空返回（不更新）
    return {}

    # 方式五：返回 None（不更新）
    return None
```

---

## 节点元数据

### 添加元信息

```python
def my_node(state: State) -> dict:
    ...

# 添加标签（用于中断控制等）
graph.add_node(
    "my_node",
    my_node,
    metadata={"category": "llm", "cost": "high"}
)
```

---

## 多节点协作模式

### 模式一：管道模式

```python
def step_1(state):
    """预处理"""
    return {"data": clean(state["raw_data"])}

def step_2(state):
    """特征提取"""
    return {"features": extract(state["data"])}

def step_3(state):
    """推理"""
    return {"prediction": predict(state["features"])}

builder.add_edge(START, "step_1")
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", "step_3")
builder.add_edge("step_3", END)
```

### 模式二：扇出-聚合模式

```python
from langgraph.constants import Send

def fan_out(state):
    """分发任务"""
    return [Send("worker", {"task": t}) for t in state["tasks"]]

def worker(state):
    """处理单个任务"""
    return {"results": [process(state["task"])]}

def aggregator(state):
    """聚合结果"""
    return {"final_result": combine(state["results"])}

builder.add_conditional_edges("dispatcher", fan_out)
builder.add_edge("worker", "aggregator")
```

### 模式三：循环修正模式

```python
def generator(state):
    """生成初稿"""
    return {"draft": generate(state["task"])}

def reviewer(state):
    """审查"""
    if quality_check(state["draft"]):
        return {"approved": True}
    else:
        return {"feedback": "需要改进...", "approved": False}

def should_regenerate(state):
    if state.get("approved"):
        return END
    return "generator"

builder.add_edge(START, "generator")
builder.add_edge("generator", "reviewer")
builder.add_conditional_edges("reviewer", should_regenerate)
```

---

## 节点中的错误处理

### 方式一：try-except 捕获

```python
def safe_llm_call(state: State) -> dict:
    try:
        response = llm.invoke(state["messages"])
        return {"messages": [response], "error": None}
    except Exception as e:
        return {
            "messages": [AIMessage(content=f"出错了: {e}")],
            "error": str(e)
        }
```

### 方式二：重试逻辑

```python
import time

def llm_with_retry(state: State, max_retries=3) -> dict:
    for attempt in range(max_retries):
        try:
            response = llm.invoke(state["messages"])
            return {"messages": [response]}
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 指数退避
                continue
            raise
```

### 方式三：降级策略

```python
def smart_llm_call(state: State) -> dict:
    """先试 GPT-4，失败则降级到 GPT-3.5"""
    try:
        response = ChatOpenAI(model="gpt-4o").invoke(state["messages"])
        return {"messages": [response], "model_used": "gpt-4o"}
    except:
        response = ChatOpenAI(model="gpt-3.5-turbo").invoke(state["messages"])
        return {"messages": [response], "model_used": "gpt-3.5-turbo"}
```

---

## 节点与配置

节点可以接收运行时配置：

```python
from langchain_core.runnables import ConfigurableField

def llm_node(state: State, config) -> dict:
    """使用 config 中的参数"""
    # 从 config 中读取配置
    model_name = config.get("configurable", {}).get("model", "gpt-4o")
    temperature = config.get("configurable", {}).get("temperature", 0.7)

    llm = ChatOpenAI(model=model_name, temperature=temperature)
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# 运行时传入配置
result = graph.invoke(
    {"messages": [("user", "你好")]},
    config={
        "configurable": {
            "model": "gpt-4o",
            "temperature": 0.3
        }
    }
)
```

---

## 节点调试技巧

### 1. 打印日志

```python
def debug_node(state: State) -> dict:
    print(f"[DEBUG] 当前状态: {list(state.keys())}")
    print(f"[DEBUG] 消息数: {len(state.get('messages', []))}")

    result = process(state)

    print(f"[DEBUG] 返回: {list(result.keys())}")
    return result
```

### 2. 使用 LangSmith

```python
import os
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "debug-demo"

# 所有节点的输入输出会自动记录到 LangSmith
```

### 3. 使用回调

```python
from langchain_core.callbacks import BaseCallbackHandler

class NodeTracer(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"LLM 开始: {serialized['name']}")

    def on_chain_end(self, outputs, **kwargs):
        print(f"节点完成，输出: {outputs}")
```

---

## 小结

| 要点 | 说明 |
|------|------|
| 函数签名 | `def node(state: State) -> dict` |
| 读取状态 | `state["field"]` 或 `state.get("field", default)` |
| 返回更新 | 只返回需要更新的字段 |
| 异步支持 | `async def node(state) -> dict` |
| Runnable 节点 | LLM/Chain 可直接作为节点 |
| 子图节点 | 编译后的子图可作为节点 |
| 配置参数 | 第二参数 `config` |
| 错误处理 | try-except / 重试 / 降级 |

---

## 下一篇

➡️ [Edge边与路由](./03-Edge边与路由.md)
