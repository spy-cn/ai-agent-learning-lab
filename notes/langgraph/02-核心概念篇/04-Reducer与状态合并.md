# Reducer 与状态合并

Reducer 是 LangGraph 的精髓之一——它定义了**节点返回值如何与当前状态合并**。理解 Reducer 是掌握 LangGraph 状态管理的关键。

---

## 什么是 Reducer

### 问题引入

考虑这个场景：

```python
class State(TypedDict):
    messages: list  # 没有 reducer

# 节点 A 返回
def node_a(state):
    return {"messages": [new_message]}

# 当前状态: {"messages": [msg1, msg2]}
# 节点A返回后，messages 变成什么？
```

**没有 Reducer（默认行为：覆盖）**：
```
当前: [msg1, msg2]
返回: [new_message]
结果: [new_message]       ← msg1 和 msg2 丢失了！
```

**有 Reducer（追加行为）**：
```
当前: [msg1, msg2]
返回: [new_message]
结果: [msg1, msg2, new_message]  ← 正确追加
```

### Reducer 定义

Reducer 是一个**二元函数**：接收旧值和新值，返回合并后的值。

```python
# Reducer 函数签名
def reducer(old_value, new_value) -> merged_value:
    ...
```

---

## 使用 Annotated 标注 Reducer

在 TypedDict 中使用 `Annotated` 为字段指定 Reducer：

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
import operator

class State(TypedDict):
    # 追加 reducer：新消息追加到列表末尾
    messages: Annotated[list, add_messages]

    # operator.add reducer：列表拼接
    collected: Annotated[list, operator.add]

    # 默认（无 Annotated）：覆盖
    current_answer: str

    # 自定义 reducer
    score: Annotated[int, lambda old, new: max(old, new)]  # 取最大值
```

---

## 内置 Reducer

### 1. add_messages

最常用的 Reducer，专为对话消息设计：

```python
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]
```

特点：
- 新消息**追加**到列表
- 支持 LangChain 的各种 Message 类型
- 自动处理消息 ID（去重）

```python
# 当前状态
state = {"messages": [HumanMessage("你好")]}

# 节点返回
return {"messages": [AIMessage("你好！我是助手")]}

# 合并后
state = {"messages": [HumanMessage("你好"), AIMessage("你好！我是助手")]}
```

### 2. operator.add

简单的列表拼接或加法：

```python
import operator

class State(TypedDict):
    # 列表拼接
    search_results: Annotated[list, operator.add]
    # 整数累加
    total_count: Annotated[int, operator.add]
```

```python
# 列表拼接
current: ["doc1", "doc2"]
returned: ["doc3", "doc4"]
merged: ["doc1", "doc2", "doc3", "doc4"]

# 整数累加
current: 5
returned: 3
merged: 8
```

### 3. operator.or_

字典合并：

```python
import operator

class State(TypedDict):
    config: Annotated[dict, operator.or_]
```

```python
current: {"a": 1, "b": 2}
returned: {"b": 3, "c": 4}
merged: {"a": 1, "b": 3, "c": 4}  # 后者覆盖相同 key
```

---

## 自定义 Reducer

### 基本自定义

```python
from typing import Annotated

def keep_max(old: int, new: int) -> int:
    """保留最大值"""
    return max(old, new) if old is not None else new

def keep_last(old, new):
    """保留最后一个非空值"""
    return new if new is not None else old

class State(TypedDict):
    best_score: Annotated[int, keep_max]
    latest_response: Annotated[str, keep_last]
```

### 去重追加 Reducer

```python
def append_unique(old: list, new: list) -> list:
    """追加但去重"""
    result = list(old) if old else []
    for item in new:
        if item not in result:
            result.append(item)
    return result

class State(TypedDict):
    unique_docs: Annotated[list, append_unique]
```

### 限制长度的追加

```python
def append_with_limit(old: list, new: list, max_len: int = 10) -> list:
    """追加但限制列表总长度"""
    combined = (old or []) + new
    return combined[-max_len:]  # 只保留最后 max_len 个

class State(TypedDict):
    recent_messages: Annotated[list, append_with_limit]
```

### 字典深度合并

```python
def deep_merge(old: dict, new: dict) -> dict:
    """深度合并两个字典"""
    result = dict(old) if old else {}
    for key, value in (new or {}).items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result

class State(TypedDict):
    metadata: Annotated[dict, deep_merge]
```

---

## Reducer 在并行执行中的作用

当多个节点**并行执行**并返回同一个字段时，Reducer 决定了如何合并：

### 没有 Reducer 的问题

```python
class State(TypedDict):
    results: list  # 没有 reducer

# 两个节点同时返回 results
def node_a(state): return {"results": ["a"]}
def node_b(state): return {"results": ["b"]}
```

```
并行执行:
  node_a 返回 {"results": ["a"]}
  node_b 返回 {"results": ["b"]}

没有 reducer: 后执行的覆盖先执行的 → 只剩 ["b"]！
```

### 使用 Reducer 解决

```python
class State(TypedDict):
    results: Annotated[list, operator.add]

# 两个节点同时返回
def node_a(state): return {"results": ["a"]}
def node_b(state): return {"results": ["b"]}
```

```
并行执行:
  node_a 返回 {"results": ["a"]}
  node_b 返回 {"results": ["b"]}

有 reducer: ["a"] + ["b"] = ["a", "b"] ✓
```

### 图解并行合并

```
         ┌──► [Node A] returns {"results": ["a"]} ──┐
[Start] ──┤                                          │  operator.add
         └──► [Node B] returns {"results": ["b"]} ──┘  ============►
                                                                   │
                                                                   ▼
                                                    state["results"] = ["a", "b"]
```

---

## 实战：RAG 状态设计

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
import operator

class RAGState(TypedDict):
    # 对话消息（追加）
    messages: Annotated[list, add_messages]

    # 用户问题（覆盖）
    question: str

    # 检索到的文档（追加，支持多次检索累积）
    retrieved_docs: Annotated[list, operator.add]

    # 重写后的查询（覆盖）
    rewritten_query: str

    # 最终答案（覆盖）
    answer: str

    # 检索评分列表（追加）
    doc_scores: Annotated[list[float], operator.add]

    # 步骤计数（累加）
    retrieval_count: Annotated[int, operator.add]
```

---

## Reducer 调试

### 查看状态快照

```python
# 启用 checkpointer 后，可以查看每一步的状态
graph = builder.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "1"}}

graph.invoke(input, config)

# 查看所有状态快照
for snapshot in graph.get_state_history(config):
    print(f"Step: {snapshot.metadata['step']}")
    print(f"State: {snapshot.values}")
    print(f"Next: {snapshot.next}")
    print("---")
```

### 打印状态变化

```python
def debug_reducer_node(state: State) -> dict:
    """调试用：打印当前状态"""
    print(f"  messages count: {len(state.get('messages', []))}")
    print(f"  docs count: {len(state.get('retrieved_docs', []))}")
    print(f"  answer: {state.get('answer', 'N/A')[:50]}")
    return {}  # 不修改状态
```

---

## 常见 Reducer 模式速查

| 模式 | Reducer | 场景 |
|------|---------|------|
| 对话消息 | `add_messages` | 聊天历史 |
| 列表追加 | `operator.add` | 搜索结果累积 |
| 整数累加 | `operator.add` | 步骤计数器 |
| 取最大值 | `lambda o,n: max(o,n)` | 最佳分数 |
| 取最小值 | `lambda o,n: min(o,n)` | 最小成本 |
| 字典合并 | `operator.or_` | 配置更新 |
| 覆盖（默认） | 无需标注 | 当前状态、答案 |
| 保留首个 | `lambda o,n: o if o else n` | 初始值不变 |
| 保留末个 | `lambda o,n: n if n else o` | 最新结果 |

---

## 小结

| 要点 | 说明 |
|------|------|
| Reducer 作用 | 定义节点返回值如何与当前状态合并 |
| 默认行为 | 覆盖（新值替换旧值） |
| add_messages | 追加消息到列表 |
| operator.add | 列表拼接/数字累加 |
| 自定义 Reducer | `def func(old, new) -> merged` |
| 并行场景 | Reducer 是正确合并并行结果的关键 |
| Annotated 标注 | `Annotated[Type, reducer_func]` |

---

## 下一篇

➡️ [图的编译与运行](./05-图的编译与运行.md)
