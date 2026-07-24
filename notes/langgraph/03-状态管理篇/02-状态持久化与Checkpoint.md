# 状态持久化与 Checkpoint

Checkpoint 是 LangGraph 实现**状态持久化**的核心机制。它让图可以在任意时刻暂停、恢复、回滚，是构建可靠 Agent 的基础设施。

---

## 为什么需要 Checkpoint

### 没有 Checkpoint

```
用户: "你好"                    用户: "我叫什么？"
  │                               │
  ▼                               ▼
graph.invoke()                 graph.invoke()
  │                               │
  ▼                               ▼
AI: "你好！我是助手"            AI: "抱歉，我不知道你的名字"
                                 ↑ 忘记了之前的对话！
```

### 有 Checkpoint

```
用户: "你好"                    用户: "我叫什么？"
  │                               │
  ▼                               ▼
graph.invoke()                 graph.invoke()
  │ (thread_id="conv1")          │ (thread_id="conv1") ← 同一对话
  ▼                               ▼
AI: "你好！"                   AI: "你之前说叫小明"
  │                               ↑ 记住了！
  └── 状态自动保存到 Checkpoint
```

---

## Checkpoint 的工作原理

```
每次节点执行后:

[Node A 执行] → 保存快照 → [Checkpoint 1]
[Node B 执行] → 保存快照 → [Checkpoint 2]
[Node C 执行] → 保存快照 → [Checkpoint 3]
                              │
                              ▼
                        存储后端
                     (内存/Redis/PostgreSQL)

恢复时:
读取 Checkpoint 3 → 从 Node C 之后继续执行
```

---

## Checkpoint 存储后端

### 1. MemorySaver（开发/测试）

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
```

特点：
- 存储在内存中，进程退出即丢失
- 无需额外依赖
- 适合开发和测试

### 2. SqliteSaver（单机持久化）

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# 持久化到 SQLite 文件
with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
```

### 3. PostgresSaver（生产环境）

```python
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://user:password@localhost:5432/langgraph"

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()  # 创建必要的表
    graph = builder.compile(checkpointer=checkpointer)
```

### 4. Redis（高性能缓存型）

```python
from langgraph.checkpoint.redis import RedisSaver

checkpointer = RedisSaver.from_conn_string("redis://localhost:6379")
graph = builder.compile(checkpointer=checkpointer)
```

### 后端选择指南

| 后端 | 适用场景 | 持久性 | 性能 | 复杂度 |
|------|----------|--------|------|--------|
| MemorySaver | 开发/测试 | 进程内 | 快 | 低 |
| SqliteSaver | 单机应用 | 文件 | 中 | 低 |
| PostgresSaver | 生产环境 | 数据库 | 中 | 中 |
| RedisSaver | 高并发 | 内存+磁盘 | 快 | 中 |

---

## 使用 Checkpoint

### 基本用法

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

# 编译时传入 checkpointer
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# 必须指定 thread_id
config = {"configurable": {"thread_id": "conversation-1"}}

# 第一轮对话
result = graph.invoke(
    {"messages": [("user", "我叫小明")]},
    config=config
)

# 第二轮对话（同一个 thread_id）
result = graph.invoke(
    {"messages": [("user", "我叫什么名字？")]},
    config=config
)
# AI 能回答出"小明"
```

### thread_id 的作用

```python
# 不同的 thread_id 完全隔离
config_alice = {"configurable": {"thread_id": "alice"}}
config_bob = {"configurable": {"thread_id": "bob"}}

# Alice 的对话
graph.invoke({"messages": [("user", "我是Alice")]}, config=config_alice)

# Bob 的对话
graph.invoke({"messages": [("user", "我是Bob")]}, config=config_bob)

# Alice 继续
result = graph.invoke(
    {"messages": [("user", "我叫什么？")]},
    config=config_alice
)
# AI: "你是Alice"
```

---

## 状态历史与回滚

### 查看状态历史

```python
config = {"configurable": {"thread_id": "history-demo"}}

# 执行几步
graph.invoke({"messages": [("user", "你好")]}, config=config)
graph.invoke({"messages": [("user", "今天天气")]}, config=config)

# 查看完整历史
for snapshot in graph.get_state_history(config):
    step = snapshot.metadata.get("step", 0)
    next_nodes = snapshot.next
    msg_count = len(snapshot.values.get("messages", []))
    print(f"Step {step}: {msg_count} messages, next={next_nodes}")
```

输出示例：
```
Step 4: 4 messages, next=()
Step 3: 3 messages, next=('chatbot',)
Step 2: 2 messages, next=()
Step 1: 1 messages, next=('chatbot',)
Step 0: 0 messages, next=('chatbot',)
```

### 回滚到历史状态

```python
# 获取历史快照列表
history = list(graph.get_state_history(config))

# 回滚到第 2 步
target_config = history[2].config

# 从该状态继续执行
result = graph.invoke(None, config=target_config)
```

### 图解回滚

```
正常执行:
  Step 0 → Step 1 → Step 2 → Step 3 → Step 4
                                     (当前)

回滚到 Step 2:
  Step 0 → Step 1 → Step 2 → Step 3' → Step 4'
                         ↑
                      (新分支从这里开始)
```

---

## 手动操作状态

### 获取当前状态

```python
state = graph.get_state(config)

print(state.values)      # 当前状态值
print(state.next)        # 下一个要执行的节点
print(state.metadata)    # 元数据
print(state.tasks)       # 待执行的任务
```

### 手动更新状态

```python
# 直接更新状态
graph.update_state(config, {
    "messages": [HumanMessage("手动添加的消息")]
})

# 模拟某个节点的输出更新
graph.update_state(
    config,
    {"answer": "这是手动设置的答案"},
    as_node="generate_node"  # 声明为某个节点的输出
)
```

### 实战：人工修正后继续

```python
# 执行到中断点
graph.invoke(input, config=config)

# 查看当前草稿
state = graph.get_state(config)
draft = state.values.get("draft", "")
print(f"AI生成的草稿: {draft}")

# 人工修正
corrected = draft.replace("错误内容", "正确内容")
graph.update_state(config, {"draft": corrected})

# 从修正后的状态继续
result = graph.invoke(None, config=config)
```

---

## 多线程并发

```python
# 多个用户同时使用
async def handle_user_message(user_id: str, message: str):
    config = {"configurable": {"thread_id": f"user-{user_id}"}}
    result = await graph.ainvoke(
        {"messages": [("user", message)]},
        config=config
    )
    return result["messages"][-1].content

# 并发处理多个用户
import asyncio
tasks = [
    handle_user_message("alice", "你好"),
    handle_user_message("bob", "今天新闻"),
    handle_user_message("charlie", "帮我算个题"),
]
results = await asyncio.gather(*tasks)
```

---

## 完整示例：带历史回滚的对话系统

```python
"""
完整的 Checkpoint 管理示例
"""
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

class State(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatOpenAI(model="gpt-4o")

def chatbot(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# === 演示 ===
config = {"configurable": {"thread_id": "demo"}}

# 对话几轮
graph.invoke({"messages": [HumanMessage("我喜欢吃苹果")]}, config)
graph.invoke({"messages": [HumanMessage("我也喜欢香蕉")]}, config)
result = graph.invoke({"messages": [HumanMessage("我喜欢什么水果？")]}, config)
print("当前回答:", result["messages"][-1].content)

# 查看历史
print("\n=== 历史快照 ===")
for snapshot in graph.get_state_history(config):
    step = snapshot.metadata.get("step", 0)
    msgs = snapshot.values.get("messages", [])
    print(f"Step {step}: {len(msgs)} messages")

# 回滚到第2步（只保留"喜欢吃苹果"）
history = list(graph.get_state_history(config))
rollback_config = history[-3].config  # 倒数第3个是Step 2

# 从回滚点继续
result = graph.invoke(
    {"messages": [HumanMessage("我喜欢什么水果？")]},
    config=rollback_config
)
print("\n回滚后回答:", result["messages"][-1].content)
# 应该不知道"香蕉"（因为回滚了）
```

---

## Checkpoint 数据结构

每个 Checkpoint 包含：

```python
{
    "thread_id": "conversation-1",        # 线程ID
    "checkpoint_id": "uuid-xxx",          # 快照唯一ID
    "parent_id": "uuid-yyy",              # 父快照ID（形成链）
    "channel_values": {                   # 状态值
        "messages": [...],
        "answer": "..."
    },
    "channel_versions": {                 # 各字段版本号
        "messages": 3,
        "answer": 2
    },
    "versions_seen": {                    # 每个节点看到的版本
        "node_a": {"messages": 2},
        "node_b": {"messages": 1}
    },
    "metadata": {
        "step": 4,                        # 执行步数
        "source": "loop",                 # 来源
        "writes": {                       # 本次写入
            "node_a": {"messages": [...]}
        }
    }
}
```

---

## 小结

| 要点 | 说明 |
|------|------|
| Checkpoint | 图执行的状态快照 |
| MemorySaver | 内存存储，开发用 |
| SqliteSaver | 文件持久化，单机用 |
| PostgresSaver | 数据库持久化，生产用 |
| thread_id | 对话线程隔离标识 |
| get_state | 获取当前状态 |
| get_state_history | 获取历史快照列表 |
| update_state | 手动更新状态 |
| 回滚 | 从历史 config 继续 |

---

## 下一篇

➡️ [对话记忆管理](./03-对话记忆管理.md)
