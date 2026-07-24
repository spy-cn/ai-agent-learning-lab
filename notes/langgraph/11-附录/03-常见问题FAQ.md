# 常见问题 FAQ

---

## 安装与环境

### Q: Python 版本要求？

**A:** Python 3.10+，推荐 3.11 或 3.12。

### Q: pip install 失败/超时？

**A:** 使用国内镜像：
```bash
pip install langgraph -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q: LangGraph 和 LangChain 版本冲突？

**A:** 确保版本兼容：
```bash
pip install langgraph>=0.2.0 langchain-core>=0.3.0
```

---

## 状态管理

### Q: 节点返回后状态没更新？

**A:** 检查以下几点：
1. 返回的 key 是否与 State 定义匹配
2. 如果是列表，是否使用了 Annotated reducer
3. 是否返回了空 dict

```python
# ❌ 不会追加（没有 reducer）
class State(TypedDict):
    docs: list

# ✅ 会追加
class State(TypedDict):
    docs: Annotated[list, operator.add]
```

### Q: 多轮对话没有记忆？

**A:** 需要 Checkpointer + thread_id：
```python
graph = builder.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "conv1"}}
graph.invoke(input, config=config)  # 每次用同一个 thread_id
```

---

## 工具调用

### Q: LLM 不调用工具？

**A:** 检查：
1. 是否调用了 `bind_tools()`
2. 工具描述是否清晰
3. 尝试用 `tool_choice="auto"` 或指定工具

### Q: 工具执行报错？

**A:** 确保返回 `ToolMessage`：
```python
from langchain_core.messages import ToolMessage

results.append(ToolMessage(
    content=str(result),
    tool_call_id=tc["id"]  # 必须关联 ID
))
```

### Q: 用 ToolNode vs 手动执行？

**A:** 简单场景用 `ToolNode`，需要自定义逻辑（重试、日志等）时手动执行。

---

## 图执行

### Q: GraphRecursionError？

**A:** 循环太多，调整限制或检查退出条件：
```python
# 调整限制
graph.invoke(input, config={"recursion_limit": 50})

# 检查条件边是否正确返回 END
```

### Q: 图执行很慢？

**A:** 参见[性能优化](../09-部署运维篇/04-性能优化与扩展.md)：
- 减少 LLM 调用
- 并行化
- 使用缓存

### Q: interrupt 后怎么恢复？

**A:** 传 `None` 继续：
```python
# 暂停后
graph.invoke(None, config=config)  # 从暂停处继续
```

---

## 多 Agent

### Q: Supervisor 一直循环不结束？

**A:** 检查：
1. Supervisor 是否有 `FINISH` 退出条件
2. 设置步数限制
3. 检查 LLM 是否正确决策

### Q: Agent 之间怎么共享数据？

**A:** 通过共享 State：
```python
class TeamState(TypedDict):
    shared_data: Annotated[list, operator.add]  # 所有 Agent 可读写
```

---

## 部署

### Q: 多实例部署状态不同步？

**A:** 使用外部 Checkpointer（PostgreSQL/Redis），不要用 MemorySaver。

### Q: 流式输出在前端不显示？

**A:** 检查：
1. FastAPI 返回 `StreamingResponse`
2. 设置正确的 `media_type="text/event-stream"`
3. Nginx 关闭 `proxy_buffering`

---

## 调试

### Q: 怎么看每一步的状态？

**A:** 三种方式：
```python
# 1. stream updates
for event in graph.stream(input, stream_mode="updates"):
    print(event)

# 2. get_state
state = graph.get_state(config)
print(state.values)

# 3. LangSmith（推荐）
# 自动记录所有步骤
```

### Q: 怎么调试条件路由？

**A:** 在路由函数加日志：
```python
def router(state):
    decision = logic(state)
    print(f"[Router] → {decision}")
    return decision
```
