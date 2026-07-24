# 人机回路 Human-in-the-Loop

Human-in-the-Loop（HITL）是 LangGraph 的杀手级特性——让人类在 Agent 执行过程中介入，实现审批、修正、指导。

---

## 为什么需要人机回路

### 纯自动 Agent 的问题

```
用户: "帮我发一封邮件给老板，说我下周请假"

Agent自动执行:
  1. 理解意图 → 写请假邮件
  2. 获取老板邮箱
  3. 发送邮件 ← ❌ 直接发送了！

问题: 邮件内容可能有误、语气可能不当、收件人可能错误
```

### 有 HITL 的流程

```
用户: "帮我发邮件请假"

Agent执行:
  1. 理解意图 → 写邮件草稿
  2. [暂停] 等待人类审批 ← ✅
     ┌───────────────────────┐
     │ 📧 邮件草稿预览          │
     │ 收件人: boss@company.com │
     │ 主题: 请假申请            │
     │ 内容: ...               │
     │                         │
     │ [✅ 批准发送] [✏️ 修改]  │
     └───────────────────────┘
  3. 人类批准 → 发送
     人类修改 → 重新确认 → 发送
```

---

## HITL 的核心机制

### interrupt_before / interrupt_after

```python
graph = builder.compile(
    checkpointer=MemorySaver(),          # 必须有 checkpointer
    interrupt_before=["approval_node"],  # 在指定节点前暂停
    interrupt_after=["draft_node"],      # 在指定节点后暂停
)
```

### 执行流程

```
graph.invoke(input, config)
    │
    ▼
执行 Node A → 执行 Node B → [到达 interrupt 节点] → 暂停
                                                           │
                    ┌──────────────────────────────────────┘
                    │
                    ▼
              graph.get_state(config)  ← 检查当前状态
                    │
                    ▼
              人类审查 / 修改
                    │
                    ▼
              graph.update_state(config,修改)  ← 可选：修改状态
                    │
                    ▼
              graph.invoke(None, config)  ← 恢复执行
                    │
                    ▼
              继续执行后续节点 → END
```

---

## HITL 的三种模式

### 模式一：审批（Approve/Reject）

```python
"""
在执行敏感操作前请求人类批准
"""
builder.add_node("prepare_action", prepare)
builder.add_node("execute_action", execute)

# 在执行前暂停
graph = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["execute_action"]
)

config = {"configurable": {"thread_id": "approval-demo"}}

# 执行到 execute_action 前暂停
result = graph.invoke(input, config=config)

# 检查待执行的操作
state = graph.get_state(config)
print("即将执行:")
print(state.values.get("pending_action"))

# 人类决定
human_approved = True  # 实际中从 UI 获取

if human_approved:
    # 批准 → 继续
    result = graph.invoke(None, config=config)
else:
    # 拒绝 → 修改状态后跳过
    graph.update_state(config, {"skip_action": True})
    result = graph.invoke(None, config=config)
```

### 模式二：编辑（Edit State）

```python
"""
人类审查并修改 Agent 的输出
"""
def generate_email(state: State) -> dict:
    draft = llm.invoke("写一封商务邮件...")
    return {"email_draft": draft.content}

graph = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_after=["generate_email"]  # 生成后暂停
)

config = {"configurable": {"thread_id": "edit-demo"}}
result = graph.invoke(input, config=config)

# 获取 AI 生成的草稿
state = graph.get_state(config)
ai_draft = state.values["email_draft"]
print(f"AI草稿:\n{ai_draft}")

# 人类修改
human_edited = ai_draft.replace("尊敬的", "亲爱的")  # 简化示例
human_edited += "\n\n此致\n敬礼"

# 更新状态
graph.update_state(config, {"email_draft": human_edited})

# 继续执行（发送等）
result = graph.invoke(None, config=config)
```

### 模式三：引导（Provide Input）

```python
"""
Agent 需要额外信息时向人类提问
"""
def check_info_needed(state: State) -> dict:
    """检查是否有足够的信息"""
    if not state.get("recipient_email"):
        return {"need_input": "recipient_email"}
    return {"need_input": None}

def request_human_input(state: State) -> dict:
    """请求人类输入"""
    # 这个节点只是标记暂停
    return {}

# 构建
builder.add_node("check", check_info_needed)
builder.add_node("request_input", request_human_input)

builder.add_edge(START, "check")
builder.add_conditional_edges("check", lambda s: "request_input" if s["need_input"] else "proceed")
builder.add_edge("request_input", END)  # 暂停后结束本轮

graph = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["request_input"]
)

# 第一次执行：暂停在 request_input 前
config = {"configurable": {"thread_id": "input-demo"}}
result = graph.invoke({"task": "send_email"}, config=config)

# 检查需要什么信息
state = graph.get_state(config)
needed = state.values.get("need_input")
print(f"需要提供: {needed}")

# 人类提供信息
graph.update_state(config, {"recipient_email": "boss@company.com"})

# 继续
result = graph.invoke(None, config=config)
```

---

## 完整示例：带审批的邮件 Agent

```python
"""
完整的 HITL 邮件发送 Agent
"""
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage


class State(TypedDict):
    messages: Annotated[list, add_messages]
    recipient: str
    subject: str
    email_body: str
    approved: bool
    sent: bool


llm = ChatOpenAI(model="gpt-4o")


def draft_email(state: State) -> dict:
    """生成邮件草稿"""
    user_request = state["messages"][-1].content

    response = llm.invoke(
        f"根据用户请求撰写邮件。请求: {user_request}\n"
        f"返回格式: 收件人|主题|正文"
    )

    parts = response.content.split("|")
    return {
        "recipient": parts[0].strip() if len(parts) > 0 else "",
        "subject": parts[1].strip() if len(parts) > 1 else "",
        "email_body": parts[2].strip() if len(parts) > 2 else "",
    }


def wait_for_approval(state: State) -> dict:
    """等待批准（会被 interrupt 暂停）"""
    return {}


def send_email(state: State) -> dict:
    """发送邮件"""
    if state.get("approved"):
        print(f"📧 发送邮件到: {state['recipient']}")
        print(f"📋 主题: {state['subject']}")
        print(f"📝 正文: {state['email_body']}")
        return {"sent": True}
    else:
        print("❌ 邮件未发送（未批准）")
        return {"sent": False}


# 构建图
builder = StateGraph(State)
builder.add_node("draft", draft_email)
builder.add_node("approve", wait_for_approval)
builder.add_node("send", send_email)

builder.add_edge(START, "draft")
builder.add_edge("draft", "approve")
builder.add_edge("approve", "send")
builder.add_edge("send", END)

# 编译：在 approve 节点前暂停
graph = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["approve"]
)


# === 运行演示 ===
config = {"configurable": {"thread_id": "email-1"}}

print("=== 第一步：生成草稿 ===")
result = graph.invoke(
    {"messages": [HumanMessage("给张经理发邮件，请假三天")], "approved": False},
    config=config
)

# 查看草稿
state = graph.get_state(config)
print(f"\n收件人: {state.values.get('recipient')}")
print(f"主题: {state.values.get('subject')}")
print(f"正文: {state.values.get('email_body')}")
print(f"\n下一步: {state.next}")  # ['approve']

print("\n=== 第二步：人工审查 ===")
# 模拟人类修改
graph.update_state(config, {
    "email_body": state.values["email_body"] + "\n\n谢谢批准！"
})

# 批准
graph.update_state(config, {"approved": True})

print("\n=== 第三步：发送 ===")
result = graph.invoke(None, config=config)
print(f"发送状态: {result.get('sent')}")
```

---

## 动态中断

有时你不想在固定节点暂停，而是根据条件决定：

```python
def maybe_interrupt(state: State) -> dict:
    """根据风险等级决定是否需要人工审批"""
    risk = assess_risk(state)

    if risk == "high":
        # 高风险：必须人工审批
        return {"requires_approval": True}
    elif risk == "medium":
        # 中风险：可选审批
        return {"requires_approval": state.get("user_prefers_review", False)}
    else:
        # 低风险：直接执行
        return {"requires_approval": False}


def route_by_risk(state: State) -> str:
    if state.get("requires_approval"):
        return "approval_flow"
    return "auto_execute"
```

---

## HITL 在 Web 应用中的实现

### FastAPI 集成

```python
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()
graphs = {}  # thread_id → graph state


@app.post("/chat")
async def chat(message: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}

    result = graph.invoke(
        {"messages": [HumanMessage(message)]},
        config=config
    )

    state = graph.get_state(config)

    # 如果暂停在审批节点
    if "approve" in (state.next or []):
        return JSONResponse({
            "status": "awaiting_approval",
            "draft": state.values.get("email_body"),
            "thread_id": thread_id
        })

    return JSONResponse({
        "status": "completed",
        "response": result["messages"][-1].content
    })


@app.post("/approve")
async def approve(thread_id: str, approved: bool, edited_content: str = None):
    config = {"configurable": {"thread_id": thread_id}}

    if edited_content:
        graph.update_state(config, {"email_body": edited_content})

    graph.update_state(config, {"approved": approved})

    result = graph.invoke(None, config=config)

    return {"status": "done", "sent": result.get("sent")}
```

---

## HITL 最佳实践

### 1. 清晰的暂停点

```python
# ✅ 好：在关键操作前暂停
interrupt_before=["send_email", "delete_file", "make_payment"]

# ❌ 不好：在无关节点暂停
interrupt_before=["parse_input", "format_output"]
```

### 2. 提供足够上下文

```python
def prepare_for_review(state: State) -> dict:
    """为人类审查准备信息"""
    return {
        "review_context": {
            "action": "发送邮件",
            "recipient": state["recipient"],
            "risk_level": "high",
            "reason": "收件人是外部客户",
            "preview": state["email_body"][:200]
        }
    }
```

### 3. 超时处理

```python
import asyncio

async def hitl_with_timeout(graph, config, timeout=300):
    """带超时的 HITL"""
    try:
        # 等待人类响应，最多 timeout 秒
        result = await asyncio.wait_for(
            wait_for_human_input(),
            timeout=timeout
        )
        graph.update_state(config, result)
    except asyncio.TimeoutError:
        graph.update_state(config, {"status": "timeout", "approved": False})
```

### 4. 审计日志

```python
def log_human_action(action: str, details: dict):
    """记录人类的操作"""
    import logging
    logger = logging.getLogger("hitl_audit")
    logger.info(f"Human action: {action}, Details: {details}")
```

---

## 小结

| HITL 模式 | 场景 | 实现 |
|-----------|------|------|
| 审批 | 敏感操作前确认 | interrupt_before |
| 编辑 | 修改 AI 输出 | update_state |
| 引导 | 请求额外信息 | interrupt + 更新状态 |
| 动态中断 | 条件性暂停 | 风险评估路由 |

| 要点 | 说明 |
|------|------|
| 必须 Checkpointer | HITL 依赖状态持久化 |
| interrupt_before | 节点执行前暂停 |
| interrupt_after | 节点执行后暂停 |
| get_state | 检查暂停时的状态 |
| update_state | 修改状态后恢复 |
| invoke(None) | 传 None 表示从暂停处继续 |

---

## 下一篇

➡️ [并行执行与扇出扇入](./04-并行执行与扇出扇入.md)
