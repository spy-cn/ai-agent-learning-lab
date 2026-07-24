# Human-in-the-Loop 详解

本篇深入讲解 LangGraph 中 Human-in-the-Loop（HITL）的实现细节和高级模式。

> 基础的 HITL 概念已在 [04-高级控制篇](../04-高级控制篇/03-人机回路Human-in-the-Loop.md) 中介绍，本篇聚焦于交互设计和前端集成。

---

## HITL 设计原则

```
1. 最小中断 — 只在必要时暂停
2. 清晰呈现 — 让人类知道要审什么
3. 低操作成本 — 批准/拒绝应一键完成
4. 容错处理 — 超时和拒绝的优雅降级
5. 审计追踪 — 记录所有人的操作
```

---

## 动态中断策略

### 风险评估驱动

```python
from pydantic import BaseModel
from typing import Literal

class RiskAssessment(BaseModel):
    level: Literal["low", "medium", "high"]
    reason: str

def assess_risk(state: State) -> dict:
    """评估操作风险"""
    assessor = llm.with_structured_output(RiskAssessment)
    result = assessor.invoke(
        f"评估以下操作的风险等级:\n{state['pending_action']}"
    )
    return {
        "risk_level": result.level,
        "risk_reason": result.reason
    }

def route_by_risk(state: State) -> str:
    risk = state.get("risk_level", "medium")
    if risk == "high":
        return "require_approval"
    elif risk == "medium":
        return "optional_approval"
    else:
        return "auto_execute"
```

### 金额/范围阈值

```python
def check_amount_threshold(state: State) -> str:
    """金额超阈值需要审批"""
    amount = state.get("transaction_amount", 0)

    if amount > 10000:
        return "manager_approval"
    elif amount > 1000:
        return "supervisor_approval"
    else:
        return "auto_approve"
```

---

## 多级审批链

```python
"""
三级审批链: 直属经理 → 部门总监 → CFO
"""
def level_1_approval(state: State) -> dict:
    """直属经理审批"""
    return {"approval_level": 1}

def route_after_l1(state: State) -> str:
    if not state.get("approved"):
        return "rejected"
    if state.get("amount", 0) > 10000:
        return "level_2"  # 继续上报
    return "execute"  # 批准

def level_2_approval(state: State) -> dict:
    """部门总监审批"""
    return {"approval_level": 2}

def route_after_l2(state: State) -> str:
    if not state.get("approved"):
        return "rejected"
    if state.get("amount", 0) > 100000:
        return "level_3"
    return "execute"

# 构建多级审批图
builder.add_node("l1", level_1_approval)
builder.add_node("l2", level_2_approval)
builder.add_node("l3", level_3_approval)
builder.add_node("execute", execute_action)
builder.add_node("rejected", reject_action)

builder.add_conditional_edges("l1", route_after_l1, {
    "level_2": "l2",
    "execute": "execute",
    "rejected": "rejected",
})
```

---

## 会话超时处理

```python
import asyncio
from datetime import datetime, timedelta

async def hitl_with_timeout(graph, config, timeout_seconds=300):
    """带超时的 HITL 等待"""
    try:
        result = await asyncio.wait_for(
            wait_for_human(config),
            timeout=timeout_seconds
        )

        if result.get("approved"):
            graph.update_state(config, {"approved": True})
        else:
            graph.update_state(config, {
                "approved": False,
                "rejection_reason": result.get("reason", "")
            })

        return graph.invoke(None, config)

    except asyncio.TimeoutError:
        # 超时处理
        graph.update_state(config, {
            "status": "timeout",
            "approved": False,
            "message": "审批超时，操作已取消"
        })
        return graph.invoke(None, config)
```

---

## 批量审批

```python
def batch_approval_node(state: State) -> dict:
    """批量审批多个操作"""
    pending = state.get("pending_actions", [])

    # 将所有操作展示给用户
    batch_display = format_batch(pending)

    return {
        "batch_display": batch_display,
        "awaiting_batch_approval": True
    }

# 在中断后，人类可以一次性批准/拒绝多个
def process_batch_approval(state: State) -> dict:
    decisions = state.get("batch_decisions", {})

    approved = []
    rejected = []

    for action in state["pending_actions"]:
        action_id = action["id"]
        if decisions.get(action_id, False):
            approved.append(action)
        else:
            rejected.append(action)

    return {"approved_actions": approved, "rejected_actions": rejected}
```

---

## 小结

| 要点 | 说明 |
|------|------|
| 动态中断 | 根据风险/金额决定是否暂停 |
| 多级审批 | 按金额/范围逐级上报 |
| 超时处理 | 防止无限等待 |
| 批量审批 | 一次处理多个操作 |
| 审计日志 | 记录所有人工决策 |

---

## 下一篇

➡️ [交互式对话设计](./02-交互式对话设计.md)
