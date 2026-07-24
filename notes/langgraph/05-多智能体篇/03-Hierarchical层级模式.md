# Hierarchical 层级模式

层级模式通过多级管理结构组织大量 Agent，适合复杂大型项目。

---

## 层级模式结构

```
               ┌──────────────────┐
               │  Top Supervisor  │  ← L1: 顶层决策
               │  (项目总监)       │
               └────────┬─────────┘
                        │
           ┌────────────┼────────────┐
           │            │            │
           ▼            ▼            ▼
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │ Research │ │ Dev Team │ │ QA Team  │  ← L2: 团队领导
    │ Lead     │ │ Lead     │ │ Lead     │
    └────┬─────┘ └────┬─────┘ └────┬─────┘
         │            │            │
    ┌────┴────┐  ┌────┴────┐  ┌────┴────┐
    │ │       │  │ │       │  │ │       │   ← L3: 执行者
    ▼ ▼       ▼  ▼ ▼       ▼  ▼ ▼       ▼
   R1 R2     R3  D1 D2     D3  Q1 Q2    Q3
```

---

## 实现层级模式

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import operator

class HierarchicalState(TypedDict):
    messages: Annotated[list, add_messages]
    project: str
    # 各团队的结果
    research_findings: Annotated[list, operator.add]
    dev_output: str
    qa_report: str
    final_deliverable: str

# === L1: 顶层 Supervisor ===
def top_supervisor(state: HierarchicalState) -> dict:
    """决定哪个团队工作"""
    # 分析当前状态，决定下一步
    if not state.get("research_findings"):
        return {"next_team": "research_lead"}
    elif not state.get("dev_output"):
        return {"next_team": "dev_lead"}
    elif not state.get("qa_report"):
        return {"next_team": "qa_lead"}
    else:
        return {"next_team": "FINISH"}

# === L2: 团队领导 ===
def research_lead(state: HierarchicalState) -> dict:
    """研究团队领导：分配子任务"""
    return {"research_findings": ["背景资料收集完成", "竞品分析完成"]}

def dev_lead(state: HierarchicalState) -> dict:
    """开发团队领导"""
    return {"dev_output": "代码开发完成"}

def qa_lead(state: HierarchicalState) -> dict:
    """QA团队领导"""
    return {"qa_report": "质量检查通过"}

# === 构建 ===
builder = StateGraph(HierarchicalState)
builder.add_node("top", top_supervisor)
builder.add_node("research_lead", research_lead)
builder.add_node("dev_lead", dev_lead)
builder.add_node("qa_lead", qa_lead)

builder.add_edge(START, "top")
builder.add_conditional_edges("top", lambda s: s.get("next_team", "FINISH"), {
    "research_lead": "research_lead",
    "dev_lead": "dev_lead",
    "qa_lead": "qa_lead",
    "FINISH": END,
})
for lead in ["research_lead", "dev_lead", "qa_lead"]:
    builder.add_edge(lead, "top")  # 回到顶层

graph = builder.compile()
```

## 使用 langgraph-supervisor 库

```python
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent

# L3: 执行 Agent
searcher = create_react_agent(llm, tools=[search_tool], name="searcher")
writer = create_react_agent(llm, tools=[], name="writer")

# L2: 研究团队 Supervisor
research_team = create_supervisor(
    llm=llm, agents=[searcher, writer], name="research_lead"
).compile()

# L2: 开发团队
coder = create_react_agent(llm, tools=[code_tool], name="coder")
tester = create_react_agent(llm, tools=[test_tool], name="tester")
dev_team = create_supervisor(
    llm=llm, agents=[coder, tester], name="dev_lead"
).compile()

# L1: 顶层 Supervisor
project = create_supervisor(
    llm=llm, agents=[research_team, dev_team]
).compile()
```

---

## 小结

层级模式适合 Agent 数量 >7 的场景，通过分层降低单层 Supervisor 的决策复杂度。

---

## 下一篇

➡️ [Swarm群智协作模式](./04-Swarm群智协作模式.md)
