# 实战：研究报告生成 Agent

构建一个能自主搜索、分析、撰写研究报告的 Agent。

---

## 架构

```
用户主题 → [分解子主题] → [并行搜索] → [综合分析]
                                        │
                                        ▼
                                   [撰写大纲]
                                        │
                                        ▼
                                   [生成报告]
                                        │
                                        ▼
                                   [质量审查] ──不满意──→ [修改]
                                        │
                                      满意
                                        ▼
                                   [最终报告]
```

---

## 实现

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.constants import Send
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
import operator

llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
search_tool = TavilySearch(max_results=3)


class ReportState(TypedDict):
    messages: Annotated[list, add_messages]
    topic: str
    subtopics: list[str]
    research_data: Annotated[list[str], operator.add]
    outline: str
    draft: str
    review_feedback: str
    quality_score: int
    iteration: int
    final_report: str

class WorkerState(TypedDict):
    subtopic: str


# === 1. 分解主题 ===
def decompose_topic(state: ReportState) -> dict:
    response = llm.invoke(
        f"将以下研究主题分解为3-5个子主题（返回JSON列表）:\n{state['topic']}"
    )
    import json
    try:
        subtopics = json.loads(response.content)
    except:
        subtopics = [state["topic"]]  # 降级处理
    return {"subtopics": subtopics}

# === 2. 并行搜索 ===
def fan_out(state: ReportState) -> list[Send]:
    return [Send("research", {"subtopic": s}) for s in state["subtopics"]]

def research(state: WorkerState) -> dict:
    """搜索单个子主题"""
    results = search_tool.invoke(state["subtopic"])
    summary = llm.invoke(f"总结以下搜索结果:\n{results}")
    return {"research_data": [f"[{state['subtopic']}]\n{summary.content}"]}

# === 3. 生成大纲 ===
def create_outline(state: ReportState) -> dict:
    data = "\n\n".join(state["research_data"])
    response = llm.invoke(
        f"基于以下研究数据创建报告大纲:\n{data}\n\n主题: {state['topic']}"
    )
    return {"outline": response.content}

# === 4. 撰写初稿 ===
def write_draft(state: ReportState) -> dict:
    response = llm.invoke(
        f"按大纲撰写完整研究报告:\n大纲: {state['outline']}\n\n数据:\n{state['research_data']}"
    )
    return {"draft": response.content, "iteration": state.get("iteration", 0) + 1}

# === 5. 审查 ===
def review(state: ReportState) -> dict:
    response = llm.invoke(
        f"审查这份研究报告的质量（1-10分）:\n{state['draft']}\n\n"
        f"返回格式: 分数|反馈意见"
    )
    import re
    numbers = re.findall(r'\d+', response.content)
    score = int(numbers[0]) if numbers else 7
    return {"quality_score": score, "review_feedback": response.content}

def should_revise(state: ReportState) -> str:
    if state["quality_score"] >= 8 or state["iteration"] >= 3:
        return "finalize"
    return "revise"

def revise(state: ReportState) -> dict:
    response = llm.invoke(
        f"根据反馈修改报告:\n原稿: {state['draft']}\n\n反馈: {state['review_feedback']}"
    )
    return {"draft": response.content}

def finalize(state: ReportState) -> dict:
    return {"final_report": state["draft"]}


# 构建
builder = StateGraph(ReportState)
builder.add_node("decompose", decompose_topic)
builder.add_node("research", research)
builder.add_node("outline", create_outline)
builder.add_node("write", write_draft)
builder.add_node("review", review)
builder.add_node("revise", revise)
builder.add_node("finalize", finalize)

builder.add_edge(START, "decompose")
builder.add_conditional_edges("decompose", fan_out)
builder.add_edge("research", "outline")
builder.add_edge("outline", "write")
builder.add_edge("write", "review")
builder.add_conditional_edges("review", should_revise, {
    "revise": "revise",
    "finalize": "finalize",
})
builder.add_edge("revise", "review")
builder.add_edge("finalize", END)

report_agent = builder.compile()
```

---

## 使用

```python
result = report_agent.invoke({
    "topic": "2025年AI Agent发展趋势",
    "messages": [],
    "subtopics": [],
    "research_data": [],
    "outline": "",
    "draft": "",
    "review_feedback": "",
    "quality_score": 0,
    "iteration": 0,
    "final_report": ""
})

print(result["final_report"])
```

---

## 小结

| 要点 | 说明 |
|------|------|
| 主题分解 | 大主题拆为子主题 |
| 并行搜索 | Send API 并行检索 |
| 大纲生成 | 结构化报告大纲 |
| 迭代修改 | 审查→修改→再审查 |
| 质量评分 | 客观评估标准 |

---

## 下一篇

➡️ [数据分析Agent](./04-数据分析Agent.md)
