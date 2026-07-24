# Corrective-RAG 纠正式检索

Corrective-RAG（CRAG）在标准 RAG 基础上加入**纠正机制**——当检索质量不好时，自动转向 Web 搜索补充。

---

## CRAG 流程

```
问题 → 检索 → [评估文档质量]
                    │
            ┌───────┼───────┐
            │       │       │
          高质量   中等     低质量
            │       │       │
            ▼       ▼       ▼
        [直接生成] [精炼]  [Web搜索]
                    │       │
                    ▼       ▼
              [内部+Web合并生成]
```

---

## 实现

```python
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
import operator

llm = ChatOpenAI(model="gpt-4o", temperature=0)

class CRAGState(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    local_docs: list
    doc_assessment: str  # high, medium, low
    web_results: str
    answer: str


def retrieve(state: CRAGState) -> dict:
    docs = vector_store.similarity_search(state["question"], k=3)
    return {"local_docs": docs}

class DocAssessment(BaseModel):
    quality: Literal["high", "medium", "low"]

def assess_docs(state: CRAGState) -> dict:
    """评估检索文档整体质量"""
    if not state["local_docs"]:
        return {"doc_assessment": "low"}

    assessor = llm.with_structured_output(DocAssessment)
    result = assessor.invoke(
        f"评估这些文档对回答问题的整体质量:\n"
        f"问题: {state['question']}\n"
        f"文档数: {len(state['local_docs'])}\n"
        f"预览: {[d.page_content[:100] for d in state['local_docs']]}"
    )
    return {"doc_assessment": result.quality}

def route_by_quality(state: CRAGState) -> str:
    quality = state["doc_assessment"]
    if quality == "high":
        return "generate"
    elif quality == "medium":
        return "refine"  # 精炼后生成
    else:
        return "web_search"  # 直接转Web

def web_search(state: CRAGState) -> dict:
    from langchain_tavily import TavilySearch
    search = TavilySearch(max_results=3)
    results = search.invoke(state["question"])
    return {"web_results": results}

def refine_docs(state: CRAGState) -> dict:
    """精炼文档：去掉不相关部分"""
    refined = []
    for doc in state["local_docs"]:
        # 只保留相关段落
        refined.append(doc.page_content[:500])  # 简化
    return {"local_docs": refined}

def generate(state: CRAGState) -> dict:
    # 合并本地和Web结果
    context_parts = []

    if state.get("local_docs"):
        context_parts.append("知识库: " + "\n".join(state["local_docs"]))

    if state.get("web_results"):
        context_parts.append("Web搜索: " + state["web_results"])

    context = "\n\n".join(context_parts)

    response = llm.invoke(
        f"基于以下信息回答:\n{context}\n\n问题: {state['question']}"
    )
    return {"answer": response.content, "messages": [response]}


# 构建
builder = StateGraph(CRAGState)
builder.add_node("retrieve", retrieve)
builder.add_node("assess", assess_docs)
builder.add_node("web", web_search)
builder.add_node("refine", refine_docs)
builder.add_node("generate", generate)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "assess")
builder.add_conditional_edges("assess", route_by_quality, {
    "generate": "generate",
    "refine": "refine",
    "web_search": "web",
})
builder.add_edge("refine", "generate")
builder.add_edge("web", "generate")
builder.add_edge("generate", END)

crag = builder.compile()
```

---

## 小结

CRAG 的核心是**评估 → 路由 → 纠正**——根据检索质量动态选择生成策略。

---

## 下一篇

➡️ [多模态RAG](./05-多模态RAG.md)
