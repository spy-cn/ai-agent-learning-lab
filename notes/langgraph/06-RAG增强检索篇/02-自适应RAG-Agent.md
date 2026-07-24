# 自适应 RAG Agent

自适应 RAG 能根据问题类型和检索质量动态调整策略——不是所有问题都需要检索，也不是所有检索结果都有用。

---

## 自适应策略

```
用户提问
    │
    ▼
┌─────────────┐
│ 查询分析     │
└──────┬──────┘
       │
       ├── 简单事实 → [直接回答] → END
       │
       ├── 需要知识 → [本地知识库检索]
       │                  │
       │            ┌─────┴─────┐
       │            │           │
       │         找到文档    未找到
       │            │           │
       │            ▼           ▼
       │        [生成答案]  [Web搜索]
       │                        │
       │                        ▼
       │                   [生成答案]
       │
       └── 需要最新 → [Web搜索] → [生成答案]
```

---

## 完整实现

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel
from typing import Literal
import operator

llm = ChatOpenAI(model="gpt-4o", temperature=0)
embeddings = OpenAIEmbeddings()
vector_store = Chroma(embedding_function=embeddings, collection_name="docs")


class AdaptiveRAGState(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    query_type: str  # simple, knowledge, web_search
    local_docs: list
    web_results: str
    answer: str


# === 查询路由 ===
class RouteDecision(BaseModel):
    route: Literal["simple", "knowledge", "web_search"]

def analyze_query(state: AdaptiveRAGState) -> dict:
    """分析查询类型"""
    decision_llm = llm.with_structured_output(RouteDecision)

    decision = decision_llm.invoke(
        f"分析以下问题的类型:\n"
        f"- simple: 简单常识问题，LLM可直接回答\n"
        f"- knowledge: 需要专业知识库\n"
        f"- web_search: 需要最新信息\n\n"
        f"问题: {state['question']}"
    )
    return {"query_type": decision.route}

def route_query(state: AdaptiveRAGState) -> str:
    return state["query_type"]


# === 直接回答 ===
def direct_answer(state: AdaptiveRAGState) -> dict:
    response = llm.invoke(state["question"])
    return {"answer": response.content, "messages": [response]}


# === 知识库检索 ===
def local_retrieve(state: AdaptiveRAGState) -> dict:
    docs = vector_store.similarity_search(state["question"], k=3)
    return {"local_docs": docs}

def check_docs_quality(state: AdaptiveRAGState) -> str:
    """检查检索质量"""
    if not state["local_docs"]:
        return "web_search"  # 没找到，转Web搜索
    return "generate"

def generate_from_local(state: AdaptiveRAGState) -> dict:
    context = "\n".join(d.page_content for d in state["local_docs"])
    response = llm.invoke(f"基于以下信息回答:\n{context}\n\n问题: {state['question']}")
    return {"answer": response.content, "messages": [response]}


# === Web 搜索 ===
def web_search(state: AdaptiveRAGState) -> dict:
    from langchain_tavily import TavilySearch
    search = TavilySearch(max_results=3)
    results = search.invoke(state["question"])
    return {"web_results": results}

def generate_from_web(state: AdaptiveRAGState) -> dict:
    response = llm.invoke(f"基于以下搜索结果回答:\n{state['web_results']}\n\n问题: {state['question']}")
    return {"answer": response.content, "messages": [response]}


# === 构建图 ===
builder = StateGraph(AdaptiveRAGState)

builder.add_node("analyze", analyze_query)
builder.add_node("direct", direct_answer)
builder.add_node("local_retrieve", local_retrieve)
builder.add_node("local_generate", generate_from_local)
builder.add_node("web_search", web_search)
builder.add_node("web_generate", generate_from_web)

builder.add_edge(START, "analyze")
builder.add_conditional_edges("analyze", route_query, {
    "simple": "direct",
    "knowledge": "local_retrieve",
    "web_search": "web_search",
})
builder.add_edge("direct", END)

builder.add_conditional_edges("local_retrieve", check_docs_quality, {
    "generate": "local_generate",
    "web_search": "web_search",  # 本地没找到 → 转Web
})
builder.add_edge("local_generate", END)

builder.add_edge("web_search", "web_generate")
builder.add_edge("web_generate", END)

rag_agent = builder.compile()
```

### 图结构

```
                 ┌──► direct ──────────► END
                 │
START → analyze ─┼──► local_retrieve ──┬──► local_generate ──► END
                 │                      │
                 │                (没找到)│
                 │                      ▼
                 └──► web_search ──► web_generate ──► END
```

---

## 小结

自适应 RAG 的核心是**根据情况选择不同策略**，而非对所有问题用同一套流程。

---

## 下一篇

➡️ [Self-RAG自我评估](./03-Self-RAG自我评估.md)
