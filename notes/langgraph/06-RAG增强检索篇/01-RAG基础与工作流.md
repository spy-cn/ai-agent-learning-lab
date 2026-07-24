# RAG 基础与工作流

RAG（Retrieval-Augmented Generation，检索增强生成）让 LLM 能基于外部知识回答问题，减少幻觉。

---

## RAG 核心流程

```
用户提问 → [检索] → [增强] → [生成]
              │         │         │
              ▼         ▼         ▼
          向量搜索   拼接上下文   LLM回答
          关键词     Prompt模板
          混合检索
```

### 基本流程

```python
用户: "LangGraph 支持哪些持久化后端？"

1. 检索: 向量数据库搜索 "LangGraph persistence"
   → [doc1: MemorySaver, doc2: PostgresSaver, doc3: Redis]

2. 增强: 构建 Prompt
   → "基于以下信息回答: [doc1+doc2+doc3] 问题: ..."

3. 生成: LLM 回答
   → "LangGraph 支持 MemorySaver(内存), PostgresSaver(生产), RedisSaver(高并发)"
```

---

## 用 LangGraph 构建 RAG

### 基本架构

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage
import operator

class RAGState(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    retrieved_docs: list
    answer: str

llm = ChatOpenAI(model="gpt-4o")
embeddings = OpenAIEmbeddings()
vector_store = Chroma(embedding_function=embeddings, collection_name="docs")


# === 1. 检索节点 ===
def retrieve(state: RAGState) -> dict:
    """从向量数据库检索相关文档"""
    docs = vector_store.similarity_search(state["question"], k=3)
    return {"retrieved_docs": docs}

# === 2. 生成节点 ===
def generate(state: RAGState) -> dict:
    """基于检索结果生成答案"""
    # 拼接上下文
    context = "\n\n".join(doc.page_content for doc in state["retrieved_docs"])

    # 构建 Prompt
    messages = [
        SystemMessage(f"你是助手。基于以下参考信息回答问题。如果信息不足，请说明。\n\n参考信息:\n{context}"),
        HumanMessage(state["question"])
    ]

    response = llm.invoke(messages)
    return {"answer": response.content, "messages": [response]}

# === 3. 构建图 ===
builder = StateGraph(RAGState)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)

rag_graph = builder.compile()
```

---

## 文档加载与向量化

### 构建知识库

```python
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    WebBaseLoader,
    DirectoryLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. 加载文档
loader = DirectoryLoader("./docs", glob="**/*.md", loader_cls=TextLoader)
docs = loader.load()

# 2. 分割
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(docs)

# 3. 向量化并存储
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(),
    collection_name="knowledge_base"
)
```

---

## 进阶：带查询分析的 RAG

```python
class AdvancedRAGState(TypedDict):
    messages: Annotated[list, add_messages]
    original_question: str
    analyzed_query: str
    retrieved_docs: list
    answer: str
    confidence: float

def analyze_query(state: AdvancedRAGState) -> dict:
    """分析并改写查询以提高检索质量"""
    response = llm.invoke(
        f"改写以下问题为更适合检索的查询（提取关键词）:\n{state['original_question']}"
    )
    return {"analyzed_query": response.content}

def retrieve(state: AdvancedRAGState) -> dict:
    docs = vector_store.similarity_search(state["analyzed_query"], k=5)
    return {"retrieved_docs": docs}

def grade_docs(state: AdvancedRAGState) -> dict:
    """评估检索文档的相关性"""
    relevant = []
    for doc in state["retrieved_docs"]:
        grade = llm.invoke(
            f"评估文档是否与问题相关(相关/不相关):\n问题: {state['original_question']}\n文档: {doc.page_content[:200]}"
        )
        if "相关" in grade.content:
            relevant.append(doc)
    return {"retrieved_docs": relevant}

def generate(state: AdvancedRAGState) -> dict:
    if not state["retrieved_docs"]:
        return {"answer": "抱歉，没有找到相关信息。", "confidence": 0.0}

    context = "\n\n".join(d.page_content for d in state["retrieved_docs"])
    response = llm.invoke(f"基于以下信息回答:\n{context}\n\n问题: {state['original_question']}")
    return {"answer": response.content, "confidence": 0.8}

builder = StateGraph(AdvancedRAGState)
builder.add_node("analyze", analyze_query)
builder.add_node("retrieve", retrieve)
builder.add_node("grade", grade_docs)
builder.add_node("generate", generate)

builder.add_edge(START, "analyze")
builder.add_edge("analyze", "retrieve")
builder.add_edge("retrieve", "grade")
builder.add_edge("grade", "generate")
builder.add_edge("generate", END)

rag = builder.compile()
```

---

## 小结

| 要点 | 说明 |
|------|------|
| RAG = 检索 + 增强 + 生成 | 核心三步流程 |
| 向量数据库 | 存储和检索文档 |
| 文档分割 | chunk_size + overlap |
| 查询分析 | 改写查询提升检索质量 |
| 文档评分 | 过滤不相关文档 |

---

## 下一篇

➡️ [自适应RAG-Agent](./02-自适应RAG-Agent.md)
