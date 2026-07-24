# Self-RAG 自我评估

Self-RAG 让 Agent 具备**自我反思**能力——评估检索质量、判断答案是否需要修正。

---

## Self-RAG 流程

```
问题 → 检索 → [评估文档] → 不相关 → [改写查询] → 重新检索
                  │
                相关
                  │
                  ▼
             生成答案
                  │
                  ▼
           [评估答案]
                  │
         ┌────────┼────────┐
         │        │        │
       有依据    部分依据   无依据
         │        │        │
         ▼        ▼        ▼
       输出    [修正]    [重新检索]
```

---

## 实现

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import Literal
import operator

llm = ChatOpenAI(model="gpt-4o", temperature=0)

class SelfRAGState(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    docs: list
    filtered_docs: list
    draft: str
    generation_attempts: int
    final_answer: str


# === 1. 检索 ===
def retrieve(state: SelfRAGState) -> dict:
    docs = vector_store.similarity_search(state["question"], k=5)
    return {"docs": docs}

# === 2. 文档评估 ===
class DocGrade(BaseModel):
    score: Literal["relevant", "irrelevant"]

def grade_documents(state: SelfRAGState) -> dict:
    """评估每个文档的相关性"""
    relevant = []
    for doc in state["docs"]:
        grader = llm.with_structured_output(DocGrade)
        result = grader.invoke(
            f"文档是否与问题相关?\n问题: {state['question']}\n文档: {doc.page_content[:300]}"
        )
        if result.score == "relevant":
            relevant.append(doc)
    return {"filtered_docs": relevant}

# === 3. 决策 ===
def decide_to_generate(state: SelfRAGState) -> str:
    if not state["filtered_docs"]:
        return "rewrite_query"
    return "generate"

# === 4. 查询改写 ===
def rewrite_query(state: SelfRAGState) -> dict:
    response = llm.invoke(
        f"改写以下问题以获得更好的检索结果:\n{state['question']}"
    )
    return {"question": response.content}

# === 5. 生成 ===
def generate(state: SelfRAGState) -> dict:
    context = "\n\n".join(d.page_content for d in state["filtered_docs"])
    response = llm.invoke(
        f"基于以下信息回答:\n{context}\n\n问题: {state['question']}"
    )
    return {"draft": response.content, "generation_attempts": state.get("generation_attempts", 0) + 1}

# === 6. 答案质量评估 ===
class AnswerGrade(BaseModel):
    score: Literal["grounded", "not_grounded"]
    feedback: str

def grade_answer(state: SelfRAGState) -> dict:
    grader = llm.with_structured_output(AnswerGrade)
    result = grader.invoke(
        f"答案是否有文档支持?\n问题: {state['question']}\n文档: {state['filtered_docs']}\n答案: {state['draft']}"
    )
    return {"answer_grade": result.score, "feedback": result.feedback}

def decide_final(state: SelfRAGState) -> str:
    grade = state.get("answer_grade")
    attempts = state.get("generation_attempts", 0)
    if grade == "grounded" or attempts >= 3:
        return "final"
    return "rewrite"  # 不满意，重新来

def finalize(state: SelfRAGState) -> dict:
    return {"final_answer": state["draft"]}


# === 构建 ===
builder = StateGraph(SelfRAGState)
builder.add_node("retrieve", retrieve)
builder.add_node("grade_docs", grade_documents)
builder.add_node("rewrite", rewrite_query)
builder.add_node("generate", generate)
builder.add_node("grade_answer", grade_answer)
builder.add_node("finalize", finalize)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "grade_docs")
builder.add_conditional_edges("grade_docs", decide_to_generate, {
    "generate": "generate",
    "rewrite_query": "rewrite",
})
builder.add_edge("rewrite", "retrieve")  # 重新检索
builder.add_edge("generate", "grade_answer")
builder.add_conditional_edges("grade_answer", decide_final, {
    "final": "finalize",
    "rewrite": "rewrite",
})
builder.add_edge("finalize", END)

self_rag = builder.compile()
```

---

## 小结

Self-RAG 通过**文档评估 + 答案评估 + 查询改写**的循环，确保答案有据可依。

---

## 下一篇

➡️ [Corrective-RAG纠正式检索](./04-Corrective-RAG纠正式检索.md)
