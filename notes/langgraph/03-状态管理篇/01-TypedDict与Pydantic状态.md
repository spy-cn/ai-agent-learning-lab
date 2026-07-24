# TypedDict 与 Pydantic 状态

State 是 LangGraph 的数据骨架。本篇对比两种主流的状态定义方式：TypedDict 和 Pydantic，帮你选择最适合的方案。

---

## 两种状态定义方式

### 方式一：TypedDict（官方推荐）

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    context: dict
    step_count: int
```

### 方式二：Pydantic BaseModel

```python
from pydantic import BaseModel, Field
from typing import Annotated

class AgentState(BaseModel):
    messages: Annotated[list, add_messages] = Field(default_factory=list)
    user_id: str = ""
    context: dict = Field(default_factory=dict)
    step_count: int = 0
```

---

## 对比总览

| 维度 | TypedDict | Pydantic |
|------|-----------|----------|
| **类型检查** | 仅静态检查（mypy） | 运行时校验 |
| **数据验证** | ❌ 无 | ✅ 字段类型、约束自动校验 |
| **默认值** | ❌ 不支持默认值 | ✅ 支持 |
| **序列化** | 需手动实现 | ✅ 内置 `.model_dump()` |
| **反序列化** | 需手动实现 | ✅ `ModelClass(**data)` |
| **文档生成** | 需手动 | ✅ JSON Schema 自动生成 |
| **性能** | 高（纯字典） | 中等（有校验开销） |
| **IDE 提示** | ✅ 好 | ✅ 好 |
| **学习曲线** | 低 | 中 |
| **官方推荐** | ✅ 文档主要使用 | 可选 |

---

## TypedDict 详解

### 基本用法

```python
from typing_extensions import TypedDict
from typing import Annotated
import operator

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
    retrieved_docs: Annotated[list, operator.add]
    question: str
    answer: str
```

### 使用 Required/NotRequired

```python
from typing_extensions import TypedDict, Required, NotRequired

class State(TypedDict, total=False):
    # total=False 表示所有字段都是可选的
    messages: Annotated[list, add_messages]  # 可选
    question: str                             # 可选
    # 但这种方式不够精确

# 更好的方式：使用 Required / NotRequired
class State(TypedDict):
    messages: Required[Annotated[list, add_messages]]  # 必须提供
    question: Required[str]                             # 必须提供
    context: NotRequired[dict]                          # 可选
    debug_info: NotRequired[dict]                       # 可选
```

### 继承

```python
class BaseChatState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str

class RAGState(BaseChatState):  # 继承
    retrieved_docs: list
    answer: str
    # 自动包含 messages 和 user_id
```

### 多状态合并

```python
# 定义不同模块的状态
class RetrievalState(TypedDict):
    query: str
    docs: Annotated[list, operator.add]

class GenerationState(TypedDict):
    draft: str
    final_answer: str

# 组合
class FullState(RetrievalState, GenerationState):
    messages: Annotated[list, add_messages]
    # 拥有所有字段：query, docs, draft, final_answer, messages
```

---

## Pydantic 详解

### 基本用法

```python
from pydantic import BaseModel, Field
from typing import Annotated
from langgraph.graph.message import add_messages

class AgentState(BaseModel):
    messages: Annotated[list, add_messages] = Field(default_factory=list)
    user_id: str = ""
    question: str = ""
    retrieved_docs: list = Field(default_factory=list)
    answer: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)  # 约束范围
```

### 数据验证

```python
from pydantic import field_validator

class State(BaseModel):
    messages: Annotated[list, add_messages] = Field(default_factory=list)
    age: int = Field(ge=0, le=150)  # 0-150之间
    email: str = ""

    @field_validator("email")
    @classmethod
    def validate_email(cls, v):
        if v and "@" not in v:
            raise ValueError("Invalid email")
        return v

# 自动校验
state = State(age=200)  # ❌ ValidationError: age must be <= 150
```

### 嵌套模型

```python
from pydantic import BaseModel
from typing import Optional

class Document(BaseModel):
    content: str
    score: float
    source: str

class UserInfo(BaseModel):
    user_id: str
    name: str
    preferences: dict = {}

class State(BaseModel):
    messages: Annotated[list, add_messages] = Field(default_factory=list)
    user: Optional[UserInfo] = None
    retrieved_docs: list[Document] = Field(default_factory=list)
```

### 序列化与反序列化

```python
# 序列化为字典
state_dict = state.model_dump()
# {'messages': [...], 'user_id': '123', ...}

# 序列化为 JSON
state_json = state.model_dump_json()

# 从字典创建
state = AgentState(**data_dict)

# JSON Schema
schema = AgentState.model_json_schema()
```

---

## 实战：RAG 应用状态设计

### TypedDict 版

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
import operator

class RAGState(TypedDict):
    """RAG 应用状态"""
    # 对话
    messages: Annotated[list, add_messages]

    # 检索
    question: str
    rewritten_query: str
    retrieved_docs: Annotated[list, operator.add]  # 多次检索累积
    doc_scores: Annotated[list[float], operator.add]

    # 生成
    context: str
    draft_answer: str
    final_answer: str

    # 元数据
    retrieval_count: int
    needs_refinement: bool
```

### Pydantic 版（带验证）

```python
from pydantic import BaseModel, Field, field_validator
from typing import Annotated
from langgraph.graph.message import add_messages
import operator

class RAGState(BaseModel):
    """RAG 应用状态（带数据验证）"""
    messages: Annotated[list, add_messages] = Field(default_factory=list)

    question: str = ""
    rewritten_query: str = ""
    retrieved_docs: Annotated[list, operator.add] = Field(default_factory=list)
    doc_scores: Annotated[list[float], operator.add] = Field(default_factory=list)

    context: str = ""
    draft_answer: str = ""
    final_answer: str = ""

    retrieval_count: int = Field(default=0, ge=0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("question")
    @classmethod
    def question_not_empty_when_set(cls, v):
        if v and len(v.strip()) == 0:
            raise ValueError("Question cannot be empty string")
        return v
```

---

## 在节点中使用

### TypedDict 在节点中

```python
def retrieve_node(state: RAGState) -> dict:
    # 直接通过键访问
    question = state["question"]
    docs = vector_store.search(question)

    return {
        "retrieved_docs": docs,      # 通过 reducer 追加
        "retrieval_count": state["retrieval_count"] + 1  # 手动累加
    }
```

### Pydantic 在节点中

```python
def retrieve_node(state: RAGState) -> dict:
    # 通过属性访问
    question = state.question
    docs = vector_store.search(question)

    return {
        "retrieved_docs": docs,
        "retrieval_count": state.retrieval_count + 1
    }
```

> 注意：无论用哪种定义方式，节点返回的都是普通 dict。

---

## 如何选择

### 选 TypedDict 如果你：

- ✅ 追求简单和性能
- ✅ 状态结构简单
- ✅ 不需要运行时验证
- ✅ 团队更熟悉字典操作
- ✅ 与 LangChain 生态高度集成

### 选 Pydantic 如果你：

- ✅ 需要严格的数据验证
- ✅ 状态结构复杂、嵌套深
- ✅ 需要序列化/反序列化
- ✅ 想生成 API 文档（JSON Schema）
- ✅ 团队已在使用 Pydantic

### 混合使用

也可以在 TypedDict 的字段中使用 Pydantic 模型：

```python
class Document(BaseModel):
    content: str
    score: float
    source: str = ""

class State(TypedDict):
    messages: Annotated[list, add_messages]
    docs: list[Document]     # Pydantic 模型作为字段类型
    user_query: str
```

两全其美：简单的顶层状态用 TypedDict，复杂的数据结构用 Pydantic。

---

## 最佳实践

### 1. 状态尽量精简

```python
# ❌ 不好：状态太大
class State(TypedDict):
    messages: Annotated[list, add_messages]
    user_profile: dict        # 大对象
    all_search_results: list  # 可能很大
    full_conversation_log: str

# ✅ 好：只保留必要字段
class State(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    answer: str
    relevant_docs: list       # 只保留相关的
```

### 2. 用 Reducer 管理列表

```python
# ❌ 不好：手动管理列表
def node(state):
    docs = state["docs"] + [new_doc]  # 手动拼接
    return {"docs": docs}

# ✅ 好：使用 reducer
class State(TypedDict):
    docs: Annotated[list, operator.add]

def node(state):
    return {"docs": [new_doc]}  # reducer 自动追加
```

### 3. 文档化你的状态

```python
class State(TypedDict):
    """主对话状态

    Attributes:
        messages: 对话历史（自动追加）
        question: 当前用户问题
        retrieved_docs: 检索到的文档（多次检索累积）
        answer: 最终答案
    """
    messages: Annotated[list, add_messages]
    question: str
    retrieved_docs: Annotated[list, operator.add]
    answer: str
```

---

## 小结

| 要点 | 说明 |
|------|------|
| TypedDict | 轻量、高性能、官方推荐 |
| Pydantic | 数据验证、序列化、Schema生成 |
| Annotated | 标注 reducer 的关键语法 |
| 选择原则 | 简单→TypedDict，严格→Pydantic |
| 混合使用 | TypedDict 顶层 + Pydantic 嵌套 |
| 精简原则 | 状态只保留必要字段 |

---

## 下一篇

➡️ [状态持久化与Checkpoint](./02-状态持久化与Checkpoint.md)
