# 多模态 RAG

多模态 RAG 不仅能检索文本，还能处理图片、表格、PDF 中的图表等。

---

## 多模态检索策略

```
策略一: 文本检索（对图片生成描述）
策略二: 多向量检索（图片+文本分别嵌入）
策略三: 统一嵌入（多模态嵌入模型）
```

---

## 文本描述法

最简单的方案——用 LLM 为图片生成文本描述，然后按文本检索：

```python
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

def describe_image(image_path: str) -> str:
    """用 GPT-4o 描述图片"""
    import base64
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()

    response = llm.invoke([
        HumanMessage(content=[
            {"type": "text", "text": "详细描述这张图片的内容。"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
        ])
    ])
    return response.content

# 构建多模态知识库
def build_multimodal_kb(documents: list, images: list):
    """构建包含文本和图片的知识库"""
    all_chunks = []

    # 文本
    for doc in documents:
        all_chunks.append({"type": "text", "content": doc})

    # 图片描述
    for img_path in images:
        description = describe_image(img_path)
        all_chunks.append({
            "type": "image",
            "content": description,
            "source": img_path
        })

    # 向量化
    texts = [c["content"] for c in all_chunks]
    metadatas = [{"type": c["type"], "source": c.get("source", "")} for c in all_chunks]

    vector_store.add_texts(texts, metadatas=metadatas)
```

---

## 表格处理

```python
from langchain_community.document_loaders import PyPDFLoader

def extract_tables_from_pdf(pdf_path: str) -> list:
    """提取PDF中的表格"""
    import camelot  # pip install camelot-py

    tables = camelot.read_pdf(pdf_path, pages='all')
    return [t.df for t in tables]

def table_to_text(df) -> str:
    """将表格转为文本描述"""
    return df.to_markdown()
```

---

## 多模态 RAG 节点

```python
def multimodal_retrieve(state: State) -> dict:
    """检索包含文本和图片的结果"""
    docs = vector_store.similarity_search(state["question"], k=5)

    text_results = []
    image_results = []

    for doc in docs:
        if doc.metadata.get("type") == "image":
            image_results.append({
                "description": doc.page_content,
                "source": doc.metadata.get("source")
            })
        else:
            text_results.append(doc.page_content)

    return {"text_docs": text_results, "image_refs": image_results}

def multimodal_generate(state: State) -> dict:
    """生成包含图片引用的答案"""
    context = "\n".join(state["text_docs"])

    prompt_parts = [{"type": "text", "text": f"基于以下信息回答:\n{context}\n\n问题: {state['question']}"}]

    # 如果有相关图片，加入图片
    for img_ref in state.get("image_refs", []):
        prompt_parts.append({
            "type": "text",
            "text": f"\n相关图片: {img_ref['description']}"
        })

    response = llm.invoke([HumanMessage(content=prompt_parts)])
    return {"answer": response.content}
```

---

## 小结

多模态 RAG 扩展了知识库的维度，支持图片、表格等多种数据类型。

---

## 下一篇

➡️ 前往 [07-工具集成篇](../07-工具集成篇/01-工具定义与绑定.md)
