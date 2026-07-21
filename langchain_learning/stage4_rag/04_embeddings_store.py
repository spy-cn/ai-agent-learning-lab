"""
Stage4 RAG - VectorStore 多引擎对比 Demo
========================================
对比四种常见向量数据库：
1. Milvus（单机嵌入式 / 轻量）
2. FAISS（纯本地、无服务）
3. Chroma（轻量、LangChain 亲儿子）
4. Qdrant（功能强、API 现代）

适用环境：
- 无 GPU / 无显存
- CPU 优先
- 本地开发 / 学习 / PoC
"""

import os
from typing import List

from pymilvus import DataType, MilvusClient
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS, Chroma

# ========================
# 1. 全局配置
# ========================

# ✅ 使用 FastEmbed
# - 纯 CPU 推理
# - ONNX 加速
# - 中文友好
embedding_model = FastEmbedEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",  # 输出维度：512
    threads=os.cpu_count(),  # 自动利用多核 CPU
)

# 测试文本数据
texts: List[str] = [
    "Python 是一种流行的编程语言",
    "Java 也是一种广泛使用的编程语言",
    "今天天气很好，适合出门散步",
    "机器学习需要大量数据来训练模型",
]


# ========================
# 2. Milvus Demo（嵌入式模式）
# ========================

def run_milvus_demo():
    """
    Milvus 嵌入式模式（Standalone Embedded）

    特点：
    - 无需 Docker / 服务
    - 数据存本地 SQLite + 向量文件
    - 适合本地开发 / 学习

    注意：
    - vector 维度必须与 embedding 模型一致
    - bge-small-zh-v1.5 → 512 维
    """
    print("\n=== 1. Milvus（嵌入式模式）===")

    db_path = "./milvus_demo.db"

    # 2. 初始化 Client
    client = MilvusClient(uri=db_path)
    collection_name = "demo_collection"

    # 定义 Schema
    schema = MilvusClient.create_schema(
        auto_id=True,
        enable_dynamic_field=True,
    )
    schema.add_field(
        field_name="id",
        datatype=DataType.INT64,
        is_primary=True,
    )
    schema.add_field(
        field_name="vector",
        datatype=DataType.FLOAT_VECTOR,
        dim=512,  # ✅ 必须与 embedding 模型输出维度一致
    )
    schema.add_field(
        field_name="text",
        datatype=DataType.VARCHAR,
        max_length=1024,
    )

    # 索引参数（AUTOINDEX 适合入门）
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="AUTOINDEX",
        metric_type="L2",  # 欧氏距离
    )

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )

    # 向量化并插入
    doc_embeddings = embedding_model.embed_documents(texts)
    data = [
        {"vector": emb, "text": text}
        for emb, text in zip(doc_embeddings, texts)
    ]
    client.insert(collection_name=collection_name, data=data)
    print(f"✅ Milvus 成功插入 {len(data)} 条数据")

    # 向量检索
    query = "什么语言适合编程"
    query_vector = embedding_model.embed_query(query)

    results = client.search(
        collection_name=collection_name,
        data=[query_vector],
        anns_field="vector",
        search_params={"metric_type": "L2"},
        output_fields=["text"],
        limit=2,
    )

    print(f"🔍 查询：'{query}'")
    for hit in results[0]:
        print(f"  - {hit['entity']['text']} (距离: {hit['distance']:.4f})")


# ========================
# 3. FAISS Demo（纯本地）
# ========================

def run_faiss_demo():
    """
    FAISS（Facebook AI Similarity Search）

    特点：
    - 无任何服务进程
    - 文件即数据库
    - 启动最快
    - LangChain 最常用

    缺点：
    - 不支持元数据过滤（原生）
    - 不适合大规模分布式
    """
    print("\n=== 2.1 FAISS（纯本地）===")

    # ✅ LangChain 一行完成：文本 → embedding → FAISS
    vectorstore = FAISS.from_texts(
        texts=texts,
        embedding=embedding_model,
    )

    query = "什么语言适合编程"
    results = vectorstore.similarity_search_with_score(query, k=2)

    print(f"🔍 查询：'{query}'")
    for doc, score in results:
        print(f"  - {doc.page_content} (L2 距离: {score:.4f})")


# ========================
# 4. Chroma Demo（LangChain 亲儿子）
# ========================

def run_chroma_demo():
    """
    Chroma

    特点：
    - LangChain 官方推荐
    - 支持持久化
    - 支持元数据过滤
    - 上手最简单

    适合：
    - 学习 RAG
    - 本地 PoC
    """
    print("\n=== 2.2 Chroma（持久化）===")

    persist_dir = "./chroma_db"

    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embedding_model,
        persist_directory=persist_dir,
    )

    query = "什么语言适合编程"
    results = vectorstore.similarity_search_with_score(query, k=2)

    print(f"🔍 查询：'{query}'")
    for doc, score in results:
        print(f"  - {doc.page_content} (距离: {score:.4f})")


# ========================
# 5. Qdrant Demo（功能最强）
# ========================

def run_qdrant_demo():
    """
    Qdrant

    特点：
    - Rust 编写，性能好
    - 支持 COSINE / EUCLID
    - 支持复杂过滤
    - API 设计现代

    适合：
    - 生产环境
    - 多条件检索
    """
    print("\n=== 3. Qdrant（本地磁盘模式）===")

    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, PointStruct, VectorParams

    client = QdrantClient(path="./qdrant_db")
    collection_name = "demo_collection"

    # 重建 Collection
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=512,  # ✅ 与 embedding 模型一致
            distance=Distance.COSINE,  # ✅ 余弦相似度（推荐）
        ),
    )

    # 插入数据
    doc_embeddings = embedding_model.embed_documents(texts)
    points = [
        PointStruct(
            id=idx,
            vector=emb,
            payload={"text": text},  # 元数据
        )
        for idx, (emb, text) in enumerate(zip(doc_embeddings, texts))
    ]
    client.upsert(collection_name=collection_name, points=points)
    print(f"✅ Qdrant 成功插入 {len(points)} 条数据")

    # 检索
    query = "什么语言适合编程"
    query_vector = embedding_model.embed_query(query)

    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=2,
    ).points

    print(f"🔍 查询：'{query}'")
    for hit in results:
        print(f"  - {hit.payload['text']} (相似度得分: {hit.score:.4f})")


# ========================
# 6. 主入口
# ========================

if __name__ == "__main__":
    run_milvus_demo()
    run_faiss_demo()
    run_chroma_demo()
    run_qdrant_demo()
