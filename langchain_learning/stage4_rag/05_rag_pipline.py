import os
from dotenv import load_dotenv

# 1. 数据与切分
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 2. 向量与数据库
from langchain_community.embeddings import FastEmbedEmbeddings
from pymilvus import MilvusClient, DataType

# 3. Prompt 与 LLM 链构建
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# 加载环境变量 (OPENAI_API_KEY, OPENAI_BASE_URL 等)
load_dotenv()


# =====================================================================
# 模块一：数据准备 (Data Preparation)
# =====================================================================
def prepare_sample_data(filepath: str = "data/langchain_info.txt"):
    """步骤 1：生成并加载测试文档"""
    sample_text = """
LangChain 是一个用于开发大语言模型应用的开源框架。
LangChain 由 Harrison Chase 于 2022 年 10 月发起。
LangChain 的核心组件包括 Model I/O、Chains、Retrieval、Agents。
LCEL 是 LangChain 表达式语言，用管道符组合组件。
RAG 是检索增强生成，可以解决大模型知识滞后的问题。
Agent 是能自主决策和调用工具的智能体。
Milvus 是一个开源的向量数据库，适合存储和检索向量。
Embedding 模型把文本转换为向量，用于相似度计算。
文件中出现的人名是：小七；
""".strip()

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(sample_text)

    loader = TextLoader(filepath, encoding="utf-8")
    return loader.load()


# =====================================================================
# 模块二：索引构建 (Indexing Pipeline)
# =====================================================================
def build_index(docs, db_path: str = "data/embedding/milvus_rag.db", collection_name: str = "rag_collection"):
    """步骤 2-4：文本切分 -> 向量化 -> 存储至向量数据库"""
    print("=== 1. 开始索引阶段 ===")

    # 步骤 2：文本切分 (Chunking)
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", "，", ""],
        chunk_size=50,
        chunk_overlap=10,
    )
    chunks = splitter.split_documents(docs)
    print(f"-> 文档已切分为 {len(chunks)} 个 Text Chunks")

    # 步骤 3：初始化嵌入模型 (Embedding)
    embed_model = FastEmbedEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",  # 中文友好，体积小
        threads=os.cpu_count(),
    )

    # 步骤 4：初始化向量数据库 (Vector DB)
    client = MilvusClient(uri=db_path)

    # 如果存在同名集合先清理，重置数据库
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=512)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=512)

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="L2")
    client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)

    # 批量生成向量并插入数据库
    texts = [c.page_content for c in chunks]
    embeddings = embed_model.embed_documents(texts)
    data = [{"vector": e, "text": t} for e, t in zip(embeddings, texts)]
    client.insert(collection_name=collection_name, data=data)
    print(f"-> 已将 {len(data)} 条向量及其文本数据存入 Milvus\n")

    return client, embed_model


# =====================================================================
# 模块三：信息检索 (Retrieval Engine)
# =====================================================================
def create_retriever_function(client: MilvusClient, embed_model: FastEmbedEmbeddings,
                              collection_name: str = "rag_collection"):
    """步骤 5：构建向量检索函数"""

    def retrieve(query: str) -> str:
        # 将用户的输入问题转为向量
        query_emb = embed_model.embed_query(query)

        # 在 Milvus 中检索相似度最高的前 k 个文本块
        results = client.search(
            collection_name=collection_name,
            data=[query_emb],
            anns_field="vector",
            search_params={"metric_type": "L2"},
            output_fields=["text"],
            limit=2,
        )

        # 提取相关文本并组装为上下文字符串
        context = "\n".join([hit["entity"]["text"] for hit in results[0]])
        return context

    return retrieve


# =====================================================================
# 模块四：检索增强生成 (Generation & LCEL Chain)
# =====================================================================
def build_rag_chain(retrieve_fn):
    """步骤 6-7：构造 Prompt、初始化 LLM，并通过 LCEL 组合成 RAG 链"""
    print("=== 2. 组装 RAG 执行链 ===")

    MODEL_NAME = "deepseek-v4-flash"
    MODEL_URL = "https://llm-lniax9q66senyb7s.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    MODEL_KEY = "sk-ws-H.EHHMLPX.GItO.MEUCIQDOAzCACqmt9lyJ1RrfpNRVVu9sgD9QAZRJo8cXtUKE1wIgbdfCUdVgwKvGPDb-wwa8-lj4kGXtiTJHjTLnNwUniZg"

    # 初始化大语言模型
    llm = init_chat_model(
        model=MODEL_NAME,
        base_url=MODEL_URL,
        model_provider="openai",
        temperature=0,
        api_key=MODEL_KEY,
        streaming=True,
    )

    # 定义提示词模板
    template = ChatPromptTemplate.from_messages(
        [
            ("system", "根据以下参考信息回答问题。\n\n参考信息：\n{context}"),
            ("human", "{query}"),
        ]
    )

    # 使用 LCEL 表达式组合管道
    rag_chain = (
            {
                "query": RunnablePassthrough(),
                "context": RunnableLambda(lambda x: retrieve_fn(x)),
            }
            | template
            | llm
            | StrOutputParser()
    )

    return rag_chain


# =====================================================================
# 主入口：运行与测试
# =====================================================================
if __name__ == "__main__":
    # 1. 准备数据
    docs = prepare_sample_data()

    # 2. 构建索引
    client, embed_model = build_index(docs)

    # 3. 创建检索器
    retrieve_fn = create_retriever_function(client, embed_model)

    # 4. 构建 RAG 链
    rag_chain = build_rag_chain(retrieve_fn)

    # 5. 执行测试问答 (流式输出)
    questions = [
        "LangChain 是什么？",
        "RAG 解决什么问题？",
        "Milvus 是什么？",
        "文件中涉及到的人名是谁？"
    ]

    print("=== 3. 开始执行问答测试 ===")
    for q in questions:
        print(f"\n问：{q}")
        print("答：", end="")
        for chunk in rag_chain.stream(q):
            print(chunk, end="", flush=True)
        print()

    print("\n🎉 恭喜！RAG 管道运行完毕！")