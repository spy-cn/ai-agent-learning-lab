import logging
from typing import List, Dict, Any

import requests
from langchain.embeddings.base import Embeddings
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_openai import ChatOpenAI
from pymilvus import MilvusClient, connections, DataType

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Qwen3Embeddings(Embeddings):
    """自定义通义千问嵌入模型类"""

    def __init__(self, api_key: str, api_url: str = "http://192.168.1.99:8565/embeddings"):
        self.api_key = api_key
        self.url = api_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.batch_size = 25
        self._dimension = None  # 动态获取维度

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """调用API获取嵌入向量"""
        try:
            payload = {"texts": texts}
            response = requests.post(self.url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            embeddings = result.get("embeddings", [])

            # 首次调用时确定向量维度
            if self._dimension is None and embeddings:
                self._dimension = len(embeddings[0])
                logger.info(f"Detected embedding dimension: {self._dimension}")

            return embeddings
        except Exception as e:
            logger.error(f"Error calling Tongyi Embedding API: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """对文档列表进行嵌入"""
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self._call_api(batch_texts)
            all_embeddings.extend(batch_embeddings)
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """对查询文本进行嵌入"""
        embeddings = self._call_api([text])
        return embeddings[0] if embeddings else []

    @property
    def dimension(self) -> int:
        """获取嵌入向量维度"""
        if self._dimension is None:
            # 如果没有调用过API，先获取一个示例向量的维度
            test_embedding = self.embed_query("test")
            self._dimension = len(test_embedding)
        return self._dimension


class MilvusVectorStore:
    """Milvus向量存储管理类"""

    def __init__(self, host: str, port: str, collection_name: str, dimension: int = None):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dimension = dimension
        self.connection_uri = f"http://{host}:{port}"

        # 连接Milvus
        connections.connect(alias="default", host=host, port=port)
        self.client = MilvusClient(uri=self.connection_uri)

    def create_collection_if_not_exists(self, dimension: int):
        """创建集合（如果不存在）"""
        if not self.client.has_collection(self.collection_name):
            # 定义集合schema
            schema = self.client.create_schema(
                auto_id=True,
                enable_dynamic_field=True
            )

            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
            schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dimension)
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)

            # 创建集合
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                index_type="IVF_FLAT",
                metric_type="L2",
                params={"nlist": 1024}
            )

            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params
            )
            logger.info(f"Created collection: {self.collection_name} with dimension: {dimension}")
        else:
            # 检查现有集合的维度是否匹配
            collection_info = self.client.describe_collection(self.collection_name)
            existing_dim = None
            for field in collection_info.get('fields', []):
                if field.get('name') == 'vector' and field.get('type') == DataType.FLOAT_VECTOR:
                    existing_dim = field.get('params', {}).get('dim')
                    break

            if existing_dim != dimension:
                logger.warning(
                    f"Existing collection dimension {existing_dim} doesn't match required dimension {dimension}")
                logger.info("Dropping and recreating collection...")
                self.client.drop_collection(self.collection_name)
                self.create_collection_if_not_exists(dimension)
            else:
                logger.info(f"Collection {self.collection_name} already exists with correct dimension: {dimension}")

    def validate_embedding_dimension(self, embedding: List[float], expected_dim: int) -> bool:
        """验证向量维度"""
        actual_dim = len(embedding)
        if actual_dim != expected_dim:
            logger.error(f"Embedding dimension mismatch: expected {expected_dim}, got {actual_dim}")
            return False
        return True

    def drop_documents(self):
        """删除集合"""
        self.client.drop_collection(self.collection_name)
        print(f"已删除集合：{self.collection_name}")

    def insert_documents(self, documents: List[Document], embeddings: List[List[float]]):
        """插入文档到Milvus"""
        texts = [doc.page_content for doc in documents]
        metadata_list = [doc.metadata for doc in documents]

        # 准备插入数据
        data = []
        for i, (emb, text, metadata) in enumerate(zip(embeddings, texts, metadata_list)):
            # 验证向量维度
            if not self.validate_embedding_dimension(emb, self.dimension):
                continue

            data.append({
                "vector": emb,
                "text": text,
                "metadata": metadata
            })

        # 批量插入
        batch_size = 100
        total_inserted = 0
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i + batch_size]
            try:
                insert_result = self.client.insert(
                    collection_name=self.collection_name,
                    data=batch_data
                )
                total_inserted += len(batch_data)
                logger.info(f"Inserted {len(batch_data)} documents, total: {total_inserted}")
            except Exception as e:
                logger.error(f"Error inserting batch {i // batch_size + 1}: {e}")

        return total_inserted

        def search_similar(self, query_embedding: List[float], limit: int = 5):
            """搜索相似文档"""
        # 验证查询向量维度
        if not self.validate_embedding_dimension(query_embedding, self.dimension):
            return []

        search_params = {
            "metric_type": "L2",  # 必须与集合定义一致
            "params": {"nprobe": 32}  # 可以适当增大
        }

        # 添加调试日志
        logger.debug(f"Searching with params: {search_params}")
        logger.debug(f"Query embedding sample: {query_embedding[:5]}...")

        try:
            search_results = self.client.search(
                collection_name=self.collection_name,
                data=[query_embedding],
                limit=limit,
                output_fields=["text", "metadata"],
                search_params=search_params
            )
            return search_results
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []


class RAGSystem:
    """检索增强生成系统"""

    def __init__(self, embeddings: Qwen3Embeddings, milvus_host: str, milvus_port: str, collection_name: str,
                 llm: ChatOpenAI):
        self.embeddings = embeddings

        # 动态获取嵌入维度
        embedding_dimension = embeddings.dimension
        logger.info(f"Using embedding dimension: {embedding_dimension}")

        # 初始化Milvus
        self.vector_store = MilvusVectorStore(
            host=milvus_host,
            port=milvus_port,
            collection_name=collection_name,
            dimension=embedding_dimension
        )
        self.vector_store.create_collection_if_not_exists(embedding_dimension)

        self.llm = llm

        # 创建提示模板
        self.prompt_template = """基于以下上下文信息，请回答用户的问题。如果上下文中的信息不足以回答问题，请如实告知。

上下文信息：
{context}

问题：{question}

请提供准确、简洁且专业的回答："""

        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )

    def index_documents(self, file_path: str):
        """索引文档"""
        try:
            #loader = TextLoader(file_path, encoding="utf-8")
            loader = PyPDFLoader(file_path)
            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,  # 尝试更小的分块
                chunk_overlap=300,
                separators=["\n\n", "\n", "。", "；", " ", ""]  # 添加中文分句符号
            )
            split_docs = text_splitter.split_documents(documents)

            # 生成嵌入向量
            texts = [doc.page_content for doc in split_docs]
            logger.info(f"Generating embeddings for {len(texts)} documents...")
            embeddings_list = self.embeddings.embed_documents(texts)

            # 插入文档到Milvus
            total_inserted = self.vector_store.insert_documents(split_docs, embeddings_list)
            logger.info(f"Successfully indexed {total_inserted} documents")

        except Exception as e:
            logger.error(f"Document indexing failed: {e}")
            raise

    def retrieve_documents(self, query: str, limit: int = 5) -> List[Document]:
        """检索相关文档"""
        query_embedding = self.embeddings.embed_query(query)
        search_results = self.vector_store.search_similar(query_embedding, limit)

        documents = []
        if search_results:
            for result in search_results[0]:
                doc = Document(
                    page_content=result['entity'].get('text', ''),
                    metadata=result['entity'].get('metadata', {})
                )
                documents.append(doc)

        return documents

    def generate_answer(self, query: str, context: str) -> str:
        """生成答案"""
        prompt = self.prompt.format(context=context, question=query)

        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "抱歉，生成答案时出现错误。"

    def query(self, question: str, limit: int = 5) -> Dict[str, Any]:
        """完整的RAG查询流程"""
        # 检索相关文档
        relevant_docs = self.retrieve_documents(question, limit)

        if not relevant_docs:
            return {
                "question": question,
                "answer": "没有找到相关的文档信息。",
                "source_documents": [],
                "context": ""
            }

        # 组合上下文
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # 生成答案
        answer = self.generate_answer(question, context)

        return {
            "question": question,
            "answer": answer,
            "source_documents": relevant_docs,
            "context": context
        }


def main():
    # 配置参数
    MILVUS_HOST = "192.168.1.99"
    MILVUS_PORT = "19530"
    COLLECTION_NAME = "qc_collection"
    EMBEDDING_API_KEY = "sk-82c55d978b6b4e16843c728766c25fd1"
    LLM_API_KEY = "lanbigdata-key-1234"
    LLM_API_BASE = "http://192.168.1.99:8562/v1"
    DATA_FILE_PATH = "E:\code_project\self_project\langchain-project\data\病理专业医疗质量控制指标(2024 年版).pdf"

    # 1. 初始化嵌入模型
    embeddings = Qwen3Embeddings(api_key=EMBEDDING_API_KEY)

    # 2. 初始化LLM
    llm = ChatOpenAI(
        model_name="Qwen3-32B",
        openai_api_key=LLM_API_KEY,
        openai_api_base=LLM_API_BASE,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        temperature=0.1
    )

    # 3. 初始化RAG系统
    rag_system = RAGSystem(
        embeddings=embeddings,
        milvus_host=MILVUS_HOST,
        milvus_port=MILVUS_PORT,
        collection_name=COLLECTION_NAME,
        llm=llm
    )

    # 4. 索引文档
    try:
        rag_system.index_documents(DATA_FILE_PATH)
    except Exception as e:
        logger.error(f"Failed to index documents: {e}")
        # 即使索引失败，也可以继续使用系统（如果已有数据）

    # 5. 示例查询
    questions = [
        "什么是分子病理室间质评合格率。",
        "什么是每百张床位病理医师数"
    ]

    for question in questions:
        print(f"\n{'=' * 50}")
        print(f"问题: {question}")
        result = rag_system.query(question)
        print(f"答案: {result['answer']}")
        print(f"参考文档数量: {len(result['source_documents'])}")


if __name__ == "__main__":
    main()