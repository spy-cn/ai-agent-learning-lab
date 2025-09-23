import logging
import os
from typing import List, Dict, Any

import requests
from elasticsearch import Elasticsearch
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.embeddings import Embeddings
from langchain_elasticsearch import ElasticsearchStore
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 配置日志
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Qwen3Embeddings(Embeddings):
    """自定义通义千问嵌入模型类"""

    def __init__(self, api_key: str, api_url: str = "http://192.168.1.99:8565/embeddings", batch_size: int = 25):
        self.api_key = api_key
        self.url = api_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.batch_size = batch_size
        self._dimension = None  # 动态获取维度

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """调用API获取嵌入向量"""
        try:
            payload = {"texts": texts}
            print("pyload====json:", payload)
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
        if not texts:
            return []

        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self._call_api(batch_texts)
            all_embeddings.extend(batch_embeddings)
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """对查询文本进行嵌入"""
        if not text.strip():
            raise ValueError("查询文本不能为空!")
        print("要查询的文本：", text)
        embeddings = self._call_api([text])
        print("查询结果：", embeddings)
        return embeddings[0] if embeddings else []

    @property
    def dimension(self) -> int:
        """获取嵌入向量维度"""
        if self._dimension is None:
            # 如果没有调用过API，先获取一个示例向量的维度
            test_embedding = self.embed_query("test")
            self._dimension = len(test_embedding)
        return self._dimension


class EsVectorStore:
    """ES向量存储管理类"""

    def __init__(self, host: str, port: str, user: str, pwd: str, index_name: str, dimension: int = None):
        self.host = host
        self.port = port
        self.index_name = index_name
        self.dimension = dimension
        self.connection_uri = f"http://{host}:{port}"
        try:
            self.client = Elasticsearch(
                hosts=[self.connection_uri],
                basic_auth=(user, pwd),
                request_timeout=60
            )
            if not self.client.ping():
                raise ConnectionError("连接ES失败!")
        except Exception as e:
            logger.error(f"ES 连接失败：{e}")

    def create_collection_if_not_exists(self, dimension: int, similarity: str = "cosine", shards: int = 1,
                                        replicas: int = 1):
        """如果向量库不存在就创建"""
        if not self.client.indices.exists(index=self.index_name):
            # 创建索引映射，使用cosine相似度
            cosine_index_mapping = {
                "mappings": {
                    "dynamic": "strict",# 禁止未定义的字段
                    "properties": {
                        "content": {"type": "text", "analyzer": "standard"},
                        "source": {"type": "keyword"},
                        "vector": {
                            "type": "dense_vector",
                            "dims": dimension,
                            "index": True,
                            "similarity": similarity
                        }
                    }
                },
                "settings": {
                    "number_of_shards": shards,
                    "number_of_replicas": replicas
                }
            }
            # 创建相似度索引
            self.client.indices.create(index=self.index_name, body=cosine_index_mapping)
            logger.info(f"Created new index: {self.index_name} with dimension {dimension}")

    def clear_index(self):
        """清空索引"""
        try:
            self.client.delete_by_query(index=self.index_name, body={"query": {"match_all": {}}})
            logger.info("索引清除成功!")
        except Exception as e:
            logger.error(f"清除索引失败：{e}")
            raise

    def insert_documents(self, documents: List[Document], embeddings: List[List[float]]):
        """插入文档到向量库"""
        if len(documents) != len(embeddings):
            raise ValueError("文档和嵌入必须具有相同的长度！")

        bulk_operations = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            bulk_operations.append({
                "index": {
                    "_index": self.index_name,
                    "_id": f"doc_{i}_{hash(doc.page_content)}"  # 生成唯一ID
                }
            })
            bulk_operations.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "vector": embedding
            })

        try:
            response = self.client.bulk(operations=bulk_operations, refresh=True)
            if response["errors"]:
                logger.error(f"插入文档错误: {response['errors']}")
            else:
                logger.info(f"成功插入 {len(documents)} 个文档")
            return response
        except Exception as e:
            logger.error(f"插入文档错误: {e}")
            raise

    def search_similar(self, query_embedding: List[float], limit: int = 5):
        """搜索相似文档"""
        print(query_embedding)

        query = {
            "knn": {
                "field": "vector",
                "query_vector": query_embedding,
                "k": limit,
                "num_candidates": 100
            }
        }

        try:
            response = self.client.search(index=self.index_name, body=query)
            hits = response["hits"]["hits"]
            documents = []
            for hit in hits:
                doc = Document(
                    page_content=hit["_source"]["content"],
                    metadata={
                        "source": hit["_source"].get("source", "unknown"),
                        "score": hit["_score"],
                        "id": hit["_id"]
                    }
                )
                documents.append(doc)
            logger.info(f"发现 {len(documents)}个相似文档")
            return documents
        except Exception as e:
            logger.error(f"查询相似文档错误: {e}")
            raise


class RAGSystem:
    """RAG系统类"""

    def __init__(
            self,
            es_host: str,
            es_port: str,
            es_user: str,
            es_pwd: str,
            es_index: str,
            embedding_model: Qwen3Embeddings,
            llm: Any,
            similarity: str = "cosine"
    ):
        self.embedding_model = embedding_model
        self.llm = llm
        self.es_vector_store = EsVectorStore(
            host=es_host,
            port=es_port,
            user=es_user,
            pwd=es_pwd,
            index_name=es_index,
            dimension=embedding_model.dimension
        )

        # 创建索引（如果不存在）
        self.es_vector_store.create_collection_if_not_exists(
            dimension=embedding_model.dimension,
            similarity=similarity
        )

        # 初始化ElasticsearchStore
        self.vector_store = ElasticsearchStore(
            embedding=self.embedding_model,
            es_url=f"http://{es_host}:{es_port}",
            index_name=es_index,
            es_user=es_user,
            es_password=es_pwd,
            strategy=ElasticsearchStore.ExactRetrievalStrategy()
        )

        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

        # 初始化检索器
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})

        # 定义提示模板
        self.template = """基于以下提供的上下文信息，请回答用户的问题。如果上下文中的信息不足以回答问题，请直接说明"根据已知信息无法回答该问题"，不要编造信息。
                        上下文信息：
                        {context}
                        用户问题：{question}
                        请根据上下文信息提供准确、有用的回答：
                        """

        self.prompt = PromptTemplate(
            template=self.template,
            input_variables=["context", "question"]
        )

    def add_documents(self, file_path: str):
        """添加文档到向量库"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"没有发现文件: {file_path}")

        self.es_vector_store.client.delete_by_query(index=self.es_vector_store.index_name,
                                                    body={"query": {"match_all": {}}})
        try:
            # 加载文档
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            logger.info(f"从PDF中加载了 {len(documents)} 页")
            # 分割文本
            split_documents = self.text_splitter.split_documents(documents)
            logger.info(f"将文档分割为 {len(split_documents)} 块")
            # 生成嵌入
            texts = [doc.page_content for doc in split_documents]
            logger.info("正在构建向量...")
            embeddings = self.embedding_model.embed_documents(texts)
            # 插入到ES
            self.es_vector_store.insert_documents(documents, embeddings)
            logger.info("文档添加成功")
        except Exception as e:
            logger.error(f"文档添加失败: {e}")
            raise

    def query(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """查询RAG系统"""
        if not question.strip():
            return {"answer": "问题不能为空", "source_documents": []}
        try:
            # 首先获取要查询问题的向量
            query_embedding = self.embedding_model.embed_query(question)
            # 获取相似的文档内容
            similar_docs = self.es_vector_store.search_similar(query_embedding, top_k)
            if not similar_docs:
                return {
                    "answer": "未找到相关文档信息",
                    "source_documents": [],
                    "query_embedding": query_embedding
                }
            # 组合上下文
            context = "\n\n".join([doc.page_content for doc in similar_docs])
            # 生成答案
            answer = self.generate_answer(question, context)

            return {
                "answer": answer,
                "source_documents": similar_docs,
                "query_embedding": query_embedding
            }
        except Exception as e:
            logger.error(f"查询错误：{e}")
            return {
                "answer": "系统处理查询时出现错误",
                "source_documents": [],
                "query_embedding": []
            }

    def generate_answer(self, query: str, context: str) -> str:
        """生成答案"""
        prompt = self.prompt.format(context=context, question=query)

        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"生成答案时出现错误: {e}")
            return "抱歉，生成答案时出现错误。"


def main():
    LLM_API_KEY = "lanbigdata-key-1234"
    LLM_API_BASE = "http://192.168.1.99:8562/v1"
    DATA_FILE_PATH = "E:\code_project\self_project\langchain-project\data\病理专业医疗质量控制指标(2024 年版).pdf"
    # 初始化组件
    logger.info("正在初始化向量模型...")
    embedding_model = Qwen3Embeddings(api_key="your-api-key")
    logger.info("正在初始大语言量模型...")
    llm = ChatOpenAI(
        model_name="Qwen3-32B",
        openai_api_key=LLM_API_KEY,
        openai_api_base=LLM_API_BASE,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        temperature=0.1
    )  # 初始化你的LLM，比如ChatOpenAI或其它兼容LangChain的模型
    logger.info("正在初始化RAG系统...")
    # 创建RAG系统
    rag_system = RAGSystem(
        es_host="192.168.1.99",
        es_port="9200",
        es_user="elastic",
        es_pwd="yourpassword",
        es_index="knowledge_base",
        embedding_model=embedding_model,
        llm=llm
    )

    # 添加文档
    logger.info("正在添加文档到索引...")
    rag_system.add_documents(DATA_FILE_PATH)

    # 测试查询
    test_questions = [
        "每百张床位病理医师数怎么计算？",
        "什么是每百张床位病理医师数？",
        "怎么计算术中快速诊断与石蜡诊断符合率？",
        "哪个指标可以反应反映术中快速诊断准确性"
    ]

    for question in test_questions:
        logger.info(f"Query: {question}")
        result = rag_system.query(question)
        print(f"\n问题: {question}")
        print(f"回答: {result['answer']}")
        print(f"来源文档数量: {len(result['source_documents'])}")
        print("-" * 50)


if __name__ == "__main__":
    main()
