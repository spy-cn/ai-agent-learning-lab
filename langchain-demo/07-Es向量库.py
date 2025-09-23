import logging
from typing import List

from elasticsearch import Elasticsearch
from langchain import requests
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.embeddings import Embeddings
from langchain_elasticsearch import ElasticsearchStore

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


from langchain_elasticsearch.client import create_elasticsearch_client
from langchain_elasticsearch.embeddings import ElasticsearchEmbeddings

ES_URL = "http://192.168.1.99:9200"
ES_USER = "elastic"
ES_PWD = "yourpassword"
MODEL_ID="" #你的模型在ES中的ID
# 创建ES客户端
es_client = create_elasticsearch_client(url=ES_URL, username=ES_USER, password=ES_PWD)

es_embeddings = ElasticsearchEmbeddings.from_es_connection(
    model_id=MODEL_ID,
    es_connection=es_client
)

documents = [
    "这是第一段文本。",
    "这是另一段需要生成向量的文本。"
]
vectors = es_embeddings.embed_documents(documents)
print(f"生成了 {len(vectors)} 个向量，每个维度为 {len(vectors[0])}")

# 连接到Elasticsearch
es = Elasticsearch(
    hosts=["http://192.168.1.99:9200"],
    basic_auth=("elastic", "yourpassword"),  # 如果有认证
    request_timeout=60
)

# 创建索引映射，使用cosine相似度
cosine_index_mapping = {
    "mappings": {
        "properties": {
            "title": {"type": "text", "analyzer": "standard"},
            "content": {"type": "text", "analyzer": "standard"},
            "source": {"type": "keyword"},
            "content_vector": {
                "type": "dense_vector",
                "dims": 1536,  # 根据嵌入模型调整维度
                "index": True,
                "similarity": "cosine"  # 余弦相似度
            }
        }
    },
    "settings": {
        "number_of_shards": 2,
        "number_of_replicas": 1
    }
}

# 创建使用余弦相似度的索引
# es.indices.create(index="rag-knowledge-cosine", body=cosine_index_mapping)

# 创建索引映射，使用dot_product相似度
dot_product_index_mapping = {
    "mappings": {
        "properties": {
            "title": {"type": "text", "analyzer": "standard"},
            "content": {"type": "text", "analyzer": "standard"},
            "source": {"type": "keyword"},
            "content_vector": {
                "type": "dense_vector",
                "dims": 1536,
                "index": True,
                "similarity": "dot_product"  # 点积相似度
            }
        }
    },
    "settings": {
        "number_of_shards": 2,
        "number_of_replicas": 1
    }
}

# 创建使用点积相似度的索引
# es.indices.create(index="rag-knowledge-dot-product", body=dot_product_index_mapping)

# 创建索引映射，使用l2_norm相似度
l2_norm_index_mapping = {
    "mappings": {
        "properties": {
            "title": {"type": "text", "analyzer": "standard"},
            "content": {"type": "text", "analyzer": "standard"},
            "source": {"type": "keyword"},
            "content_vector": {
                "type": "dense_vector",
                "dims": 1536,
                "index": True,
                "similarity": "l2_norm"  # 欧几里得距离（L2范数）
            }
        }
    },
    "settings": {
        "number_of_shards": 2,
        "number_of_replicas": 1
    }
}

# 创建使用L2范数相似度的索引
# es.indices.create(index="rag-knowledge-l2-norm", body=l2_norm_index_mapping)

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 加载文档
loader = PyPDFLoader(
    file_path="E:\code_project\self_project\langchain-project\data\病理专业医疗质量控制指标(2024 年版).pdf")
documents = loader.load()

# 文档分块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
split_docs = text_splitter.split_documents(documents)

# 初始化嵌入模型
EMBEDDING_API_KEY = "sk-82c55d978b6b4e16843c728766c25fd1"
embeddings = Qwen3Embeddings(api_key=EMBEDDING_API_KEY)

texts = [doc.page_content for doc in split_docs]

# 处理文档并索引到不同的Elasticsearch索引
for i, chunk in enumerate(split_docs):
    # 生成向量嵌入
    vector = embeddings.embed_documents([texts[i]])

    # 准备索引文档
    doc = {
        "title": f"Chunk {i} from {chunk.metadata.get('source', 'unknown')}",
        "content": chunk.page_content,
        "source": chunk.metadata.get("source", "unknown"),
        "content_vector": vector
    }

    # 索引文档到不同相似度计算方法的索引
    es.index(index="rag-knowledge-cosine", document=doc)

    # 对于dot_product，通常需要归一化向量
    es.index(index="rag-knowledge-dot-product", document=doc)

    # 对于l2_norm，直接使用原始向量
    es.index(index="rag-knowledge-l2-norm", document=doc)

# 刷新索引
es.indices.refresh(index="rag-knowledge-cosine")
es.indices.refresh(index="rag-knowledge-dot-product")
es.indices.refresh(index="rag-knowledge-l2-norm")

# 初始化不同相似度计算方法的Elasticsearch向量存储
es_store_cosine = ElasticsearchStore(
    es_url="http://192.168.1.99:9200",
    index_name="rag-knowledge-cosine",
    embedding=embeddings,
    es_user="elastic",  # 如果有认证
    es_password="yourpassword"  # 如果有认证
)

es_store_dot_product = ElasticsearchStore(
    es_url="http://192.168.1.99:9200",
    index_name="rag-knowledge-dot-product",
    embedding=embeddings,
    es_user="elastic",
    es_password="yourpassword"
)

es_store_l2_norm = ElasticsearchStore(
    es_url="http://192.168.1.99:9200",
    index_name="rag-knowledge-l2-norm",
    embedding=embeddings,
    es_user="elastic",
    es_password="yourpassword"
)

# 创建不同相似度计算方法的检索器
retriever_cosine = es_store_cosine.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

retriever_dot_product = es_store_dot_product.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

retriever_l2_norm = es_store_l2_norm.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)


# 创建多相似度混合检索器
def multi_similarity_retriever(query, top_k=5):
    # 从三种不同相似度方法获取结果
    cosine_docs = retriever_cosine.get_relevant_documents(query)
    dot_product_docs = retriever_dot_product.get_relevant_documents(query)
    l2_norm_docs = retriever_l2_norm.get_relevant_documents(query)

    # 合并结果
    all_docs = cosine_docs + dot_product_docs + l2_norm_docs

    # 去重
    unique_docs = list({doc.page_content: doc for doc in all_docs}.values())

    # 限制返回数量
    return unique_docs[:top_k]


# 定义提示模板
template = """
你是一个专业的AI助手。请基于以下提供的上下文信息，回答用户的问题。
如果你无法从上下文中找到答案，请直接说"我无法从提供的信息中找到答案"，不要编造信息。
上下文信息:
{context}
用户问题: {question}
回答:
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)


# 格式化文档函数
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])
