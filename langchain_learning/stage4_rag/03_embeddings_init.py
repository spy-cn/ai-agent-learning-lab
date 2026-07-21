import os
from typing import Literal
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

load_dotenv()

# 阿里云 MaaS OpenAI 兼容接口配置
MODEL_NAME = os.getenv("MODEL_NAME", "qwen3.7-text-embedding")  # ✅ 修正为真实存在的模型
MODEL_URL = os.getenv(
    "MODEL_URL",
    "https://llm-lniax9q66senyb7s.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
)
MODEL_KEY = os.getenv("MODEL_KEY",
                      "sk-ws-H.EHHMLPX.GItO.MEUCIQDOAzCACqmt9lyJ1RrfpNRVVu9sgD9QAZRJo8cXtUKE1wIgbdfCUdVgwKvGPDb-wwa8-lj4kGXtiTJHjTLnNwUniZg")

# ========================
# Embedding 初始化方式一：FastEmbed（⭐⭐⭐⭐⭐ 强烈推荐 CPU）
# ========================
def init_embedding_model_by_fast() -> Embeddings:
    """
    【CPU / 无显存首选】FastEmbed + ONNX Runtime

    优势：
    - 纯 CPU 推理，速度快（比 sentence-transformers 快 3~10 倍）
    - 自动下载模型，无需 PyTorch
    - 内存占用低，适合 RAG / 本地检索

    依赖：
    pip install fastembed

    适用模型：
    - BAAI/bge-small-zh-v1.5（中文推荐）
    - sentence-transformers/all-MiniLM-L6-v2
    """
    from langchain_community.embeddings import FastEmbedEmbeddings

    embeddings = FastEmbedEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",  # 中文友好，体积小
        #cache_dir="./models/fastembed",        # cache_dir 缓存模型，避免重复下载
        threads=os.cpu_count(),                # ✅ 自动利用多核 CPU
    )
    return embeddings


# ========================
# Embedding 初始化方式二：HuggingFace Sentence-Transformers
# ========================
def init_embedding_model_by_huggingface(device: Literal["cpu", "cuda"] = "cpu") -> Embeddings:
    """
    【PyTorch 生态】HuggingFace Transformers

    优势：
    - 模型最全
    - 可切换 GPU / CPU
    - 适合研究和定制

    劣势：
    - 启动慢
    - 内存占用高
    - 无 GPU 时推理较慢

    依赖：
    pip install sentence-transformers torch
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={
            "device": device,          # cpu / cuda
        },
        encode_kwargs={
            "normalize_embeddings": True,  # ✅ 余弦相似度必须归一化
        },
    )
    return embeddings


# ========================
# Embedding 初始化方式三：OpenAI / 阿里云 MaaS 兼容接口
# ========================
def init_embedding_model_by_openai() -> Embeddings:
    """
    【云端方案】OpenAI / 阿里云 MaaS / 兼容接口

    优势：
    - 无需本地算力
    - 效果稳定
    - 运维成本低

    劣势：
    - 需付费
    - 网络延迟
    - 隐私合规风险

    注意：
    - MODEL_NAME 必须是服务端真实存在的模型
    - 阿里云 Embedding 模型名为 text-embedding-v3 / v2
    """
    from langchain_openai import OpenAIEmbeddings

    embedding_model = OpenAIEmbeddings(
        model=MODEL_NAME,               # ✅ 使用真实模型名
        openai_api_base=MODEL_URL,
        openai_api_key=MODEL_KEY,
        dimensions=1024,                # text-embedding-v3 支持
        check_embedding_ctx_length=False,  # 兼容部分国产接口
    )
    return embedding_model



def embedding_vec():

    embedding_model = init_embedding_model_by_fast()
    #embedding_model = init_embedding_model_by_huggingface(device="cpu")
    #embedding_model = init_embedding_model_by_openai()

    text = "向量测试"
    vec = embedding_model.embed_query(text)

    print(f"✅ 向量维度: {len(vec)}")
    print(f"✅ 向量前 5 维: {vec[:5]}")


if __name__ == "__main__":
    embedding_vec()