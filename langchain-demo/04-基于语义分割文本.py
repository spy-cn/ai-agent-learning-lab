import json
from pathlib import Path

from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, JSONLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings


def load_and_chunk_json(json_path: str, api_base: str) -> list[Document]:
    """
    加载JSON文件，并进行语义分块

    Args:
        json_path: JSON文件路径
        api_base: 嵌入模型API地址

    Returns:
        分块后的文档列表
    """
    # 1. 加载JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 提取文本内容（假设JSON中包含文本字段）
    # 根据实际JSON结构调整这部分代码
    if isinstance(data, dict):
        # 如果是字典，提取所有文本值
        texts = [str(value) for value in data.values() if isinstance(value, (str, int, float))]
    elif isinstance(data, list):
        # 如果是列表，提取所有元素的文本表示
        texts = [str(item) for item in data]
    else:
        texts = [str(data)]

    text = "\n".join(texts).strip()  # 移除首尾空白
    text = " ".join(text.split())  # 合并多余空格

    # 3. 初始化嵌入模型
    embeddings = OpenAIEmbeddings(
        api_key="EMPTY",  # 如果API不需要key可以设为EMPTY
        openai_api_base=api_base
    )


    # 4. 创建语义分块器
    text_splitter = SemanticChunker(embeddings)

    # 5. 进行语义分块
    docs = text_splitter.create_documents([text])

    return docs


def load_and_chunk_pdf(pdf_path: str, api_base: str) -> list[Document]:
    """
    加载PDF文件并进行语义分块

    Args:
        pdf_path: PDF文件路径
        api_base: 嵌入模型API地址

    Returns:
        分块后的文档列表
    """
    # 1. 加载PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # 2. 合并所有页面文本
    text = "\n".join([page.page_content for page in pages])

    # 3. 初始化嵌入模型
    embeddings = OpenAIEmbeddings(
        api_key="EMPTY",  # 如果API不需要key可以设为EMPTY
        openai_api_base=api_base
    )

    # 4. 创建语义分块器
    text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="gradient")

    # 5. 进行语义分块
    docs = text_splitter.create_documents([text])

    return docs


if __name__ == "__main__":
    # 配置参数
    pdf_path = "E:/code_project/self_project/langchain-project/data/病理专业医疗质量控制指标(2024 年版).pdf"
    json_path = "E:/code_project/self_project/langchain-project/data/dataset_PROJ1756733581307_1757905149077.json"
    api_base = "http://192.168.1.99:8565/embeddings"

    # 加载并分块
    #chunks = load_and_chunk_pdf(pdf_path, api_base)
    chunks = load_and_chunk_json(json_path, api_base)

    # 打印第一个分块
    print("第一个分块内容:")
    print(chunks[0].page_content)
    print(f"\n总共有 {len(chunks)} 个分块")
