"""
LangChain 多源文档加载器示例 (Document Loaders)

涵盖格式：
1. TXT / Markdown (文本/文档)
2. PDF (单页/多页 PDF)
3. HTML / Web (网页数据)
4. CSV / JSON / YAML (结构化数据)
"""

import os
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    BSHTMLLoader,
    WebBaseLoader,
    CSVLoader,
    JSONLoader,
)

def load_txt():
    """
    1. 加载普通 TXT 文本
    说明：最基础的加载器，整篇文本会被作为一个 Document 加载。
    注意：在 Windows 环境下建议显式指定 encoding="utf-8"。
    """
    print("=" * 20 + " 1. TXT 加载 " + "=" * 20)
    file_path = "data/sample.txt"

    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()

    print(f"加载文档数量: {len(docs)}")
    print(f"内容: {docs[0].page_content}")
    print(f"元数据: {docs[0].metadata}")


def load_pdf():
    """
    2. 加载 PDF 文件
    依赖安装：pip install pypdf
    说明：PyPDFLoader 会默认将 PDF 的【每一页】解析为一个独立的 Document 对象。
    元数据中会包含源文件路径和页码（page）。
    """
    print("\n" + "=" * 20 + " 2. PDF 加载 " + "=" * 20)
    file_path = "data/sample.pdf"

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    print(f"总页数（Document 数量）: {len(docs)}")
    if docs:
        print(f"第一页内容预览: {docs[0].page_content[:100]}...")
        print(f"第一页元数据: {docs[0].metadata}")


def load_markdown():
    """
    3. 加载 Markdown 文件
    说明：
    - 方式 A：纯文本加载，使用 TextLoader 即可（轻量、高效）。
    - 方式 B：如需按 Header 结构拆分 Markdown 元素，推荐 UnstructuredMarkdownLoader。
      (需安装 pip install unstructured)
    """
    print("\n" + "=" * 20 + " 3. Markdown 加载 " + "=" * 20)
    file_path = "data/sample.md"

    # 简单场景直接使用 TextLoader 即可
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()

    print(f"加载文档数量: {len(docs)}")
    print(f"内容:\n{docs[0].page_content}")
    print(f"元数据: {docs[0].metadata}")


def load_html():
    """
    4. 加载 HTML / 网页数据
    说明：
    - 本地 HTML: BSHTMLLoader（基于 BeautifulSoup4 解析正文内容）。
    - 网络 URL: WebBaseLoader（支持批量爬取在线网页）。
      (需安装 pip install beautifulsoup4)
    """
    print("\n" + "=" * 20 + " 4. HTML / Web 加载 " + "=" * 20)

    # 示例 A: 爬取网络页面 (WebBaseLoader)
    web_url = "https://python.langchain.com/"
    print(f"正在抓取网页: {web_url} ...")
    try:
        web_loader = WebBaseLoader(web_url)
        web_docs = web_loader.load()
        print(f"[Web] 加载文档数量: {len(web_docs)}")
        print(f"[Web] 网页标题与元数据: {web_docs[0].metadata}")
        print(f"[Web] 正文前 100 字符: {web_docs[0].page_content.strip()[:100]}...")
    except Exception as e:
        print(f"网页抓取失败: {e}")


def load_csv():
    """
    5. 加载 CSV 文件
    说明：CSVLoader 默认会将【每一行数据】作为一个独立的 Document。
    可以通过 source_column 参数指定某列作为 metadata['source']。
    """
    print("\n" + "=" * 20 + " 5. CSV 加载 " + "=" * 20)
    file_path = "data/sample.csv"

    # 指定 utf-8 编码加载
    loader = CSVLoader(file_path, encoding="utf-8")
    docs = loader.load()

    print(f"总行数（Document 数量）: {len(docs)}")
    print(f"第一行转为 Document 后的格式:\n{docs[0].page_content}")
    print(f"元数据（包含行号）: {docs[0].metadata}")


def load_json():
    """
    6. 加载 JSON 文件
    依赖安装：pip install jq
    说明：JSONLoader 需要传入 jq_schema 语法提取指定字段中的文本。
    """
    print("\n" + "=" * 20 + " 6. JSON 加载 " + "=" * 20)
    file_path = "data/sample.json"

    try:
        # jq_schema='.[] | .text' 表示遍历数组并提取每个元素的 text 字段
        loader = JSONLoader(
            file_path=file_path,
            jq_schema=".[] | .text",
            text_content=True
        )
        docs = loader.load()
        print(f"解析提取出的 Document 数量: {len(docs)}")
        print(f"第一个 JSON 节点内容: {docs[0].page_content}")
    except Exception as e:
        print(f"加载 JSON 失败（确保已安装 jq: `pip install jq`）: {e}")


def load_yaml():
    """
    7. 加载 YAML 配置文件
    说明：LangChain 官方没有单独的 YAMLLoader，通常先用 PyYAML 库解析为 Python 字典/列表，
    然后再通过 Document 对象手动构建。
    依赖安装：pip install pyyaml
    """
    print("\n" + "=" * 20 + " 7. YAML 加载 " + "=" * 20)
    import yaml
    from langchain_core.documents import Document

    file_path = "data/sample.yaml"

    # 通过 PyYAML 解析并封装为 Document
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # 手动转换为 Document 格式
    docs = [Document(page_content=str(data), metadata={"source": file_path})]

    print(f"加载 YAML 结果:\n{docs[0].page_content}")
    print(f"元数据: {docs[0].metadata}")


if __name__ == "__main__":
    #load_txt()
    #load_pdf()
    #load_markdown()
    #load_html()
    #load_csv()
    #load_json()
    load_yaml()
