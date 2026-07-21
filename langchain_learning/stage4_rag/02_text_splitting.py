from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    BSHTMLLoader,
    WebBaseLoader,
    CSVLoader,
    JSONLoader,
)
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

def split_txt():
    """切分纯文本字符串，返回字符串列表。"""
    file_path = "data/sample.txt"
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    text = docs[0].page_content

    # 针对中文优化分隔符优先级：段落 -> 换行 -> 句号/叹号/问号 -> 空格 -> 字符
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
    )
    chunks = text_splitter.split_text(text)
    for i, chunk in enumerate(chunks, start=1):
        print(f"\n[Chunk {i}/{len(chunks)}]")
        print("-" * 30)
        print(chunk)
        print("-" * 30)


def split_md():

    file_path = "data/LangChain_revised.md"
    # 简单场景直接使用 TextLoader 即可
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()

    md_text = docs[0].page_content
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    # 基于标题进行初次切分（标题文本会被自动放入 metadata 中）
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )
    md_header_splits = markdown_splitter.split_text(md_text)
    # 2. 对初次切分后的 Document 再用递归切分器兜底，防止某个章节过长
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    )
    final_docs = text_splitter.split_documents(md_header_splits)
    for i, chunk in enumerate(final_docs, start=1):
        print(f"\n[Chunk {i}/{len(final_docs)}]")
        print("-" * 30)
        print(chunk.page_content)
        print("-" * 30)

if __name__ == "__main__":
    split_md()