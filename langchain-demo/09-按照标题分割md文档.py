from langchain.text_splitter import MarkdownHeaderTextSplitter
from pathlib import Path


def split_markdown_by_headers(file_path):
    """
    按标题分割Markdown文件

    Args:
        file_path (str): Markdown文件路径

    Returns:
        list: 分割后的文档块列表
    """
    # 定义要分割的标题层级
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]

    # 创建分割器
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False  # 是否从内容中移除标题
    )

    # 读取Markdown文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        markdown_content = f.read()

    # 分割文档
    md_header_splits = markdown_splitter.split_text(markdown_content)

    return md_header_splits


if __name__ == "__main__":
    # 示例使用
    markdown_file = "E:\code_project\self_project\langchain-project\data\病理专业医疗质量控制指标(2024 年版).md"  # 替换为你的Markdown文件路径

    # 检查文件是否存在
    if not Path(markdown_file).exists():
        print(f"错误：文件 {markdown_file} 不存在")
        exit(1)

    # 分割文档
    splits = split_markdown_by_headers(markdown_file)

    # 打印分割结果
    print(f"共分割出 {len(splits)} 个块:")
    for i, chunk in enumerate(splits, 1):
        print(f"\n块 {i}:")
        print("元数据:", chunk.metadata)
        print("内容:", chunk.page_content)