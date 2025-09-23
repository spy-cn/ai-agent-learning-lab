import getpass
import os

from langchain_community.document_loaders import PyPDFLoader

# 修正1：使用原始字符串或双反斜杠处理文件路径
file_path = r"E:\code_project\self_project\langchain-project\data\病理专业医疗质量控制指标(2024 年版).pdf"
# 或者
# file_path = "E:\\code_project\\self_project\\langchain-project\\data\\病理专业医疗质量控制指标(2024 年版).pdf"

loader = PyPDFLoader(file_path)

# 修正2：使用同步加载方式
pages = loader.load()  # 使用同步load()方法而不是异步alazy_load()

# 或者如果你想使用异步方式，需要这样写：
"""
import asyncio

async def load_pdf():
    loader = PyPDFLoader(file_path)
    pages = []
    async for page in loader.alazy_load():
        pages.append(page)
    return pages

pages = asyncio.run(load_pdf())
"""
print(pages)
for page in pages:
    print(page.page_content)



from langchain_unstructured import UnstructuredLoader

if "UNSTRUCTURED_API_KEY" not in os.environ:
    os.environ["UNSTRUCTURED_API_KEY"] = getpass.getpass(
        "Enter your Unstructured API key: "
    )

loader = UnstructuredLoader(
    file_path=file_path,
    strategy="hi_res",
    partition_via_api=True,
    coordinates=True,
)
docs = []
for doc in loader.lazy_load():
    docs.append(doc)

