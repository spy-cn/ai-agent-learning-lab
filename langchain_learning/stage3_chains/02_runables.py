"""
LangChain LCEL 进阶 Runnable 组件速查与实践

核心组件说明：
1. RunnableParallel ：并行执行多个链/任务，用于加速或组合多元输出。
2. RunnablePassthrough：透传输入数据，常用于在链的流动中保留原始 Query 或组合上下文。
3. RunnableLambda    ：将任意自定义 Python 函数快速转化为 Runnable 节点。
4. RunnableBranch    ：根据条件流转到不同的处理分支（逻辑路由）。
"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableParallel,
    RunnableLambda,
    RunnablePassthrough,
    RunnableBranch,
)

# 1. 环境与模型初始化
load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-v4-flash")
MODEL_URL = os.getenv("MODEL_URL", "https://llm-lniax9q66senyb7s.cn-beijing.maas.aliyuncs.com/compatible-mode/v1")
MODEL_KEY = os.getenv("MODEL_KEY", "sk-ws-H.EHHMLPX.GItO.MEUCIQDOAzCACqmt9lyJ1RrfpNRVVu9sgD9QAZRJo8cXtUKE1wIgbdfCUdVgwKvGPDb-wwa8-lj4kGXtiTJHjTLnNwUniZg")

llm = init_chat_model(
    model=MODEL_NAME,
    base_url=MODEL_URL,
    model_provider="openai",
    temperature=0,
    api_key=MODEL_KEY
)


def runnable_parallel():
    """
    ====================================================================
    1. RunnableParallel（并行链）
    ====================================================================
    【适用场景】：
    - 组合式生成：需要同时生成摘要、关键词、情感分析等多个独立结果。
    - RAG 检索：同时向向量数据库和关键词搜索引擎发起并发检索。
    - 性能优化：耗时较长的独立 LLM 调用并行化，降低整体响应延迟。
    """
    print("=== 1. RunnableParallel 示例 ===")

    joke_chain = PromptTemplate.from_template("讲一个关于{topic}的简短笑话") | llm | StrOutputParser()
    poem_chain = PromptTemplate.from_template("写一首关于{topic}的四句短诗") | llm | StrOutputParser()

    # 将多个链组合为一个字典并发执行，key 即为输出字典的 key
    map_chain = RunnableParallel(joke=joke_chain, poem=poem_chain)

    result = map_chain.invoke({"topic": "人工智能"})
    print(f"【笑话】:\n{result['joke']}\n")
    print(f"【诗歌】:\n{result['poem']}\n")


def runnable_passthrough():
    """
    ====================================================================
    2. RunnablePassthrough（数据透传与动态注入）
    ====================================================================
    【适用场景】：
    - RAG（检索增强生成）：在检索出相关文档的同时，保留用户原始的 question 传给 Prompt。
    - 管道赋值（assign）：在不破坏原有数据结构的前提下，向数据流中追加新的字段或中间计算结果。
    """
    print("=== 2. RunnablePassthrough 示例 ===")

    # 场景 A: 基础透传与并发处理
    chain_basic = RunnableParallel(
        original=RunnablePassthrough(),   # 原封不动保留输入
        length=lambda x: len(x),           # 结合 lambda 计算新属性
    )
    print("基础透传:", chain_basic.invoke("hello world"))

    # 场景 B: 模拟 RAG 典型应用 (Passthrough.assign)
    # 模拟一个简单的文档检索器函数
    fake_retriever = lambda inputs: f"关于 '{inputs['question']}' 的检索参考资料内容..."

    rag_chain = (
        # assign 会保留原本的 {"question": ...} 并追加 "context" 字段
        RunnablePassthrough.assign(context=fake_retriever)
        | PromptTemplate.from_template("参考资料: {context}\n回答问题: {question}")
        | llm
        | StrOutputParser()
    )

    rag_result = rag_chain.invoke({"question": "Python 怎么学"})
    print("\n模拟 RAG 结果:\n", rag_result)


def runnable_lambda():
    """
    ====================================================================
    3. RunnableLambda（自定义函数转换）
    ====================================================================
    【适用场景】：
    - 数据清洗/预处理：格式化输入文本、过滤敏感词、提取特定正则等。
    - 接入第三方服务：在 LCEL 流程中无缝插入自定义 API 调用、数据库读写。
    - 后处理：对 LLM 的输出做二次格式化转换或解析。
    """
    print("\n=== 3. RunnableLambda 示例 ===")

    # 方式 1：装饰器写法
    @RunnableLambda
    def shout(text: str) -> str:
        return text.strip().upper() + " !!!"

    # 方式 2：直接在管道中使用标准 Python 函数（LangChain 会自动将其隐式转为 RunnableLambda）
    def count_words(text: str) -> str:
        return f"{text} (字数: {len(text)})"

    chain = (
        PromptTemplate.from_template("用一个词形容{topic}")
        | llm
        | StrOutputParser()
        | shout
        | count_words
    )

    print(chain.invoke({"topic": "快乐"}))


def runnable_branch():
    """
    ====================================================================
    4. RunnableBranch（条件路由 / 分支）
    ====================================================================
    【适用场景】：
    - 意图识别分发：识别用户意图（如“客服”、“技术支持”、“退款”），将请求分发给专门的 Prompt 或 Agent。
    - 降级/备用逻辑：主模型调用失败或不满足预设条件时，流转至备用模型。
    """
    print("\n=== 4. RunnableBranch 示例 ===")

    # 语法结构: RunnableBranch((condition_1, chain_1), (condition_2, chain_2), ..., default_chain)
    branch = RunnableBranch(
        (lambda x: "技术" in x or "python" in x.lower(), lambda x: f"【路由至技术 Agent】处理：{x}"),
        (lambda x: "天气" in x, lambda x: f"【路由至天气 Agent】查询：{x}"),
        lambda x: f"【路由至通用客服】回答：{x}",  # 默认分支
    )

    print(branch.invoke("今天天气怎么样"))
    print(branch.invoke("Python 怎么学"))
    print(branch.invoke("你好，我想找人工服务"))


if __name__ == "__main__":
    #runnable_parallel()
    #runnable_passthrough()
    #runnable_lambda()
    runnable_branch()