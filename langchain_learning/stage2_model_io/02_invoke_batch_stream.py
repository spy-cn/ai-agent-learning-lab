from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
import time
import asyncio

MODEL_NAME = "deepseek-v4-flash"
MODEL_URL = "https://llm-lniax9q66senyb7s.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
MODEL_KEY = "sk-ws-H.EHHMLPX.GItO.MEUCIQDOAzCACqmt9lyJ1RrfpNRVVu9sgD9QAZRJo8cXtUKE1wIgbdfCUdVgwKvGPDb-wwa8-lj4kGXtiTJHjTLnNwUniZg"

def init_model():
    llm = init_chat_model(
        model=MODEL_NAME,
        base_url=MODEL_URL,
        model_provider="openai",
        temperature=0,
        api_key=MODEL_KEY
    )
    return llm


def print_stream():
    llm = init_model()
    print("=== stream 流式输出 ===")
    for chunk in llm.stream("请证明量子理论"):
        print(chunk.content, end="", flush=True)
    print("\n")


def print_batch():
    llm = init_model()
    print("=== batch 批量调用 ===")
    messages_list = [
        [{"role": "user", "content": "写一首关于春天的诗"}],
        [{"role": "user", "content": "写一首关于夏天的诗"}],
    ]
    resps = llm.batch(messages_list)
    for i, resp in enumerate(resps):
        print(f"第{i + 1}首：", resp.content[:30], "...")


def print_sync_invoke():
    llm = init_model()
    print("\n=== 同步耗时 ===")

    # 同步：逐个调用
    start = time.time()
    _ = [llm.invoke([{"role": "user", "content": "说一个字"}]) for _ in range(3)]
    print(f"同步 3 次：{time.time() - start:.2f}s") #同步 3 次：1.60s


async def print_async_invoke():
    llm = init_chat_model(
        model="glm-edge-v-5b",
        base_url="http://192.168.2.195:8001/v1",
        model_provider="openai",
        temperature=0,
        api_key="sk-admin-ford-glm-5b-123"
    )
    print("\n=== 异步耗时===")
    # 异步：并行调用 测试下来发现异步效果不如同步，应该是阿里云那边有限制
    tasks = [llm.ainvoke([{"role": "user", "content": "说一个字"}]) for _ in range(3)]
    return await asyncio.gather(*tasks)


def print_async_test():
    start = time.time()
    asyncio.run(print_async_invoke())
    print(f"异步 3 次：{time.time() - start:.2f}s")


def print_langchain_batch():
    llm = init_model()

    # 每一个元素都是符合要求的 str
    inputs = ["说一个字" for _ in range(3)]

    print("\n=== LangChain Batch 耗时 ===")
    start = time.time()

    # 正常运行
    results = llm.batch(inputs)

    print(f"LangChain Batch 3 次：{time.time() - start:.2f}s")

if __name__ == "__main__":
    print_stream()