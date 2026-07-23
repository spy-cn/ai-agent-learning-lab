import os
import asyncio
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

load_dotenv()

MODEL_NAME = "deepseek-v4-pro"
MODEL_URL = "https://llm-lniax9q66senyb7s.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
MODEL_KEY = "sk-ws-H.EHHMLPX.GItO.MEUCIQDOAzCACqmt9lyJ1RrfpNRVVu9sgD9QAZRJo8cXtUKE1wIgbdfCUdVgwKvGPDb-wwa8-lj4kGXtiTJHjTLnNwUniZg"

llm = init_chat_model(
    model=MODEL_NAME,
    base_url=MODEL_URL,
    model_provider="openai",
    temperature=0,
    api_key=MODEL_KEY,
    streaming=True,
)


# =========================================================
# 1. 定义原子工具 (底层 Tool)
# =========================================================
@tool
def search_info(query: str) -> str:
    """搜索信息"""
    return f"关于「{query}」的搜索结果：Python 是一种高级编程语言。"


@tool
def calculate(expression: str) -> str:
    """数学计算"""
    try:
        return f"{expression} = {eval(expression)}"
    except Exception as e:
        return f"计算失败：{e}"


# =========================================================
# 2. 创建独立的【子 Agent】 (Worker Agents)
# =========================================================

researcher_worker = create_agent(
    model=llm,
    tools=[search_info],
    system_prompt="你是一名专业的信息检索员，负责使用 search_info 查找用户需要的信息，并整理成清晰的摘要。"
)

math_worker = create_agent(
    model=llm,
    tools=[calculate],
    system_prompt="你是一名精通计算的数学专家，负责分析复杂的算式并通过 calculate 工具计算结果。"
)


# =========================================================
# 3. 把【子 Agent】封装为支持内嵌流式输出的工具
# =========================================================

@tool
async def run_research_agent(task: str) -> str:
    """【搜索专家 Agent】负责处理所有信息查询、资料检索类任务。"""
    print(f"\n\n🤖 [搜索专家 Agent 开始执行] -> 任务: {task}")
    print("   ↳ ", end="", flush=True)

    full_response = []
    # 💡 使用 astream 让子 Agent 的回答也实时吐字
    async for message, metadata in researcher_worker.astream(
            {"messages": [("user", task)]},
            stream_mode="messages"
    ):
        if metadata.get("langgraph_node") in ("agent", "model"):
            if isinstance(message.content, str) and message.content:
                print(message.content, end="", flush=True)
                full_response.append(message.content)

    print("\n   ✅ [搜索专家 Agent 完成]\n")
    return "".join(full_response)


@tool
async def run_math_agent(task: str) -> str:
    """【数学专家 Agent】负责处理表达式计算、逻辑运算类任务。"""
    print(f"\n\n🔢 [数学专家 Agent 开始执行] -> 任务: {task}")
    print("   ↳ ", end="", flush=True)

    full_response = []
    # 💡 使用 astream 让子 Agent 的回答也实时吐字
    async for message, metadata in math_worker.astream(
            {"messages": [("user", task)]},
            stream_mode="messages"
    ):
        if metadata.get("langgraph_node") in ("agent", "model"):
            if isinstance(message.content, str) and message.content:
                print(message.content, end="", flush=True)
                full_response.append(message.content)

    print("\n   ✅ [数学专家 Agent 完成]\n")
    return "".join(full_response)


# =========================================================
# 4. 创建【主管 Agent】 (Supervisor Agent)
# =========================================================

supervisor = create_agent(
    model=llm,
    tools=[run_research_agent, run_math_agent],
    system_prompt="""你是一名主管。你手下有两名专家 Agent：
    - run_research_agent：处理信息检索任务
    - run_math_agent：处理数学计算任务

    当收到复杂任务时：
    1. 将任务拆解后交由对应的专家 Agent 完成；
    2. 汇总各专家的结果，给用户一个完整的回答。
    """
)


# =========================================================
# 5. 测试运行（主链路流式接收）
# =========================================================
async def main():
    print("================ 监督者 Agent 启动 ================")
    print("👨‍💼 [主管 Agent]: ", end="", flush=True)

    # 💡 主管 Agent 使用 stream_mode="messages" 实现逐 Chunk 输出
    async for message, metadata in supervisor.astream(
            {"messages": [{"role": "user", "content": "帮我查一下 Python 是什么，然后算一下 2024 - 1991"}]},
            stream_mode="messages"
    ):
        # 仅打印主管 Agent 本身生成的文本，防止将工具返回的大段字符串重复打印
        if metadata.get("langgraph_node") in ("agent", "model"):
            if isinstance(message.content, str) and message.content:
                print(message.content, end="", flush=True)

    print("\n\n================ 任务完成 ================")


if __name__ == "__main__":
    asyncio.run(main())