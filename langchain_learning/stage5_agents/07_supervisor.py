

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
# 每个 Worker 都是一个拥有独立 LLM 和专属 Prompt 的 Agent
# =========================================================

# 搜寻/研究员 Agent
researcher_worker = create_agent(
    model=llm,
    tools=[search_info],
    system_prompt="你是一名专业的信息检索员，负责使用 search_info 查找用户需要的信息，并整理成清晰的摘要。"
)

# 数学计算 Agent
math_worker = create_agent(
    model=llm,
    tools=[calculate],
    system_prompt="你是一名精通计算的数学专家，负责分析复杂的算式并通过 calculate 工具计算结果。"
)


# =========================================================
# 3. 把【子 Agent】封装为主管可以调用的工具 (Agent-as-a-Tool)
# =========================================================

@tool
async def run_research_agent(task: str) -> str:
    """【搜索专家 Agent】负责处理所有信息查询、资料检索类任务。"""
    res = await researcher_worker.ainvoke({"messages": [("user", task)]})
    # 返回子 Agent 思考并执行后的最终回答
    return res["messages"][-1].content

@tool
async def run_math_agent(task: str) -> str:
    """【数学专家 Agent】负责处理表达式计算、逻辑运算类任务。"""
    res = await math_worker.ainvoke({"messages": [("user", task)]})
    # 返回子 Agent 思考并执行后的最终回答
    return res["messages"][-1].content


# =========================================================
# 4. 创建【主管 Agent】 (Supervisor Agent)
# 主管只管理子 Agent 工具，不直接持有底层原子工具
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
# 5. 测试运行
# =========================================================
async def main():
    print("--- 监督者 Agent 开始分发任务 ---\n")
    async for chunk in supervisor.astream(
        {"messages": [{"role": "user", "content": "帮我查一下 Python 是什么，然后算一下 2024 - 1991"}]},
        stream_mode="values"
    ):
        latest_msg = chunk["messages"][-1]
        # 观察 Supervisor 与子 Agent 之间的协同交互
        print(f"[{latest_msg.type.upper()}]: {latest_msg.content}\n")

if __name__ == "__main__":
    asyncio.run(main())