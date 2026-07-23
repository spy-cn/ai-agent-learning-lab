

import os
import datetime
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

load_dotenv()

# ========== 1. 定义模型和工具 ==========
MODEL_NAME = "deepseek-v4-flash"
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
@tool
def get_current_time() -> str:
    """获取当前精准的北京时间（格式：YYYY-MM-DD HH:MM:SS）"""
    # UTC+8 即北京时间
    tz_beijing = datetime.timezone(datetime.timedelta(hours=8))
    now = datetime.datetime.now(tz=tz_beijing)
    return now.strftime("%Y-%m-%d %H:%M:%S")

TAVILY_KEY ="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"
#pip install langchain-tavily
search = TavilySearch(max_results=3, tavily_api_key=TAVILY_KEY)
tools = [search,get_current_time]

# ========== 2. 创建 Agent ==========
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="你是一位助手，需要调用工具来帮助用户。",
)

# ========== 3. 调用 Agent（流式输出中间过程） ==========
print("=== 测试 Agent ===\n")

for chunk in agent.stream(
    {
        "messages": [
            {"role": "user", "content": "当前时间上海的天气是？"},
        ]
    }
):
    # Agent 会输出多步：思考 → 调用工具 → 接收结果 → 生成答案
    print(chunk, end="\n\n---\n\n")
