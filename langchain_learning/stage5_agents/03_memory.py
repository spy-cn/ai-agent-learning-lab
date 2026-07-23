
import os
import datetime
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool

load_dotenv()

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
# 关键：传入 checkpointer 启用记忆
agent = create_agent(
    model=llm,
    tools=tools,
    checkpointer=InMemorySaver(),
)

# ========== 第一轮对话（thread_id=1） ==========
print("=== 第一轮 ===")
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "今天北京天气怎么样？"}]},
    config={"configurable": {"thread_id": "1"}},  # 指定线程 ID
):
    print(chunk, end="\n\n")

# ========== 第二轮对话（同一个 thread_id，有记忆） ==========
print("\n=== 第二轮（测试记忆） ===")
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "那上海呢？"}]},  # 没说"天气"，但 Agent 应该知道
    config={"configurable": {"thread_id": "1"}},
):
    print(chunk, end="\n\n")