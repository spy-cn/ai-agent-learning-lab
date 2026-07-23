import os
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.chat_models import init_chat_model

load_dotenv()

# ========== 1. 创建工具 ==========
@tool
def add(a: int, b: int) -> int:
    """两个整数相加"""
    return a + b
@tool
def query_user_info(user_id: int) -> str:
    """根据 user_id 查询用户姓名"""
    users = {1001: "Jack", 1002: "Tom", 1003: "Alice"}
    return users.get(user_id, "用户不存在")


print(f"工具名：{add.name}")
print(f"工具描述：{add.description}")
print(f"工具参数：{add.args}")


# ========== 2. 绑定工具到模型 ==========

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

tools = [add, query_user_info]
llm_with_tools = llm.bind_tools(tools)

# ========== 3. 测试工具调用 ==========
resp = llm_with_tools.invoke("帮我查下 1001 用户的信息")
print(f"\n模型响应：{resp.content}")
print(f"工具调用信息：{resp.tool_calls}")


# 注意：此时模型只是返回了"要调用什么工具"，还没真正执行
# 需要我们手动执行
if resp.tool_calls:
    for tc in resp.tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"]
        # 执行工具
        result = {"query_user_info": query_user_info, "add": add}[tool_name].invoke(tool_args)
        print(f"执行 {tool_name}({tool_args}) = {result}")