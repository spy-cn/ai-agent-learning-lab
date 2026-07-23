import os
import asyncio
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

load_dotenv()
MODEL_NAME = "deepseek-v4-flash"
MODEL_URL = "https://llm-lniax9q66senyb7s.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
MODEL_KEY = "sk-ws-H.EHHMLPX.GItO.MEUCIQDOAzCACqmt9lyJ1RrfpNRVVu9sgD9QAZRJo8cXtUKE1wIgbdfCUdVgwKvGPDb-wwa8-lj4kGXtiTJHjTLnNwUniZg"


async def glm_mcp():
    # 智谱 API Key
    ZHIPU_API_KEY = os.getenv("ZHIPUAI_API_KEY", "7cd161e1e915448fbdb840acf5f632b4.RftsppsJbQgiPfIs")

    # 2. 配置 MCP 客户端连接
    mcp_client = MultiServerMCPClient({
        "zhipu_search": {
            "transport": "sse",
            "url": f"https://open.bigmodel.cn/api/mcp/web_search/sse?api_key={ZHIPU_API_KEY}",
            # 如果智谱服务要求在 Header 中传 Key，可以取消下面的注释：
            "headers": {
                "Authorization": f"Bearer {ZHIPU_API_KEY}"
            }
        }
    })

    tools = await mcp_client.get_tools()
    print(f"成功获取到的智谱 MCP 工具: {[t.name for t in tools]}")

    llm = init_chat_model(
        model=MODEL_NAME,
        base_url=MODEL_URL,
        model_provider="openai",
        temperature=0,
        api_key=MODEL_KEY,
        streaming=True,
    )
    agent = create_agent(model=llm, tools=tools)

    print("[AI 回答]: ", end="", flush=True)

    async for message, metadata in agent.astream(
            {"messages": [{"role": "user", "content": "帮我搜一下今天最新的科技头条新闻"}]},
            stream_mode="messages"
    ):
        if metadata.get("langgraph_node") == "model":
            print(message.content, end="", flush=True)

    print("\n\n--- 打印结束 ---")


async def bailian_mcp():
    # TODO 阿里云百炼MCP 失败
    # 同模式配置 URL 参数与 Header
    mcp_url = f"https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/sse?api_key={MODEL_KEY}"

    mcp_client = MultiServerMCPClient({
        "bailian_search": {
            "transport": "sse",
            "url": "https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/sse",
            "headers": {
                "Authorization": f"Bearer {MODEL_KEY}",
                "Accept": "text/event-stream",
                "Cache-Control": "no-cache"
            }
        }
    })

    try:
        tools = await mcp_client.get_tools()
        print(f"成功获取到的百炼 MCP 工具: {[t.name for t in tools]}\n")
    except Exception as e:
        print(f"MCP 工具获取失败，请优先检查第一步脚本返回的报错信息！",e)
        return

    llm = init_chat_model(
        model=MODEL_NAME,
        base_url=MODEL_URL,
        model_provider="openai",
        temperature=0,
        api_key=MODEL_KEY,
        streaming=True,
    )

    agent = create_agent(model=llm, tools=tools)

    print("[AI 回答]: ", end="", flush=True)
    async for message, metadata in agent.astream(
            {"messages": [{"role": "user", "content": "帮我搜一下今天最新的科技头条新闻"}]},
            stream_mode="messages"
    ):
        if metadata.get("langgraph_node") in ("agent", "model"):
            if isinstance(message.content, str) and message.content:
                print(message.content, end="", flush=True)

    print("\n\n--- 打印结束 ---")

import asyncio
import httpx


async def check_bailian_mcp():
    # 替换为你测试的 KEY
    api_key = "sk-ws-H.EHHMLPX.GItO.MEUCIQDOAzCACqmt9lyJ1RrfpNRVVu9sgD9QAZRJo8cXtUKE1wIgbdfCUdVgwKvGPDb-wwa8-lj4kGXtiTJHjTLnNwUniZg"
    url = f"https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/sse?api_key={api_key}"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "X-DashScope-SSE": "enable"
    }

    print("正在测试连接百炼 MCP SSE 端点...")
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            async with client.stream("GET", url, headers=headers) as response:
                print(f"HTTP 响应状态码: {response.status_code}")
                if response.status_code != 200:
                    content = await response.aread()
                    print(f"服务端返回的错误信息: {content.decode('utf-8', errors='ignore')}")
                else:
                    print("✅ SSE 连接握手成功！")
    except Exception as e:
        print(f"❌ 网络层直接拒绝连接: {e}")


if __name__ == "__main__":
    asyncio.run(bailian_mcp())

