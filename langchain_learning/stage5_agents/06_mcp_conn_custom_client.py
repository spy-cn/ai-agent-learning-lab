"""连接自定义MCP客户端"""

import asyncio
from pathlib import Path
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client


async def run_client_by_streamable_http():
    """
    对应客户端 是这种方式 启动：mcp.run(transport="streamable-http")
    :return:
    """

    # 服务器地址（注意路径通常为 /mcp）
    url = "http://127.0.0.1:8000/mcp"

    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            # 初始化
            await session.initialize()

            # ----- 工具调用 -----
            # 1. 列出所有工具
            tools = await session.list_tools()
            print("=== 可用工具 ===")
            for tool in tools.tools:
                print(f"- {tool.name}: {tool.description}")

            # 2. 调用 multiply_numbers
            result = await session.call_tool("multiply_numbers", arguments={"a": 3.5, "b": 2})
            print(f"\nmultiply_numbers(3.5, 2) = {result.content[0].text}")

            # 3. 调用 get_system_status
            status = await session.call_tool("get_system_status", arguments={})
            print(f"系统状态: {status.content[0].text}")

            # ----- 资源读取 -----
            resources = await session.list_resources()
            print("\n=== 可用资源 ===")
            for res in resources.resources:
                print(f"- {res.uri}: {res.name}")

            # 读取 system://logs/latest
            log = await session.read_resource("system://logs/latest")
            print(f"\n最新日志: {log.contents[0].text}")

            # ----- 提示获取 -----
            prompts = await session.list_prompts()
            print("\n=== 可用提示 ===")
            for p in prompts.prompts:
                print(f"- {p.name}: {p.description}")

            # 获取 code_review_prompt
            prompt_result = await session.get_prompt("code_review_prompt", arguments={"language": "python"})
            print(f"\n代码审查提示: {prompt_result.messages[0].content.text}")


async def run_client_by_stdio():
    """
    对应客户端 是这种启动方式：mcp.run(transport="stdio")
    :return:
    """
    # 配置你的 MCP 服务参数
    server_path = Path(__file__).with_name("05_mcp_cusom.py")
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_path)],
    )

    # 建立 Stdio 连接
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 1. 发现工具
            tools = await session.list_tools()
            print("可用工具:", [t.name for t in tools.tools])

            # 2. 调用工具
            result = await session.call_tool("multiply_numbers", {"a": 6, "b": 7})
            print("计算结果:", result.content[0].text)
            # 3. 调用 get_system_status
            status = await session.call_tool("get_system_status", arguments={})
            print(f"系统状态: {status.content[0].text}")

            # ----- 资源读取 -----
            resources = await session.list_resources()
            print("\n=== 可用资源 ===")
            for res in resources.resources:
                print(f"- {res.uri}: {res.name}")

            # 读取 system://logs/latest
            log = await session.read_resource("system://logs/latest")
            print(f"\n最新日志: {log.contents[0].text}")

            # ----- 提示获取 -----
            prompts = await session.list_prompts()
            print("\n=== 可用提示 ===")
            for p in prompts.prompts:
                print(f"- {p.name}: {p.description}")

            # 获取 code_review_prompt
            prompt_result = await session.get_prompt("code_review_prompt", arguments={"language": "python"})
            print(f"\n代码审查提示: {prompt_result.messages[0].content.text}")


if __name__ == "__main__":
    # asyncio.run(run_client_by_streamable_http())
    asyncio.run(run_client_by_stdio())
