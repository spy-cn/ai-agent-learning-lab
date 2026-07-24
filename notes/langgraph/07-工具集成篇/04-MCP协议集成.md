# MCP 协议集成

MCP（Model Context Protocol）是 Anthropic 提出的开放标准，让 AI 应用能以统一方式连接外部工具和数据源。本篇介绍在 LangGraph 中集成 MCP。

---

## 什么是 MCP

### 核心概念

MCP（Model Context Protocol）是一个**客户端-服务器协议**，标准化了 LLM 应用与外部资源（工具、数据、API）之间的通信。

```
传统方式:                     MCP 方式:
┌─────────┐  ┌──────────┐   ┌─────────┐  ┌──────────┐
│ AI App  │←→│ 工具A     │   │ AI App  │←→│ MCP      │
│         │  └──────────┘   │(MCP客户端)│   │ Client   │
│         │  ┌──────────┐   │         │  └──────────┘
│         │←→│ 工具B     │   │         │       ↕
│         │  └──────────┘   │         │  ┌──────────┐
│         │  ┌──────────┐   │         │←→│ MCP      │
│         │←→│ 工具C     │   │         │   │ Server A │
└─────────┘  └──────────┘   └─────────┘  └──────────┘
  每个工具单独集成                            统一协议
```

### MCP 的优势

| 优势 | 说明 |
|------|------|
| 标准化 | 一个协议连接所有工具 |
| 即插即用 | 工具服务器独立部署 |
| 动态发现 | 自动发现可用工具 |
| 语言无关 | 支持 Python、JS、Rust 等 |

---

## MCP 架构

```
┌───────────────────────────────────────┐
│          MCP Host (AI 应用)            │
│                                       │
│  ┌─────────────────────────────────┐  │
│  │       MCP Client                │  │
│  │  ┌────────┐  ┌────────┐        │  │
│  │  │Client A│  │Client B│        │  │
│  │  └────┬───┘  └────┬───┘        │  │
│  └───────┼───────────┼────────────┘  │
│          │           │               │
└──────────┼───────────┼───────────────┘
           │           │
      stdio/SSE   stdio/SSE
           │           │
   ┌───────▼──────┐   ┌▼──────────────┐
   │  MCP Server  │   │  MCP Server   │
   │  (文件系统)   │   │   (数据库)     │
   │              │   │               │
   │ Tools:       │   │ Tools:        │
   │  - read_file │   │  - query      │
   │  - write_file│   │  - insert     │
   │  - list_dir  │   │  - update     │
   └──────────────┘   └───────────────┘
```

---

## 在 LangGraph 中使用 MCP

### 安装

```bash
pip install mcp langchain-mcp-adapters
```

### 连接 MCP Server

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

# 配置 MCP Server
server_params = StdioServerParameters(
    command="python",
    args=["-m", "mcp_server_filesystem", "/path/to/workspace"],
    env=None
)

# 连接并加载工具
async def get_mcp_tools():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            return tools

# 获取工具
import asyncio
tools = asyncio.run(get_mcp_tools())

print(f"可用工具: {[t.name for t in tools]}")
```

### 在 Agent 中使用

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

# 创建使用 MCP 工具的 Agent
agent = create_react_agent(llm, tools)

result = agent.invoke({
    "messages": [("user", "列出工作目录下的所有Python文件")]
})
```

---

## 常用 MCP Server

### 文件系统 Server

```python
# 安装: pip install mcp-server-filesystem
# 启动: python -m mcp_server_filesystem /path/to/workspace

server_params = StdioServerParameters(
    command="python",
    args=["-m", "mcp_server_filesystem", "/home/user/workspace"]
)
```

提供工具：
- `read_file`: 读取文件
- `write_file`: 写入文件
- `list_directory`: 列目录
- `create_directory`: 创建目录
- `move_file`: 移动文件
- `search_files`: 搜索文件

### GitHub Server

```python
# 安装: pip install mcp-server-github
server_params = StdioServerParameters(
    command="python",
    args=["-m", "mcp_server_github"],
    env={"GITHUB_TOKEN": "your-token"}
)
```

提供工具：
- 搜索仓库
- 创建 Issue
- 管理 PR
- 读取文件

### SQLite Server

```python
server_params = StdioServerParameters(
    command="python",
    args=["-m", "mcp_server_sqlite", "--db-path", "mydb.db"]
)
```

---

## 自定义 MCP 工具

### 用 Python 创建 MCP Server

```python
# my_mcp_server.py
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import json

app = Server("my-tools")

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_stock_price",
            description="获取股票价格",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "股票代码"
                    }
                },
                "required": ["symbol"]
            }
        ),
        Tool(
            name="send_notification",
            description="发送通知",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {"type": "string"},
                    "channel": {"type": "string"}
                },
                "required": ["message"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "get_stock_price":
        symbol = arguments["symbol"]
        price = fetch_price(symbol)  # 你的实现
        return [TextContent(type="text", text=f"{symbol}: ¥{price}")]

    elif name == "send_notification":
        msg = arguments["message"]
        send_msg(msg)  # 你的实现
        return [TextContent(type="text", text=f"已发送: {msg}")]

async def main():
    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### 在 LangGraph 中使用自定义 Server

```python
server_params = StdioServerParameters(
    command="python",
    args=["my_mcp_server.py"]
)

tools = await get_mcp_tools(server_params)
# tools = [get_stock_price, send_notification]
```

---

## 多 MCP Server 组合

```python
async def load_all_tools():
    """从多个 MCP Server 加载工具"""
    servers = [
        StdioServerParameters(command="python", args=["-m", "mcp_server_filesystem", "/workspace"]),
        StdioServerParameters(command="python", args=["-m", "mcp_server_github"]),
        StdioServerParameters(command="python", args=["my_custom_server.py"]),
    ]

    all_tools = []
    for params in servers:
        tools = await load_mcp_tools(params)
        all_tools.extend(tools)

    return all_tools

# 使用所有工具
tools = await load_all_tools()
agent = create_react_agent(llm, tools)
```

---

## SSE 模式（远程 MCP）

```python
from mcp.client.sse import sse_client

# 连接远程 MCP Server
async def connect_remote():
    async with sse_client("http://mcp-server.example.com/sse") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            return tools
```

---

## MCP vs 传统工具

| 维度 | 传统 @tool | MCP |
|------|-----------|-----|
| 集成方式 | 直接在代码中定义 | 独立进程/服务 |
| 语言 | 必须 Python | 任意语言 |
| 动态发现 | ❌ 编译时确定 | ✅ 运行时发现 |
| 复用性 | 项目内 | 跨项目 |
| 部署 | 嵌入应用 | 独立部署 |
| 性能 | 直接调用 | IPC 通信 |
| 适合 | 简单工具 | 复杂工具生态 |

---

## 小结

| 要点 | 说明 |
|------|------|
| MCP | 标准化的工具连接协议 |
| MCP Client | AI 应用端 |
| MCP Server | 工具提供端 |
| load_mcp_tools | 加载 MCP 工具到 LangGraph |
| 动态发现 | 运行时自动发现工具 |
| 多 Server | 可组合多个工具源 |

---

## 下一篇

➡️ [自定义工具开发](./05-自定义工具开发.md)
