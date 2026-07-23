"""自定义实现MCP服务"""

from mcp.server.fastmcp import FastMCP

# 1. 创建 MCP 服务实例
mcp = FastMCP(
    name="MyCustomMCPServer",
    instructions="这是一个自定义的 MCP 服务，提供数学计算与系统日志查询功能。"
)


# 2. 定义 Tool (大模型可调用的函数)
@mcp.tool()
def multiply_numbers(a: float, b: float) -> float:
    """计算两个数的乘积

    Args:
        a: 第一个乘数
        b: 第二个乘数
    """
    return a * b


@mcp.tool()
def get_system_status() -> str:
    """获取当前系统的运行状态"""
    return "System Operational | CPU: 12% | Memory: 45%"


# 3. 定义 Resource (供大模型读取的静态/动态数据资源)
@mcp.resource("system://logs/latest")
def get_latest_logs() -> str:
    """读取最新的系统日志"""
    return "[INFO] 2026-07-23 10:00:00 - Service initialized successfully."


# 4. 定义 Prompt (内置的 Prompt 模板)
@mcp.prompt()
def code_review_prompt(language: str = "python") -> str:
    """生成针对特定语言的代码审查提示词"""
    return f"请作为高级 {language} 工程师，审查以下代码的性能、安全性和可读性。"


# 5. 启动服务
if __name__ == "__main__":
    # 默认使用 Stdio 方式启动（本地 CLI / Client 常用）
    mcp.run(transport="stdio")
    # 默认监听 127.0.0.1:8000
    #mcp.run(transport="streamable-http")