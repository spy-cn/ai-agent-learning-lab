from datetime import datetime
from typing import TypedDict, Optional, Annotated, List

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode


# ========== 工具定义 ==========


@tool(description="两个数的乘法")
def multiply(a: int, b: int) -> int:
    return a * b
print("获取工具的名称：", multiply.name)
print("获取工具的描述：", multiply.description)
print("获取工具的参数：", multiply.args)

@tool(description="a乘以b的最大值")
def multiply_by_max(a: Annotated[int, "比例因子"], b: Annotated[List[int], "要取最大值的整型数列表"]) -> int:
    return a * max(b)


@tool
def calculator(expression: str) -> str:
    """执行数学计算，支持加减乘除和幂运算"""
    try:
        allowed_chars = set('0123456789+-*/(). ')
        if not all(c in allowed_chars for c in expression):
            return "表达式包含非法字符"
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


@tool
def get_time() -> str:
    """获取当前日期和时间"""
    now = datetime.now()
    print(now)
    result = now.strftime("%Y年%m月%d日 %H时%M分%S秒")
    print(f"DEBUG: get_time工具被调用，返回: {result}")  # 调试信息
    return result


@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息"""
    weather_data = {
        "北京": "晴, 25°C, 湿度40%",
        "上海": "多云, 23°C, 湿度60%",
        "广州": "雨, 28°C, 湿度85%",
        "深圳": "阴, 26°C, 湿度70%"
    }
    return weather_data.get(city, f"抱歉，没有{city}的天气信息")


tools = [calculator, get_time, get_weather]

# ========== 初始化LLM ==========
LLM_API_KEY = "lanbigdata-key-1234"
LLM_API_BASE = "http://192.168.1.99:8562/v1"
llm = ChatOpenAI(
    model_name="Qwen3-32B",
    openai_api_key=LLM_API_KEY,
    openai_api_base=LLM_API_BASE,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    temperature=0.1
)


# ========== LangGraph 工作流 ==========
class AgentState(TypedDict):
    messages: list  # 存储消息历史
    question: Optional[str]  # 当前问题


# 创建 ToolNode
tool_node = ToolNode(tools)


# 定义Agent节点
def run_agent(state: AgentState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个有用的助手，可以访问工具来回答问题。"),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    response = agent_executor.invoke({
        "input": state["question"],
        "chat_history": state["messages"],
        "agent_scratchpad": []
    })
    return {"messages": state["messages"] + [HumanMessage(content=state["question"]),
                                             AIMessage(content=response["output"])]}


# 构建工作流
workflow = StateGraph(AgentState)
workflow.add_node("agent", run_agent)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")


# 定义条件分支
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    # 检查工具调用（兼容不同消息格式）
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    if hasattr(last_message, "additional_kwargs") and last_message.additional_kwargs.get("tool_calls"):
        return "tools"
    return END  # 关键修改：返回 END 而不是 "end"


# 关键修改：使用 add_conditional_edges 的正确方式
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",  # 如果返回 "tools"，跳转到 tools 节点
        END: END  # 如果返回 END，结束工作流
    }
)

workflow.add_edge("tools", "agent")  # 工具调用后返回Agent

app = workflow.compile()

# ========== 测试 ==========
if __name__ == "__main__":
    test_questions = [
        "计算 (15 + 8) * 3 等于多少？",
        "现在是什么时间？",
        "深圳的天气怎么样？",
        "先告诉我时间，然后计算365 * 24等于多少，最后查询北京的天气"
    ]

    for question in test_questions:
        print(f"\n问题: {question}")
        result = app.invoke({"messages": [], "question": question})
        last_message = result["messages"][-1]
        print(f"回答: {last_message.content}")
