"""
LangChain 结构化输出选型指南
🥇 优先级 1：with_structured_output(method="function_calling") (对应 Demo 3)
适用场景：90% 的生产环境业务（提取信息、Agent 工具调用）。
为什么好：利用大模型原生的 Tool Calling 机制，100% 不会发生 JSON 格式错误，不需要写复杂的 Prompt 约束，直接返回强类型的 Pydantic 对象。
前提：使用的模型必须支持 Function Calling（如 OpenAI, Claude, 通义千问, DeepSeek 等主流模型）。

🥈 优先级 2：with_structured_output(method="json_mode") (对应 Demo 4)
适用场景：模型不支持 Function Calling（比如某些极小的本地开源模型），但支持 JSON 模式输出。
坑点：必须在 Prompt 中明确写出 "请输出 JSON"，否则 API 会抛出 400 错误。

🥉 优先级 3：PydanticOutputParser + PromptTemplate (对应 Demo 2)
适用场景：非常老旧的模型，或者没有任何 Tool/JSON Mode 能力的接口。
坑点：极度依赖模型的智商。模型经常会回复：“好的，这是您的 JSON：json {...}”。
补救：必须配合 OutputFixingParser（Demo 8）或者在 Prompt 里加上严厉的警告（"Do NOT output markdown"）。

❌ 尽量不用：XMLOutputParser (对应 Demo 10)
原因：大模型是基于互联网文本训练的，JSON 在代码和 API 中占据统治地位，模型对 JSON 的语法直觉远好于 XML。用 XML 极容易出现标签未闭合、嵌套错误等令人抓狂的问题。

🛠️ 特殊场景：bind_tools (对应 Demo 5)
适用场景：开发 LangGraph / AutoGen 等 Agentic Workflow（多智能体工作流）。
用法：你不关心解析出的对象，你只关心大模型决定调用哪个函数，从而将流程路由到下一个节点。注意一定要做 if response.tool_calls: 的防空判断。
"""
import json
import os
from typing import TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import PromptTemplate

# 加载 .env 环境变量（建议将 API Key 写入 .env 文件：MODEL_KEY=sk-...）
load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-v4-flash")
MODEL_URL = os.getenv("MODEL_URL", "https://llm-lniax9q66senyb7s.cn-beijing.maas.aliyuncs.com/compatible-mode/v1")
MODEL_KEY = os.getenv("MODEL_KEY",
                      "sk-ws-H.EHHMLPX.GItO.MEUCIQDOAzCACqmt9lyJ1RrfpNRVVu9sgD9QAZRJo8cXtUKE1wIgbdfCUdVgwKvGPDb-wwa8-lj4kGXtiTJHjTLnNwUniZg")

# 初始化统一的 LLM 实例（推荐使用 LangChain 0.2+ 的 init_chat_model）
llm = init_chat_model(
    model=MODEL_NAME,
    base_url=MODEL_URL,
    model_provider="openai",
    temperature=0,
    api_key=MODEL_KEY
)


def demo_1_str_output_parser():
    """
    【基础场景】StrOutputParser - 纯文本提取
    适用场景：只需要模型返回纯字符串，自动去除 Message 对象包装。
    """
    print("\n=================== Demo 1: StrOutputParser ===================")

    chain = llm | StrOutputParser()
    text = chain.invoke("你好，用一句话介绍你自己")

    print(f"➜ 返回类型: {type(text)}")
    print(f"➜ 解析内容: {text}")


def demo_2_pydantic_output_parser():
    """
    【传统 Prompt 方式】PydanticOutputParser + PromptTemplate
    适用场景：模型不支持 Native Function Calling 时，利用 PromptTemplate 生成
             format_instructions 提示词，让模型强行输出规范 JSON 并由解析器反序列化。
    """
    print("\n=================== Demo 2: PydanticOutputParser ===================")

    # 1. 定义期望的 Schema
    class Actor(BaseModel):
        name: str = Field(description="演员姓名")
        filmography: list[str] = Field(description="代表作列表")

    # 2. 构建解析器与带指令的提示词模板
    parser = PydanticOutputParser(pydantic_object=Actor)
    prompt = PromptTemplate(
        template="请回答用户问题。\n{format_instructions}\n问题：{query}",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    # 3. 组装 Chain 执行
    chain = prompt | llm | parser
    result: Actor = chain.invoke({"query": "生成周星驰的简单资料"})

    print(f"➜ 返回类型: {type(result)}")
    print(f"➜ 演员姓名: {result.name}")
    print(f"➜ 代表作品: {result.filmography}")


def demo_3_structured_output_basemodel():
    """
    【最推荐方式】with_structured_output + Pydantic BaseModel
    适用场景：模型支持 Function Calling / Tool Calling
    优势：无需手写 format_instructions，返回强类型 Pydantic 对象
    """
    from pydantic import BaseModel, Field
    from langchain_openai import ChatOpenAI

    print("\n=================== Demo 3: with_structured_output (BaseModel) ===================")

    class Joke(BaseModel):
        setup: str = Field(description="铺垫/前情提要")
        punchline: str = Field(description="笑点/包袱")

    structured_llm = llm.with_structured_output(
        Joke,
        method="function_calling",
        tool_choice="auto",
    )

    result = structured_llm.invoke("讲一个关于程序员的冷笑话")

    print(f"➜ 返回数据类型: {type(result)}")
    print(f"➜ 对象属性读取 -> 铺垫: {result.setup} | 包袱: {result.punchline}")
    print(f"➜ 转化为 Python Dict: {result.model_dump()}")
    print(f"➜ 转化为 JSON 字符串:\n{result.model_dump_json(indent=2)}")


def demo_4_structured_output_typed_dict():
    """
    【轻量选择】with_structured_output + TypedDict
    适用场景：不需要 Pydantic 校验，仅需轻量级字典结构。
    """
    from typing import TypedDict
    from langchain_openai import ChatOpenAI

    print("\n=================== Demo 4: with_structured_output (TypedDict) ===================")

    class Person(TypedDict, total=True):
        name: str
        age: int

    # 显式指定 function_calling，避免不同模型 Provider 行为不一致
    structured_llm = llm.with_structured_output(
        Person,
        method="function_calling",
        include_raw=False,
        tool_choice="auto",
    )

    result: Person = structured_llm.invoke("张三今年 28 岁")

    print(f"➜ 返回类型: {type(result)}")
    print(f"➜ 字典结果: {result}")
    print(f"➜ 取值: 姓名={result['name']}, 年龄={result['age']}")


def demo_4_structured_output_typed_dict_v2():
    """
    【轻量选择】with_structured_output + TypedDict
    适用场景：不需要 Pydantic 校验，仅需轻量级字典结构。
    """
    from typing import TypedDict
    from langchain_openai import ChatOpenAI

    print("\n=================== Demo 4: with_structured_output (TypedDict) ===================")

    class Person(TypedDict, total=True):
        name: str
        age: int

    # 使用json_mode 时
    # ①要注意 prompt 中必须要提到JSON，否则报错；
    # ②没有函数调用能力；
    # ③某些模型thinking模式只支持这个
    structured_llm = llm.with_structured_output(
        Person,
        method="json_mode",
    )

    result: Person = structured_llm.invoke("JSON 张三今年 28 岁")

    print(f"➜ 返回类型: {type(result)}")
    print(f"➜ 字典结果: {result}")
    print(f"➜ 取值: 姓名={result['name']}, 年龄={result['age']}")


def demo_5_bind_tools_extraction():
    """
    【底层控制】bind_tools / tool_choice 原生工具绑定提取
    适用场景：多工具切换、工作流控制或不需要直接返回 Schema 对象，只需抓取模型触发的 Tool Call 参数。
    """
    print("\n=================== Demo 5: bind_tools & tool_choice ===================")

    class ExtractUserInfo(BaseModel):
        """获取文本中的用户信息"""
        name: str = Field(description="姓名")
        email: str = Field(description="电子邮箱")

    # 将 Pydantic 绑定为 Tool，并强制要求模型调用该 Tool
    llm_with_tools = llm.bind_tools([ExtractUserInfo], tool_choice="auto")
    response = llm_with_tools.invoke("我的名字是李四，邮箱是 lisi@example.com")

    # 从返回的 AIMessage 中提取 tool_calls 参数
    tool_call = response.tool_calls[0]
    print(f"➜ 调用的工具名称: {tool_call['name']}")
    print(f"➜ 解析出的结构化参数: {tool_call['args']}")


def demo_6_json_output_parser():
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import PromptTemplate
    print("\n=================== Demo 6: JsonOutputParser ===================")

    prompt = PromptTemplate.from_template(
        "请用 JSON 返回一个人的姓名和年龄。\n"
        "字段：name, age\n"
        "问题：{query}"
    )

    parser = JsonOutputParser()

    chain = prompt | llm | parser
    result = chain.invoke({"query": "张三今年28岁"})

    print(f"➜ 返回类型: {type(result)}")
    print(f"➜ 结果: {result}")


def demo_8_output_fixing_parser():
    from pydantic import BaseModel, Field
    from langchain_core.output_parsers import PydanticOutputParser
    # TODO ModuleNotFoundError: No module named 'langchain.output_parsers'
    from langchain.output_parsers import OutputFixingParser
    from langchain_core.prompts import PromptTemplate

    print("\n=================== Demo 8: OutputFixingParser ===================")

    class Actor(BaseModel):
        name: str = Field(description="演员姓名")
        filmography: list[str] = Field(description="代表作列表")

    parser = PydanticOutputParser(pydantic_object=Actor)

    fixing_parser = OutputFixingParser.from_llm(
        parser=parser,
        llm=llm
    )

    # 故意构造一个残缺 JSON
    bad_json = '{"name": "周星驰", "filmography": ["大话西游", "功夫"]'

    result = fixing_parser.parse(bad_json)

    print(f"➜ 修复后类型: {type(result)}")
    print(f"➜ 演员姓名: {result.name}")
    print(f"➜ 代表作品: {result.filmography}")


def demo_10_xml_output_parser():
    from langchain_core.output_parsers import XMLOutputParser
    """
    pip install defusedxml
    """
    print("\n=================== Demo 10: XMLOutputParser ===================")

    parser = XMLOutputParser(tags=["person"])
    chain = (
            PromptTemplate.from_template("用 XML 输出一个人名和年龄：{query}")
            | llm
            | parser
    )

    result = chain.invoke({"query": "张三今年28岁"})
    print(result)


if __name__ == "__main__":
    # demo_1_str_output_parser()
    # demo_2_pydantic_output_parser()
    demo_3_structured_output_basemodel()
    # demo_4_structured_output_typed_dict()
    # demo_4_structured_output_typed_dict_v2()
    # demo_5_bind_tools_extraction()
    # demo_6_json_output_parser()
    # demo_8_output_fixing_parser()
    # demo_10_xml_output_parser()
