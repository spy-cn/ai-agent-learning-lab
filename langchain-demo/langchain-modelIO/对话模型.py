from langchain_core.messages import HumanMessage, SystemMessage
import os
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import dotenv
from langchain_core.prompts import ChatPromptTemplate

# 加载配置文件
dotenv.load_dotenv()

# 初始化模型
model = ChatOpenAI(
    model_name="Qwen3-32B",
    openai_api_key=os.getenv("OPEN_AI_KEY"),
    openai_api_base=os.getenv("OPEN_BASE_URL"),
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    temperature=0.1
)

messages = [
    # 定义大模型的角色
    SystemMessage(content="你是一个专业的翻译助手,能够将中文语句，翻译成合适的英文"),
    # 用户的输入信息
    HumanMessage(content="今天天气怎么样")
]

# 调用模型进行返回
response = model.invoke(messages)
print(type(response))
print(response)
# 输出解释器，可以处理模型响应内容
parser = StrOutputParser()
# 将模型的输出结果 传入给解释器
parser_invoke = parser.invoke(response)
print(parser_invoke)

# 模型与输出解释器联合 每次都会被调用
chain = model | parser
chain_invoke = chain.invoke(messages)
print(chain_invoke)

# PromptTemplates 接收原始用户输入的信息，并返回格式化语言传递给语言模型

system_template = "翻译内容为{language}"

# 提示词模板
template_from_messages = ChatPromptTemplate.from_messages([("system", system_template), ("user", "{text}")])
result= template_from_messages.invoke({"language": "英语", "text": "今天晚上准备吃什么"})
print(result)
print(result.to_messages())

#利用管道操作符将与模型连接起来
chain_template_from_message = template_from_messages | model | parser
from_message_invoke = chain_template_from_message.invoke({"language": "英文", "text": "明天天气会下雨吗？"})
print(from_message_invoke)
