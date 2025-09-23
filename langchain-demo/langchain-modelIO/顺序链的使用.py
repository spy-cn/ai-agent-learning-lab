import os

import dotenv
from langchain.chains.sequential import SimpleSequentialChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 加载配置文件
dotenv.load_dotenv()

# 初始化模型
llm = ChatOpenAI(
    model_name="Qwen3-32B",
    openai_api_key=os.getenv("OPEN_AI_KEY"),
    openai_api_base=os.getenv("OPEN_BASE_URL"),
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    temperature=0.1
)

chain_message_template_1 = ChatPromptTemplate.from_messages(
    messages=[("system", "你是一个精通各个领域的教授"), ("human", "请你尽可能详细的解释一下：{question}")])

llm_chain_1 = llm | chain_message_template_1

chain_message_template_2 = ChatPromptTemplate.from_messages(
    messages=[("system", "你非常擅长提取文本中的信息，并作出简短的总结"),
              ("human", "这是针对提问过后的完整的回答内容：{description}"),
              ("human", "请根据上述说明尽可能简短的说明内容 20字以内")])

llm_chain_2 = llm | chain_message_template_2

full_chain = SimpleSequentialChain(chains=[chain_message_template_1, chain_message_template_2])

response = full_chain.invoke(input={"input": "什么是langchain"})
print(response)
