import os

import dotenv
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

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

from_template = PromptTemplate.from_template(template="你是一个数学高手，帮我解决以下数学问题：{question}")

llm_chain = LLMChain(llm=model, prompt=from_template)
response = llm_chain.invoke(input={"question": "1+2+4*6=?"})
print(response)
