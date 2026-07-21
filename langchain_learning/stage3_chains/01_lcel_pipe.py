# LCEL 管道符
import json
import os
from typing import TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import PromptTemplate
load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-v4-flash")
MODEL_URL = os.getenv("MODEL_URL", "https://llm-lniax9q66senyb7s.cn-beijing.maas.aliyuncs.com/compatible-mode/v1")
MODEL_KEY = os.getenv("MODEL_KEY",
                      "sk-ws-H.EHHMLPX.GItO.MEUCIQDOAzCACqmt9lyJ1RrfpNRVVu9sgD9QAZRJo8cXtUKE1wIgbdfCUdVgwKvGPDb-wwa8-lj4kGXtiTJHjTLnNwUniZg")


prompt = PromptTemplate.from_template("讲一个关于{topic}的笑话")

llm = init_chat_model(
    model=MODEL_NAME,
    base_url=MODEL_URL,
    model_provider="openai",
    temperature=0,
    api_key=MODEL_KEY
)



parser = StrOutputParser()
chain = prompt | llm | parser
# 调用链
result = chain.invoke({"topic": "程序员"})
print("笑话：", result)

# 链自动支持 batch 和 stream
print("\n=== stream 流式 ===")
for chunk in chain.stream({"topic": "猫"}):
    print(chunk, end="", flush=True)
print()
