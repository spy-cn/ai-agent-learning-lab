import os

from langchain_openai import ChatOpenAI

LLM_API_KEY = "lanbigdata-key-1234"
LLM_API_BASE = "http://192.168.1.99:8562/v1"

# 1、获取本地部署的Qwen3-32B
chat_model = ChatOpenAI(
    model_name="Qwen3-32B",
    openai_api_key=LLM_API_KEY,
    openai_api_base=LLM_API_BASE,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    temperature=0.1
)

# 2、调用模型
response_invoke = chat_model.invoke("什么是langchain")
print(response_invoke)

import dotenv

# 加载配置文件
dotenv.load_dotenv()

# 1、获取本地部署的Qwen3-32B
chat_model_env = ChatOpenAI(
    model_name="Qwen3-32B",
    openai_api_key=os.getenv("OPEN_AI_KEY"),
    openai_api_base=os.getenv("OPEN_BASE_URL"),
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    temperature=0.1
)

# 2、调用模型
response_invoke_env = chat_model_env.invoke("什么是langchain")
print(response_invoke_env)
