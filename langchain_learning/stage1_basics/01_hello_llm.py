import os
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "deepseek-v4-flash"
MODEL_URL = "https://llm-lniax9q66senyb7s.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
MODEL_KEY = "sk-ws-H.EHHMLPX.GItO.MEUCIQDOAzCACqmt9lyJ1RrfpNRVVu9sgD9QAZRJo8cXtUKE1wIgbdfCUdVgwKvGPDb-wwa8-lj4kGXtiTJHjTLnNwUniZg"


def init_model_by_openai():
    from openai import OpenAI

    client = OpenAI(
        base_url=os.getenv("OPENAI_API_URL", MODEL_URL),
        api_key=os.getenv("DASHSCOPE_MODEL_KEY", MODEL_KEY)
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "将‘你好’翻译成意大利语"}],
    )

    print("响应的类型", type(response))
    print("响应的内容", response)
    print("=" * 70)
    print(response.choices[0].message.content)


def init_model_by_chatopenai():
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(
        model=MODEL_NAME,
        openai_api_base=os.getenv("OPENAI_API_URL", MODEL_URL),
        openai_api_key=os.getenv("DASHSCOPE_MODEL_KEY", MODEL_KEY),
        temperature=0.3
    )

    messages = [
        SystemMessage(content="你是一个小孩，不会任何翻译"),
        HumanMessage(content="将‘你好’翻译成意大利语")
    ]

    # 调用模型
    response = model.invoke(messages)

    print("LangChain 响应的类型", type(response))
    print("LangChain 响应的内容", response)
    print("=" * 70)
    print(response.content)


def init_model_by_init_chat_model():
    from langchain.chat_models import init_chat_model
    model = init_chat_model(
        model=MODEL_NAME,
        base_url=os.getenv("DASHSCOPE_MODEL_URL", MODEL_URL),
        model_provider="openai",
        temperature=0,
        api_key=MODEL_KEY
    )
    response = model.invoke("你好！")
    print(response.content)


if __name__ == "__main__":
    init_model_by_init_chat_model()
