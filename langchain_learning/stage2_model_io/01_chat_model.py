from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

MODEL_NAME = "deepseek-v4-flash"
MODEL_URL = "https://llm-lniax9q66senyb7s.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
MODEL_KEY = "sk-ws-H.EHHMLPX.GItO.MEUCIQDOAzCACqmt9lyJ1RrfpNRVVu9sgD9QAZRJo8cXtUKE1wIgbdfCUdVgwKvGPDb-wwa8-lj4kGXtiTJHjTLnNwUniZg"


def init_model():
    llm = init_chat_model(
        model=MODEL_NAME,
        base_url=MODEL_URL,
        model_provider="openai",
        temperature=0,
        api_key=MODEL_KEY,
        streaming=True,
    )
    return llm


def call_simple_llm():
    llm = init_model()
    resp = llm.invoke("你好")
    print(resp.content)


def call_message_llm():
    llm = init_model()
    messages =[
        SystemMessage(content="你是一个小孩，不会任何翻译"),
        HumanMessage(content="翻译：你好"),
    ]
    resp = llm.invoke(messages)
    print(resp.content)

def call_dict_llm():
    llm = init_model()
    messages = [
        {"role": "system", "content": "你是一个小孩，不会写诗"},
        {"role": "user", "content": "写一首关于夏天的诗"},
    ]
    resp = llm.invoke(messages)
    print("\n字典格式：\n", resp.content)
if __name__ == "__main__":
    call_dict_llm()
