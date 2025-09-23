import os
from typing import Optional

import dotenv
from langchain.chains.sequential import SimpleSequentialChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict


# 自定义要返回的结构化数据Joke类
class JokeBase(BaseModel):
    """Joke to tell user."""

    setup: str = Field(description="笑话的开场部分")
    punchline: str = Field(description="笑话的妙语结尾")
    rating: Optional[int] = Field(
        default=None, description="笑话的有趣程度评分，范围从1到10"
    )

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




structured_llm = llm.with_structured_output(JokeBase)
llm_invoke = structured_llm.invoke("讲一个关于猫咪的笑话")
print(llm_invoke)


# 支持流式进行处理
class JokeType(TypedDict):
    """Joke to tell user."""

    setup: Annotated[str, ..., "笑话的开场部分"]
    punchline: Annotated[str, ..., "笑话的妙语结尾"]
    rating: Annotated[Optional[int], None, "笑话的有趣程度评分，范围从1到10"]
type_structured_llm = llm.with_structured_output(JokeType)
type_llm_invoke = type_structured_llm.invoke("讲一个关于猫咪的笑话")
print(type_llm_invoke)

# 当模型指定为TYpeDict 类型或者 JsonSchem 时，可以进行流式输出

structured_llm = llm.with_structured_output(JokeType)

for chunk in structured_llm.stream("讲一个关于猫咪的笑话"):
    print(chunk)