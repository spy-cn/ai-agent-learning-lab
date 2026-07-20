from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

def prompt_template_str():
    """字符串模板"""
    print("=== PromptTemplate ===")
    template = PromptTemplate.from_template("请评价{product}的优缺点")
    prompt = template.format(product="智能手机")
    print(prompt)



def prompt_template_var():
    print("\n=== 部分变量 ===")
    template = PromptTemplate(
        template="{foo} {bar}",
        input_variables=["foo", "bar"],
        partial_variables={"foo": "hello"},
    )
    print(template.format(bar="world"))  # hello world


def prompt_template_chat():
    print("\n=== ChatPromptTemplate ===")
    chat_template = ChatPromptTemplate(
        [
            ("system", "你是一个{name}，你的专长是{skill}。"),
            ("human", "{question}"),
        ]
    )
    messages = chat_template.format_messages(
        name="营养师",
        skill="饮食搭配",
        question="早餐吃什么健康？",
    )
    for msg in messages:
        print(f"[{msg.type}] {msg.content}")


if __name__ == "__main__":
    prompt_template_chat()


