# 实战：数据分析 Agent

构建一个能查询数据、生成图表、提取洞察的数据分析 Agent。

---

## 架构

```
用户分析请求 → [理解需求] → [生成SQL] → [执行查询]
                                              │
                                              ▼
                                         [数据分析]
                                              │
                                              ▼
                                         [生成图表]
                                              │
                                              ▼
                                         [提取洞察] → 报告
```

---

## 实现

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
import sqlite3
import json

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 模拟数据库
DB_PATH = "sales.db"


class AnalysisState(TypedDict):
    messages: Annotated[list, add_messages]
    user_request: str
    sql_query: str
    query_result: str
    analysis: str
    chart_path: str
    insights: list
    report: str


@tool
def execute_sql(query: str) -> str:
    """执行SQL查询"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        result = {
            "columns": columns,
            "rows": [list(r) for r in rows[:20]],
            "row_count": len(rows)
        }
        conn.close()
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return f"SQL错误: {e}"

@tool
def generate_chart(data_json: str, chart_type: str = "bar") -> str:
    """生成图表"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data = json.loads(data_json)
    columns = data["columns"]
    rows = data["rows"]

    fig, ax = plt.subplots(figsize=(10, 6))

    if chart_type == "bar":
        x = [str(r[0]) for r in rows]
        y = [r[1] if len(r) > 1 else 0 for r in rows]
        ax.bar(x, y)
        # 中国市场配色：涨为红
        ax.set_color_cycle(['#d62728', '#2ca02c', '#1f77b4'])
    elif chart_type == "line":
        x = range(len(rows))
        y = [r[1] for r in rows]
        ax.plot(x, y, 'r-o')  # 红色线条

    ax.set_xlabel(columns[0] if columns else "X")
    ax.set_ylabel(columns[1] if len(columns) > 1 else "Y")
    ax.set_title("数据分析图表")

    chart_path = "chart.png"
    plt.savefig(chart_path)
    plt.close()
    return f"图表已保存: {chart_path}"


def understand_request(state: AnalysisState) -> dict:
    """理解用户分析需求"""
    response = llm.invoke(
        f"你是数据分析师。数据库有表: sales(date, product, region, amount, quantity)。\n"
        f"用户请求: {state['user_request']}\n"
        f"生成SQL查询语句（只返回SQL）。"
    )
    sql = response.content.strip()
    if sql.startswith("```"):
        sql = sql.split("```")[1]
        if sql.startswith("sql"):
            sql = sql[3:]
    return {"sql_query": sql.strip()}

def run_query(state: AnalysisState) -> dict:
    """执行SQL"""
    result = execute_sql.invoke(state["sql_query"])
    return {"query_result": result}

def analyze_data(state: AnalysisState) -> dict:
    """分析查询结果"""
    response = llm.invoke(
        f"分析以下查询结果，提取关键发现:\n"
        f"SQL: {state['sql_query']}\n"
        f"结果: {state['query_result']}"
    )
    return {"analysis": response.content}

def create_chart(state: AnalysisState) -> dict:
    """生成图表"""
    chart_type = "bar"
    if "趋势" in state["user_request"] or "时间" in state["user_request"]:
        chart_type = "line"

    path = generate_chart.invoke({
        "data_json": state["query_result"],
        "chart_type": chart_type
    })
    return {"chart_path": path}

def extract_insights(state: AnalysisState) -> dict:
    """提取业务洞察"""
    response = llm.invoke(
        f"基于数据分析结果，给出3条业务洞察建议:\n{state['analysis']}"
    )
    import re
    insights = re.findall(r'\d+\.\s*(.+?)(?=\n\d+\.|$)', response.content, re.DOTALL)
    return {"insights": insights[:3]}

def compile_report(state: AnalysisState) -> dict:
    """汇总报告"""
    report = f"""
# 数据分析报告

## 分析请求
{state['user_request']}

## SQL查询
```sql
{state['sql_query']}
```

## 查询结果
{state['query_result'][:500]}...

## 分析
{state['analysis']}

## 图表
{state['chart_path']}

## 业务洞察
"""
    for i, insight in enumerate(state["insights"], 1):
        report += f"\n{i}. {insight.strip()}"

    return {"report": report}


# 构建
builder = StateGraph(AnalysisState)
builder.add_node("understand", understand_request)
builder.add_node("query", run_query)
builder.add_node("analyze", analyze_data)
builder.add_node("chart", create_chart)
builder.add_node("insights", extract_insights)
builder.add_node("report", compile_report)

builder.add_edge(START, "understand")
builder.add_edge("understand", "query")
builder.add_edge("query", "analyze")
builder.add_edge("analyze", "chart")
builder.add_edge("chart", "insights")
builder.add_edge("insights", "report")
builder.add_edge("report", END)

analysis_agent = builder.compile()
```

---

## 使用

```python
result = analysis_agent.invoke({
    "user_request": "分析2024年各地区的销售总额，找出最佳和最差地区",
    "messages": [],
    "sql_query": "", "query_result": "", "analysis": "",
    "chart_path": "", "insights": [], "report": ""
})

print(result["report"])
```

---

## 小结

| 要点 | 说明 |
|------|------|
| NL → SQL | 自然语言转SQL |
| 自动执行 | 安全执行SQL |
| 图表生成 | 自动选择图表类型 |
| 洞察提取 | LLM 业务分析 |
| 报告汇总 | 结构化输出 |

---

## 下一篇

➡️ [端到端项目模板](./05-端到端项目模板.md)
