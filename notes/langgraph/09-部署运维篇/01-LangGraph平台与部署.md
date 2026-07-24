# LangGraph 平台与部署

将 LangGraph 应用从开发环境部署到生产环境。本篇介绍 LangGraph Cloud、自托管方案、Docker 部署等。

---

## 部署方案概览

```
┌──────────────────────────────────────────────────────┐
│                   部署方案选择                         │
├──────────────┬──────────────┬───────────────────────┤
│  LangGraph   │  Docker      │  完全自托管            │
│  Cloud       │  自托管       │  (FastAPI等)           │
├──────────────┼──────────────┼───────────────────────┤
│ 最简单       │ 中等复杂      │ 最灵活                 │
│ 自动扩展     │ 需手动管理    │ 完全控制               │
│ 内置监控     │ 自建监控      │ 自建一切               │
│ 有费用       │ 服务器成本    │ 服务器+开发成本          │
│ 适合快速上线  │ 适合中型项目  │ 适合大型/特殊需求       │
└──────────────┴──────────────┴───────────────────────┘
```

---

## 方案一：LangGraph Cloud

LangGraph Cloud 是 LangChain 团队提供的托管平台。

### 配置文件

```json
// langgraph.json
{
    "dependencies": ["."],
    "graphs": {
        "agent": "./src/graph.py:graph",
        "rag": "./src/rag_graph.py:rag_graph"
    },
    "env": ".env"
}
```

### 部署

```bash
# 安装 CLI
pip install langgraph-cli

# 本地开发
langgraph dev

# 部署到 Cloud
langgraph deploy
```

### 优势

- ✅ 零运维
- ✅ 自动扩展
- ✅ 内置监控（LangSmith 集成）
- ✅ 内置状态管理
- ✅ Studio 可视化
- ❌ 有月费
- ❌ 数据在第三方

---

## 方案二：Docker 部署

### Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY src/ ./src/
COPY langgraph.json .

# 暴露端口
EXPOSE 8000

# 启动
CMD ["langgraph", "serve", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  langgraph:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - LANGSMITH_API_KEY=${LANGSMITH_API_KEY}
    depends_on:
      - postgres
      - redis
    restart: unless-stopped

  postgres:
    image: postgres:16
    environment:
      POSTGRES_DB: langgraph
      POSTGRES_USER: langgraph
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    restart: unless-stopped

volumes:
  pgdata:
```

### 启动

```bash
# 构建并启动
docker-compose up -d

# 查看日志
docker-compose logs -f langgraph

# 停止
docker-compose down
```

---

## 方案三：FastAPI 自托管

### 最灵活的方案

```python
# app.py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
import json

app = FastAPI(title="LangGraph Agent API")

# 初始化图
DB_URI = "postgresql://langgraph:password@localhost/langgraph"
checkpointer = PostgresSaver.from_conn_string(DB_URI)
checkpointer.setup()

# ... 构建图 ...
graph = builder.compile(checkpointer=checkpointer)


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"


@app.post("/chat")
async def chat(req: ChatRequest):
    """同步对话"""
    config = {"configurable": {"thread_id": req.thread_id}}

    result = graph.invoke(
        {"messages": [("user", req.message)]},
        config=config
    )

    return {
        "response": result["messages"][-1].content,
        "thread_id": req.thread_id
    }


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """流式对话"""
    config = {"configurable": {"thread_id": req.thread_id}}

    async def generate():
        async for event in graph.astream(
            {"messages": [("user", req.message)]},
            config=config,
            stream_mode="messages"
        ):
            msg, metadata = event
            if msg.content:
                yield f"data: {json.dumps({'content': msg.content})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/history/{thread_id}")
async def get_history(thread_id: str):
    """获取对话历史"""
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config)

    messages = []
    for msg in state.values.get("messages", []):
        messages.append({
            "type": msg.type,
            "content": msg.content
        })

    return {"messages": messages}


@app.get("/health")
async def health():
    return {"status": "ok"}
```

### 启动

```bash
# 使用 uvicorn（生产推荐）
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4

# 或使用 gunicorn + uvicorn worker
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

---

## 生产环境配置

### 环境变量管理

```python
# config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # LLM
    openai_api_key: str
    model_name: str = "gpt-4o"

    # 数据库
    database_url: str

    # Redis
    redis_url: str = "redis://localhost:6379"

    # 应用
    max_recursion: int = 25
    request_timeout: int = 60
    max_concurrent: int = 100

    # LangSmith
    langsmith_api_key: str = ""
    langsmith_project: str = "production"

    class Config:
        env_file = ".env"

settings = Settings()
```

### 日志配置

```python
import logging
import sys

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('app.log')
        ]
    )

    # 降低第三方库日志级别
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
```

### 速率限制

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/chat")
@limiter.limit("10/minute")  # 每分钟最多10次
async def chat(request: Request, req: ChatRequest):
    ...
```

---

## 水平扩展考虑

### 无状态 vs 有状态

```python
# 问题：LangGraph Agent 有状态（Checkpoint）
# 多实例部署时，状态需要共享存储

# 解决方案：使用外部 Checkpoint 存储
checkpointer = PostgresSaver.from_conn_string(DB_URI)
# 所有实例共享同一个数据库 → 状态一致
```

### 负载均衡

```nginx
# nginx.conf
upstream langgraph_backend {
    server app1:8000;
    server app2:8000;
    server app3:8000;
}

server {
    listen 80;

    location / {
        proxy_pass http://langgraph_backend;
        proxy_set_header Host $host;
    }

    location /chat/stream {
        proxy_pass http://langgraph_backend;
        proxy_buffering off;  # SSE 需要关闭缓冲
        proxy_cache off;
    }
}
```

---

## 小结

| 方案 | 适合 | 复杂度 | 成本 |
|------|------|--------|------|
| LangGraph Cloud | 快速上线 | 低 | 月费 |
| Docker 部署 | 中型项目 | 中 | 服务器 |
| FastAPI 自托管 | 大型/定制 | 高 | 服务器+开发 |

---

## 下一篇

➡️ [持久化与状态存储](./02-持久化与状态存储.md)
