# AI-Agent-Learning-Lab

针对系统学习LangChain 和 LangGraph的项目。

---

### 2.代码结构
```text
my-ai-learning-lab/           # 项目根目录
├── README.md                 # 项目总体介绍、学习路线图、环境配置说明
├── .gitignore                # Git 忽略文件
├── pyproject.toml            # (可选) 根目录的依赖管理，统一管理所有子项目的公共依赖
├── .env.example              # 环境变量示例文件 (如 OPENAI_API_KEY)
│
├── notes/                    # (可选) 存放你的学习笔记、思维导图、架构图等
│   └── images/
│
├── fundamentals/             # 1. 基础篇：LangChain 核心概念练习
│   ├── 01_models_prompts/    # 模型、提示词模板
│   ├── 02_chains/            # 链的使用
│   ├── 03_retrieval/         # 检索器 (RAG 基础)
│   ├── 04_memory/            # 记忆模块
│   └── 05_tools_agents/      # 工具和 Agent (ReAct) [reference:5]
│
├── langgraph-in-action/      # 2. 进阶篇：LangGraph 状态机与复杂流程
│   ├── 01_basic_chatbot/     # 基础聊天机器人[reference:6]
│   ├── 02_human_in_loop/     # 人工介入流程[reference:7]
│   ├── 03_multi_agent/       # 多智能体系统[reference:8]
│   └── 04_checkpointer/      # 状态持久化与记忆[reference:9]
│
├── projects/                 # 3. 实战篇：综合性项目
│   ├── project_rag_qa/       # 基于 RAG 的文档问答系统[reference:10]
│   ├── project_research_assistant/ # 研究助手[reference:12]
│   └── project_customer_service/   # 多智能体客服系统[reference:13]
│
└── apps/                     # 4. 部署篇：可部署的应用 (可选，展示如何打包和部署)
    ├── agent1/               # 第一个可部署的 Agent
    │   ├── langgraph.json    # LangGraph 部署配置文件[reference:15]
    │   ├── pyproject.toml    # Agent 专属依赖
    │   └── src/agent1/
    │       ├── graph.py      # 图定义
    │       └── state.py      # 状态定义
    └── agent2/               # 第二个可部署的 Agent
        └── ...
```