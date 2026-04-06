# DeepSeek Chat（AI 助手）

基于 **FastAPI + Vue 3 + Vite** 的对话应用：前端为类 DeepSeek 的聊天界面；后端使用 **LangGraph Agent**（DeepSeek 兼容接口），支持**工具调用**与**多轮会话**（`session_id` / `thread_id`），并集成 **RAG 本地知识库**（Chroma + 混合检索 + 可选重排）。

---

## 功能概览

| 能力 | 说明 |
|------|------|
| 对话 | 非流式 `POST /api/chat`、流式 `POST /api/chat/stream`（SSE，`data: {"content":...}`） |
| 多轮记忆 | LangGraph `InMemorySaver`，与前端 `session_id` 对齐为 `thread_id` |
| 工具 | 天气（高德）、股票、`search_local_knowledge`（RAG，由模型按需调用） |
| RAG | `backend/data` 下 `.txt` 入库；BM25（jieba）+ 向量 MMR，RRF 融合；可选 CrossEncoder 精排 |
| 前端 | Markdown 渲染（marked + DOMPurify）、流式打字机、侧边栏会话、模型与「一次性 / 流式」切换 |

---

## 仓库结构

```
ai-project/
├── backend/
│   ├── main.py                 # FastAPI 入口（默认 :8000）
│   ├── api/chat.py             # /api/chat、/stream、/history、/models
│   ├── agents/assistant.py     # LangGraph create_agent + 单例 + checkpointer
│   ├── chains/chat_chain.py    # RunnableWithMessageHistory（历史清理等）
│   ├── core/config.py          # 环境变量
│   ├── core/llm.py             # ChatOpenAI（DeepSeek）
│   ├── tools/                  # 天气、股票、search_txt_tool（RAG）
│   ├── rag/                    # 嵌入、加载、混合检索、Chroma、重排
│   ├── memory/                 # 预留 / Redis 等扩展
│   └── data/                   # RAG 原始 txt（可增删后按需重建向量库）
├── frontend/
│   ├── src/
│   │   ├── components/ChatInterface.vue
│   │   ├── api/chat.js         # axios + fetch(SSE)，VITE_API_BASE_URL
│   │   └── styles/
│   ├── vite.config.js          # 开发端口 3000，@ → src
│   └── package.json
├── test_rag.py                 # 根目录 RAG 实验脚本（独立跑通检索链路时可参考）
├── .env                        # 本地创建（勿提交），见下表
└── README.md
```

运行后端时工作目录一般为 **`backend/`**；Chroma 持久化目录默认为 **`backend/chroma_db`**（见 `rag/vectorstores/chroma_client.py`）。

---

## 环境要求

- **Node.js** 建议 **18+**（Vite 5）
- **Python** 建议 **3.11+**
- 网络：需能访问 DeepSeek API；RAG 嵌入使用**智谱**时需能访问智谱 API

---

## 环境变量

在项目根目录创建 `.env`（或仅在运行前导出），后端通过 `python-dotenv` 加载（`backend/core/config.py`）。

| 变量 | 说明 |
|------|------|
| `DEEPSEEK_API_KEY` | 必填：对话模型 |
| `DEEPSEEK_BASE_URL` | 可选，默认 `https://api.deepseek.com` |
| `MODEL_NAME` | 可选，默认 `deepseek-chat` |
| `ZHIPU_API_KEY` | RAG 向量嵌入（`ZhipuAIEmbeddings`）需要 |
| `AMAP_KEY` | 天气工具（高德 REST）需要 |
| `USE_RERANKER` | 设为 `0` / `false` 可关闭 CrossEncoder 精排，加快启动、省资源 |
| `RAG_FORCE_REBUILD` | 设为 `1` / `true` 时强制用文档重建 Chroma（否则复用 `chroma_db`） |
| `LANGSMITH_API_KEY` / `LANGSMITH_TRACING` | 可选，LangSmith 追踪 |

前端可选：`VITE_API_BASE_URL`（默认 `http://localhost:8000`）。

---

## 安装与运行

### 后端

```bash
cd backend
# 建议使用虚拟环境，并安装依赖（示例，可按实际环境调整）
pip install fastapi uvicorn python-dotenv pydantic requests
pip install langchain langchain-openai langchain-community langchain-chroma langgraph
pip install rank-bm25 jieba numpy sentence-transformers zhipuai
python main.py
```

服务默认：**http://127.0.0.1:8000**，交互文档：**http://127.0.0.1:8000/docs**。

### 前端

```bash
cd frontend
npm install
npm run dev
```

开发服务器默认：**http://localhost:3000**（见 `vite.config.js`）。

### 生产构建前端

```bash
cd frontend
npm run build
# 产物在 frontend/dist，由任意静态服务器或 Nginx 托管；需将 API 指到后端（`VITE_API_BASE_URL`）
```

---

## API 摘要

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/chat` | 非流式，请求体含 `message`、`model`、`session_id` |
| `POST` | `/api/chat/stream` | SSE 流式；行内 JSON `content`，结束 `data: [DONE]` |
| `DELETE` | `/api/chat/history/{session_id}` | 清空链路与 Agent 检查点中该会话 |
| `GET` | `/api/chat/models` | 返回前端可选模型列表 |

---

## 架构说明（简）

1. **Agent**：`create_assistant()` 使用 `create_agent` + `InMemorySaver`；同一进程内对默认配置复用**单例 Agent**，减少重复构建。
2. **工具**：模型根据提示词自行决定是否调用；`search_local_knowledge` 首次调用时会 `ensure_rag()` 加载文档并建/打开 Chroma。
3. **会话**：请求中的 `session_id` 会传入 `configurable.thread_id`，与前端侧边栏会话一致。
4. **前端流式**：`ChatInterface.vue` 对 SSE 分片做队列 + 帧动画，避免大块文本一次性贴出（打字机观感）。

---

## RAG 与数据

- 将 `.txt` 放入 **`backend/data/`**（或按 `loader` 中路径调整）。
- 首次检索或重建时可能较慢（嵌入 + 可选重排模型下载）；生产可预热或关闭重排。
- 根目录 **`test_rag.py`** 用于独立验证检索与向量流程，与主服务解耦。

---

## 注意事项

1. 对话依赖 **DeepSeek API Key**；天气、RAG 嵌入分别依赖 **高德**、**智谱** Key，未配置时对应能力会失败或不可用。
2. 历史与检查点均在**内存**中，进程重启后丢失；生产环境可替换为持久化 Checkpointer / 数据库。
3. CORS 当前为宽松配置（`allow_origins=["*"]`），上线请按域名收紧。

## License

MIT
