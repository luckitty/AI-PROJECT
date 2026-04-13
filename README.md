# DeepSeek Chat（AI 助手）

基于 **FastAPI + Vue 3 + Vite** 的对话应用：前端为类 DeepSeek 的聊天界面；后端对话主路径为 **LangGraph 节点编排**（`planner` → `memory` / `rag` / `tool` → `response` → `save_memory`），底层模型走 **DeepSeek 兼容 OpenAI 接口**。支持 **短期会话检查点（Redis）**、**本地 RAG（Milvus + 混合检索 + 可选重排）** 与 **天气 / 演示股价 / 联网搜索** 等工具。

---

## 功能概览

| 能力 | 说明 |
|------|------|
| 对话 | 非流式 `POST /api/chat`、流式 `POST /api/chat/stream`（SSE，行内 `data: {"content":...}`，结束 `data: [DONE]`） |
| 请求体 | `message`、`model`、`session_id`、`user_id`（可选，长期记忆等场景用） |
| 编排 | `planner` 输出 `need_rag` / `need_tool` / `need_memory`，动态路由到对应节点后再汇总到 `response` |
| 短期记忆 | LangGraph 使用 **RedisSaver** 检查点，按 `thread_id`（与前端 `session_id` 对齐）持久化；需 **Redis Stack**（含 RediSearch + RedisJSON） |
| 工具 | `tool` 节点：`get_weather`（高德）、`get_stock_price`（演示数据）、`web_search` |
| RAG | `rag` 节点：`data` 目录文档 → 嵌入（智谱）→ **Milvus** 向量库；BM25 + 向量混合检索，可选 CrossEncoder 精排（`USE_RERANKER`） |
| 前端 | Markdown（marked + DOMPurify）、流式打字机、侧边栏会话、模型与非流式/流式切换 |

---

## 仓库结构（要点）

```
ai-project/
├── backend/
│   ├── main.py                    # FastAPI 入口（默认 :8000）
│   ├── api/chat.py                # /api/chat、/stream、/history、/models
│   ├── graph/
│   │   ├── builder.py             # StateGraph：planner → … → response → save_memory
│   │   ├── orchestrator.py        # AgentOrchestrator：invoke / stream
│   │   ├── router.py              # route_by_plan
│   │   └── state.py               # AgentState
│   ├── agents/                    # planner、memory、rag、tool、response、save_memory
│   ├── agents/assistant.py        # create_agent + 工具清单（与 ToolRegistry 对齐；图编排可与单 Agent 方案并存）
│   ├── chains/chat_chain.py       # RunnableWithMessageHistory（内存历史清理等）
│   ├── core/config.py             # API Key / 模型等（dotenv）
│   ├── core/llm.py                # ChatOpenAI（DeepSeek）
│   ├── memory/
│   │   ├── short_memory.py        # RedisSaver 检查点
│   │   ├── redis_config.py        # REDIS_URL、TTL
│   │   └── long_memory*.py        # 长期记忆相关（按需启用）
│   ├── rag/                       # 加载、混合检索、Milvus、重排
│   ├── tools/                     # 天气、股票、联网搜索、ToolRegistry
│   ├── data/                      # RAG 原始文本（按需增删后重建/写入 Milvus）
│   └── requirements.txt           # 部分依赖声明（见下「安装」）
├── frontend/
│   ├── src/components/ChatInterface.vue
│   ├── src/api/chat.js            # axios + fetch(SSE)，VITE_API_BASE_URL
│   └── vite.config.js             # 开发端口 3000
└── README.md
```

运行后端时工作目录一般为 **`backend/`**（`python main.py`）。RAG 向量存储当前默认走 **Milvus**（见 `rag/retriever.py`、`rag/vectorstores/milvus_client.py`）；Chroma 相关代码保留为可选切换。

---

## 环境要求

- **Node.js** 建议 **18+**（Vite 5）
- **Python** 建议 **3.11+**
- **Redis**：需 **Redis Stack**（含 RediSearch、RedisJSON），供 LangGraph `RedisSaver` 建索引；URL 见 `backend/memory/redis_config.py`
- **Milvus**：RAG 默认连本机或配置的 Milvus 服务（`MILVUS_HOST` / `MILVUS_PORT` 等）
- 网络：需能访问 **DeepSeek API**；嵌入使用 **智谱** 时需能访问智谱 API；天气依赖 **高德** Key

---

## 环境变量

在项目根目录或 `backend/` 旁创建 **`.env`**，由 `python-dotenv` 加载（`backend/core/config.py`）。

| 变量 | 说明 |
|------|------|
| `DEEPSEEK_API_KEY` | 必填：对话模型 |
| `DEEPSEEK_BASE_URL` | 可选，默认 `https://api.deepseek.com` |
| `MODEL_NAME` | 可选，默认 `deepseek-chat` |
| `ZHIPU_API_KEY` | RAG 向量嵌入（智谱 `embedding-3`）需要 |
| `AMAP_KEY` | 天气工具（高德）需要 |
| `USE_RERANKER` | 设为 `0` / `false` 等可关闭 CrossEncoder 精排，加快启动、省资源 |
| `LANGSMITH_API_KEY` / `LANGSMITH_TRACING` | 可选，LangSmith 追踪 |

**Milvus（RAG）**（见 `milvus_client.py`）：

| 变量 | 说明 |
|------|------|
| `MILVUS_HOST` | 默认 `127.0.0.1` |
| `MILVUS_PORT` | 默认 `19530` |
| `MILVUS_COLLECTION` | 集合名，默认 `rag_collection` |
| `MILVUS_SKIP_INGEST` | 为 `1`/`true` 时跳过写入，仅连接已有集合 |
| `MILVUS_DROP_OLD` | 为 `1`/`true` 时写入前删除旧集合数据（按实现语义使用） |

**Redis**：连接串在 `memory/redis_config.py` 中配置（默认 `redis://localhost:6380/0`）；若与本地端口不一致请改源码或自行对齐部署。

前端可选：**`VITE_API_BASE_URL`**（默认 `http://localhost:8000`）。

---

## 安装与运行

### 后端

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# 若缺包，请按报错补装：fastapi、uvicorn、langchain、langchain-openai、langgraph、
# langgraph-checkpoint-redis、langchain-community、langchain-milvus、pymilvus、
# zhipuai、rank-bm25、jieba、sentence-transformers 等（以实际 import 为准）
python main.py
```

服务默认：**http://127.0.0.1:8000**，交互文档：**http://127.0.0.1:8000/docs**。

启动前请确保 **Redis Stack** 与 **Milvus**（若使用 RAG 写入）已就绪。

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
# 产物在 frontend/dist；部署时将 API 指到后端（构建前设置 VITE_API_BASE_URL）
```

---

## API 摘要

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/chat` | 非流式；请求体含 `message`、`model`、`session_id`、`user_id`（可选） |
| `POST` | `/api/chat/stream` | SSE 流式；仅 **`response` 节点** 的助手正文进入 SSE，避免 planner 的 JSON 混入 |
| `DELETE` | `/api/chat/history/{session_id}` | 清空 Runnable 链内存历史 + Agent 图检查点中该会话 |
| `GET` | `/api/chat/models` | 返回前端可选模型列表 |

---

## 架构说明（简）

1. **主路径**：`AgentOrchestrator` 编译 `build_graph()`，入口 `planner` 根据 JSON 计划路由到 `memory` / `rag` / `tool` 或直接 `response`，再经 `save_memory` 结束。
2. **流式**：`stream_mode="messages"`，API 层通过 `langgraph_node == "response"` 过滤，只把最终回复推给前端。
3. **检查点**：`get_short_term_checkpointer()` 使用 **RedisSaver**，会话与进程解耦；清空历史需同时调用链路与图上的清理逻辑（见 `chat.py`）。
4. **RAG**：文档在 `backend/data/`，首次检索会 `ensure_rag()` 加载并写入 Milvus（除非 `MILVUS_SKIP_INGEST`）；混合检索与重排在 `rag/` 下。
5. **`create_assistant`**：仍保留单 Agent + 工具调用实现，便于对比或切换；当前 HTTP 接口默认走 **图编排**。

---

## RAG 与数据

- 将 `.txt` 等放入 **`backend/data/`**（具体扫描逻辑见 `rag/loader.py`）。
- 首次构建或全量重建可能较慢（嵌入、Milvus 写入、可选重排模型下载）。
- 根目录若存在 **`test_rag.py`** 等脚本，可用于独立验证检索链路（与主服务解耦）。

---

## 注意事项

1. 对话依赖 **DeepSeek API Key**；RAG 嵌入、天气分别依赖 **智谱**、**高德** Key，未配置时对应能力会失败或不可用。
2. **Redis** 须为 Stack 能力集；纯 `redis` 官方镜像无搜索模块时，检查点 `setup()` 可能失败。
3. CORS 当前为宽松配置（`allow_origins=["*"]`），生产环境请按域名收紧。

## License

MIT
