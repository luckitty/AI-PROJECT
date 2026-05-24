# travel-guide-agent（旅游攻略助手）

基于 **FastAPI + Vue 3 + Vite** 的对话应用：前端为类 DeepSeek 的聊天界面；后端对话主路径为 **LangGraph** 编排——入口经人机闸门与 **`conversation_compress`** 后进入 **`planner`**，按计划在 **`memory` / `rag` / `amap_mcp` / `tool`** 与 **`response`** 之间流转，最后 **`save_memory`** 收口。底层对话模型走 **DeepSeek 兼容 OpenAI 接口**；RAG 嵌入可走 **智谱**；短期会话检查点使用 **Redis Stack（RediSearch + RedisJSON）**；向量库默认 **Milvus**；地图能力统一经 **高德 MCP（streamable HTTP）**。

---

## 功能概览

| 能力 | 说明 |
|------|------|
| 对话 | 非流式 `POST /api/chat`；流式 `POST /api/chat/stream`（SSE：`data: {"content":...}` 或控制字段，结束 `data: [DONE]`） |
| 请求体 | `message`、`model`、`session_id`、`user_id`（可选）；`resume`（恢复人机断点）；`human_breakpoints`（是否在 SSE 中透出 interrupt，默认服务端自动 resume） |
| 编排 | `conversation_compress` → `planner` → 按需 `rag` / `memory` / `amap_mcp` / `tool` → `response` → `save_memory`；各阶段可插入人机闸门（`interrupt()`） |
| 旅游攻略 | `response` 节点在攻略场景下调用 **`search_travel`**，结合离线旅游缓存与行程拼装（见 `tools/search_travel_tool.py`、`rag/travel_cache_retriever.py`） |
| 本地旅游 RAG | `planner` 仅在用户语句命中 **北京 / 广州 / 杭州 / 上海 / 西安 / 长沙** 之一时置 `need_rag`，其余攻略由终稿路径处理（见 `agents/planner_node.py`） |
| 高德 MCP | `amap_mcp` 节点：天气、POI 搜/详情、周边搜、地理编码、公交路径、专属地图等（见 `mcp_servers/amap_mcp_registry.py`） |
| 工具 | `tool` 节点：`web_search` 联网检索；注册表见 `tools/tool_registry.py` |
| 短期记忆 | LangGraph **RedisSaver** 检查点，按 `thread_id`（与前端 `session_id` 对齐）持久化 |
| RAG | `rag` 节点：`data` 目录文档 → 嵌入（智谱）→ **Milvus**；BM25 + 向量混合检索，可选 CrossEncoder 精排（`USE_RERANKER`） |
| 会话控制 | `GET /api/chat/stop?session_id=` 配合前端 Abort；同用户新会话可打断旧会话（`interruptController`） |
| 前端 | Markdown（marked + DOMPurify）、流式输出、侧边栏会话、模型与非流式/流式切换；`VITE_API_BASE_URL` 指向后端 |

---

## 仓库结构（要点）

```
travel-guide-agent/
├── backend/
│   ├── main.py                    # FastAPI 入口（默认 0.0.0.0:8000）
│   ├── api/chat.py                # /api/chat、/stream、/stop、/history、/models
│   ├── graph/
│   │   ├── builder.py             # StateGraph：人机闸门 + compress + planner → … → save_memory
│   │   ├── orchestrator.py        # AgentOrchestrator：invoke / stream（updates + custom）
│   │   ├── router.py              # planner 出边路由（rag / memory / amap_mcp / tool / response）
│   │   ├── state.py               # AgentState
│   │   └── interrupt.py           # 人机闸门节点
│   ├── agents/                    # planner、memory、rag、amap_mcp、tool、response、save_memory 等
│   ├── core/config.py             # dotenv 加载 API Key / 模型名等
│   ├── core/llm.py                # ChatOpenAI（DeepSeek）
│   ├── memory/
│   │   ├── short_memory.py        # LangGraph RedisSaver 检查点
│   │   ├── redis_config.py        # REDIS_URL、TTL
│   │   └── long_memory*.py        # 长期记忆（按需）
│   ├── rag/                       # 加载、混合检索、Milvus、旅游离线缓存等
│   ├── tools/                     # 联网搜索、ToolRegistry、search_travel、行程拼装等
│   ├── mcp_servers/               # 高德 streamable HTTP MCP 客户端与 LangChain 封装
│   ├── tool_registry/mcp/         # 地图类 @tool 封装（供节点与 builder 调用）
│   ├── data/                      # RAG 文本、城市缓存、POI 等数据文件
│   ├── interruptController/       # 会话 stop / 用户活跃会话绑定
│   └── requirements.txt           # 部分依赖声明（安装时按报错补全主栈）
├── frontend/
│   ├── src/components/ChatInterface.vue
│   ├── src/api/chat.js            # axios + fetch(SSE)，VITE_API_BASE_URL
│   └── vite.config.js             # 开发端口 3000
├── docker-compose.yml             # Milvus standalone（etcd + minio + milvus）
├── start.sh / start.bat           # 根目录检查 .env 后拉起 backend + frontend
└── README.md
```

后端一般在 **`backend/`** 下执行 `python main.py`（或在外层用 `start.sh`）。Milvus 路径与集合逻辑见 `backend/rag/original/vectorstores/milvus_client.py`。

---

## 环境要求

- **Node.js** 建议 **18+**（Vite 5）
- **Python** 建议 **3.11+**
- **Redis Stack**（含 **RediSearch、RedisJSON**），供 LangGraph `RedisSaver`；连接默认见下方 `redis_config.py`
- **Milvus**：RAG / 向量检索默认连接本机或配置的实例；仓库提供 **`docker-compose.yml`** 一键起 Milvus 依赖组件
- 网络：需能访问 **DeepSeek API**；嵌入使用 **智谱** 时需可达智谱接口；地图与天气依赖 **高德 MCP**（`AMAP_KEY` 等，见 `mcp_servers/config.py`）

---

## 环境变量

在项目根目录或运行目录旁放置 **`.env`**，由 `python-dotenv` 加载（`backend/core/config.py`）。

| 变量 | 说明 |
|------|------|
| `DEEPSEEK_API_KEY` | 对话模型（必填） |
| `DEEPSEEK_BASE_URL` | 可选，默认 `https://api.deepseek.com` |
| `MODEL_NAME` | 可选，默认 `deepseek-chat` |

| `ZHIPU_API_KEY` | RAG 向量嵌入（智谱 `embedding-3`）需要 |
| `AMAP_KEY` | 高德 MCP / 地图工具需要 |
| `USE_RERANKER` | 设为 `0` / `false` 等可关闭 CrossEncoder 精排，加快启动、省资源 |
| `LANGSMITH_API_KEY` | 可选：填写即开启 LangSmith 追踪（内部会设 `LANGCHAIN_TRACING_V2=true`）；项目名见 `backend/core/config.py` 中 `LANGCHAIN_PROJECT` |

**Milvus**（`backend/rag/original/vectorstores/milvus_client.py`）：

| 变量 | 说明 |
|------|------|
| `MILVUS_HOST` | 默认 `127.0.0.1` |
| `MILVUS_PORT` | 默认 `19530` |
| `MILVUS_TIMEOUT` | gRPC 超时（秒），默认 `60` |
| `MILVUS_COLLECTION` | 集合名，默认 `rag_collection` |
| `MILVUS_SKIP_INGEST` | `1`/`true` 时跳过写入，仅连接已有集合 |
| `MILVUS_DROP_OLD` | `1`/`true` 时 `from_documents` 侧按 LangChain Milvus 语义处理旧数据 |

**Redis**：在 `backend/memory/redis_config.py` 中配置 **`REDIS_URL`**（默认 `redis://localhost:6380/0`，示例端口 **6380** 以避免与本机其他 Redis 冲突）。

前端构建/开发：**`VITE_API_BASE_URL`**，默认 `http://localhost:8000`（见 `frontend/src/api/chat.js`）。

---

## 安装与运行

### 依赖服务建议顺序

1. 启动 **Redis Stack**（监听 URL 与 `redis_config.py` 一致）
2. 启动 **Milvus**（可用仓库根目录 `docker-compose up -d`）
3. 配置 **`.env`**（至少 `DEEPSEEK_API_KEY`，按需 `ZHIPU_API_KEY`、`AMAP_KEY`）

### 后端

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# 若缺包，按报错补装：fastapi、uvicorn、langchain、langchain-openai、langgraph、
# langgraph-checkpoint-redis、langchain-community、langchain-milvus、pymilvus、
# zhipuai、python-dotenv 等（以实际 import 为准）
python main.py
```

- 服务默认：**http://127.0.0.1:8000**
- OpenAPI：**http://127.0.0.1:8000/docs**

### 前端

```bash
cd frontend
npm install
npm run dev
```

开发服务器默认：**http://localhost:3000**（`vite.config.js`）。

### 一键脚本（根目录）

```bash
./start.sh
```

需先在根目录准备 **`.env`**（脚本会检查）。Windows 可使用 **`start.bat`**。

### 生产构建前端

```bash
cd frontend
npm run build
# 产物在 frontend/dist；构建前设置 VITE_API_BASE_URL 指向生产后端
```

---

## API 摘要

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/chat` | 非流式；支持 `resume`、`human_breakpoints`；若挂起人机断点则 `status=interrupted` 且返回 `interrupts` |
| `POST` | `/api/chat/stream` | SSE；首包可为 `{"typing": true}`；正文为 `{"content": "..."}`；`human_breakpoints=true` 时可下发 `{"type":"interrupt",...}` |
| `GET` | `/api/chat/stop` | Query：`session_id` — 通知后端停止该会话生成 |
| `DELETE` | `/api/chat/history/{session_id}` | 清空 Runnable 链历史与 Agent 检查点 |
| `GET` | `/api/chat/models` | 前端可选模型列表 |

---

## 架构说明（简）

1. **图编译**：`build_graph()` 在 `compile` 时挂载 **`get_short_term_checkpointer()`**（Redis）。
2. **流式**：`stream_mode=["updates", "custom"]`；`response`（等）通过 **`get_stream_writer`** 写入 `custom` 增量；API 层将增量封装为 SSE JSON。
3. **人机协作**：图中多处 **`human_checkpoint_*`**；非流式可在服务端循环 `Command(resume=True)` 直到完成；流式默认自动 resume，次数上限见 **`MAX_HITL_RESUME_STEPS`**（`graph/orchestrator.py`）。
4. **RAG 与攻略**：通用 RAG 节点与 **`travel_cache_retriever` / 离线缓存** 协同；攻略素材组装与行程指令见 **`tools/search_travel_tool.py`**、**`tools/travel_itinerary_builder.py`**。离线构建脚本位于 **`backend/tools/offlineTool/`**、**`backend/rag/offlineCache/`**。
5. **地图能力**：天气、POI、算路等由 **`amap_mcp`** 节点经 MCP 调用，不再走独立 `weather_tool` / `amap_route_tool`。
6. **`create_assistant`**（`agents/assistant.py`）：保留单 Agent + 工具实现；当前 HTTP 主路径以 **图编排 `AgentOrchestrator`** 为准（见 `api/chat.py`）。

---

## 数据与离线构建

- 文本与缓存数据多在 **`backend/data/`**（含城市 JSON、POI 缓存、Milvus 签名文件等）。
- 新增或更新语料后，需按项目内脚本重建向量库或旅游缓存（具体入口以 `offlineTool`、`offlineCache` 下脚本为准）。
- `requirements.txt` 中列有 **rapidocr-onnxruntime**、**playwright** 等，用于采集 / OCR 等离线链路，运行时按需安装。

---

## 注意事项

1. **DeepSeek / 智谱 / 高德** Key 未配置时，对应能力会失败或降级，需在日志中排查。
2. **Redis** 必须为 **Stack** 能力集；纯官方 `redis` 镜像无搜索模块时，检查点初始化可能失败。
3. **CORS** 当前为宽松配置（`allow_origins=["*"]`），生产环境请按域名收紧。
4. **算路与 POI**：`planner` 将天气、两地通勤、POI/周边搜等路由到 **`need_amap_mcp`**；复杂地名指代依赖多轮对话节选（见 `amap_mcp_registry` 与 `planner_node` 场景说明）。

## License

MIT
