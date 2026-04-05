# AI Project - DeepSeek Chat

一个基于 LangChain 和 Vue3 的 AI 聊天助手项目。

## 项目结构

```
ai-project/
├── backend/                    # 后端代码
│   ├── api/                    # API 路由
│   │   └── chat.py            # 聊天 API
│   ├── agents/                 # Agent 模块
│   │   └── assistant.py       # AI 助手 Agent
│   ├── chains/                 # Chain 模块
│   │   └── chat_chain.py      # 对话链
│   ├── tools/                  # 工具模块
│   │   └── __init__.py        # 工具定义
│   ├── core/                   # 核心配置
│   │   ├── config.py          # 配置管理
│   │   └── llm.py             # LLM 实例
│   └── main.py                 # FastAPI 主入口
├── frontend/                   # 前端代码
│   ├── src/
│   │   ├── api/               # API 接口
│   │   ├── components/        # Vue 组件
│   │   └── styles/            # 样式文件
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── data/                       # 数据文件
├── docs/                       # 文档
├── .env                        # 环境变量
└── README.md
```

## 功能特性

- **AI 对话** - 支持 DeepSeek 大模型对话
- **工具调用** - 支持天气查询、日程安排、股票查询等工具
- **历史记忆** - 支持对话历史记录
- **流式输出** - 支持打字机效果输出
- **现代化 UI** - 类似 DeepSeek 的聊天界面

## 快速开始

### 1. 安装依赖

#### 后端依赖
```bash
pip install fastapi uvicorn langchain langchain-openai python-dotenv requests
```

#### 前端依赖
```bash
cd frontend
npm install
```

### 2. 配置环境变量

创建 `.env` 文件：
```env
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
MODEL_NAME=deepseek-chat
```

### 3. 启动服务

#### 启动后端
```bash
cd backend
python main.py
```
后端将运行在 `http://localhost:8000`

#### 启动前端
```bash
cd frontend
npm run dev
```
前端将运行在 `http://localhost:3000`

### 4. 访问应用

打开浏览器访问 `http://localhost:3000` 即可使用。

## API 文档

启动后端后，访问 `http://localhost:8000/docs` 查看 API 文档。

### 主要接口

- `POST /api/chat` - 发送聊天消息
- `POST /api/chat/stream` - 流式聊天
- `DELETE /api/chat/history/{session_id}` - 清空对话历史
- `GET /api/chat/models` - 获取可用模型列表

## 模块说明

### Backend 模块

| 模块 | 说明 |
|------|------|
| `core/config.py` | 配置管理，加载环境变量 |
| `core/llm.py` | LLM 实例管理 |
| `tools/` | 工具定义（天气、日程、股票等） |
| `agents/assistant.py` | AI 助手 Agent，支持工具调用 |
| `chains/chat_chain.py` | 对话链，支持历史记忆 |
| `api/chat.py` | 聊天 API 路由 |

### Frontend 模块

| 模块 | 说明 |
|------|------|
| `src/components/ChatInterface.vue` | 聊天界面组件 |
| `src/api/chat.js` | API 接口封装 |

## 扩展功能

### 添加新工具

在 `backend/tools/__init__.py` 中添加新的工具函数：

```python
from langchain.tools import tool

@tool
def my_new_tool(param: str) -> str:
    """工具描述"""
    # 实现逻辑
    return "结果"

# 添加到工具列表
ALL_TOOLS.append(my_new_tool)
```

### 自定义 Agent

在 `backend/agents/assistant.py` 中修改系统提示词和工具列表。

## 开发

### 后端开发
```bash
cd backend
# 开发模式启动（支持热重载）
python main.py
```

### 前端开发
```bash
cd frontend
npm run dev
```

### 构建前端
```bash
cd frontend
npm run build
```

## 注意事项

1. 确保 `.env` 文件配置正确
2. DeepSeek API 需要 API Key
3. 后端服务需要先启动
4. 生产环境建议使用数据库存储对话历史

## License

MIT
