# DeepSeek Chat - AI 助手

一个基于 Vue3 和 FastAPI 的 AI 聊天应用，界面类似于 DeepSeek。

## 功能特性

- 🎨 现代化 UI 设计
- 💬 实时聊天交互
- 📝 Markdown 支持
- 🎯 多模型切换
- 💾 聊天历史保存
- 📱 响应式设计
- 🔄 流式响应

## 技术栈

### 前端
- Vue 3 (Composition API)
- Vite
- Axios
- Marked (Markdown 渲染)
- DOMPurify (XSS 防护)

### 后端
- FastAPI
- LangChain
- DeepSeek API

## 快速开始

### 1. 安装依赖

#### 前端
```bash
cd website
npm install
```

#### 后端
```bash
pip install fastapi uvicorn langchain-openai python-dotenv
```

### 2. 配置环境变量

在项目根目录创建 `.env` 文件：

```env
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

### 3. 启动服务

#### 启动后端
```bash
cd website
python server.py
```

后端将运行在 `http://localhost:8000`

#### 启动前端
```bash
npm run dev
```

前端将运行在 `http://localhost:3000`

### 4. 使用应用

打开浏览器访问 `http://localhost:3000` 即可使用。

## 项目结构

```
website/
├── src/
│   ├── api/
│   │   └── chat.js          # API 接口
│   ├── components/
│   │   └── ChatInterface.vue # 聊天界面组件
│   ├── styles/
│   │   └── main.css         # 全局样式
│   ├── App.vue              # 根组件
│   └── main.js              # 入口文件
├── index.html               # HTML 模板
├── vite.config.js          # Vite 配置
├── package.json            # 依赖配置
└── server.py               # FastAPI 后端服务器
```

## API 接口

### POST /api/chat
发送聊天消息

**请求：**
```json
{
  "message": "你好",
  "model": "deepseek-chat"
}
```

**响应：**
```json
{
  "reply": "你好！有什么我可以帮你的吗？"
}
```

### POST /api/chat/stream
流式聊天消息

**请求：**
```json
{
  "message": "写一个 Python 函数",
  "model": "deepseek-chat"
}
```

**响应：**
```
data: def
data: hello_world():
data: print("Hello, World!")
data: [DONE]
```

### GET /api/chat/history/{session_id}
获取聊天历史

### DELETE /api/chat/history/{session_id}
清空聊天历史

### GET /api/models
获取可用模型列表

## 自定义配置

### 修改 API 地址
创建 `.env` 文件：
```env
VITE_API_BASE_URL=http://your-api-server.com
```

### 修改主题颜色
在 `src/styles/main.css` 中修改颜色变量。

## 开发

### 添加新功能
1. 在 `src/components/` 下创建新组件
2. 在 `src/api/` 下添加新的 API 接口
3. 更新 `ChatInterface.vue` 集成新功能

### 构建生产版本
```bash
npm run build
```

构建产物在 `dist/` 目录下。

## 注意事项

1. 确保已配置正确的 DeepSeek API Key
2. 后端服务需要先启动
3. 聊天历史存储在浏览器 LocalStorage 中
4. 生产环境建议使用真实的数据库存储聊天历史

## License

MIT
