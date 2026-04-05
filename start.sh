#!/bin/bash

echo "🚀 启动 AI Project"
echo "================================"

# 检查 .env 文件
if [ ! -f .env ]; then
    echo "⚠️  未找到 .env 文件"
    echo "请创建 .env 文件并配置以下变量："
    echo "  DEEPSEEK_API_KEY=your_api_key"
    echo "  DEEPSEEK_BASE_URL=https://api.deepseek.com"
    exit 1
fi

# 启动后端
echo "📡 启动后端服务器..."
cd backend
python main.py &
BACKEND_PID=$!
cd ..

# 等待后端启动
sleep 3

# 启动前端
echo "🎨 启动前端开发服务器..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "✅ 项目启动成功！"
echo "📍 前端地址: http://localhost:3000"
echo "📍 后端地址: http://localhost:8000"
echo "📍 API 文档: http://localhost:8000/docs"
echo ""
echo "按 Ctrl+C 停止所有服务"

# 等待信号
wait $BACKEND_PID $FRONTEND_PID
