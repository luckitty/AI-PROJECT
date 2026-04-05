@echo off
echo 🚀 启动 AI Project
echo ========================================

REM 检查 .env 文件
if not exist .env (
    echo ⚠️  未找到 .env 文件
    echo 请创建 .env 文件并配置以下变量：
    echo   DEEPSEEK_API_KEY=your_api_key
    echo   DEEPSEEK_BASE_URL=https://api.deepseek.com
    pause
    exit /b 1
)

REM 启动后端
echo 📡 启动后端服务器...
start "Backend Server" cmd /k "cd backend && python main.py"

REM 等待后端启动
timeout /t 3 /nobreak > nul

REM 启动前端
echo 🎨 启动前端开发服务器...
start "Frontend Server" cmd /k "cd frontend && npm run dev"

echo.
echo ✅ 项目启动成功！
echo 📍 前端地址: http://localhost:3000
echo 📍 后端地址: http://localhost:8000
echo 📍 API 文档: http://localhost:8000/docs
echo.
echo 请保持此窗口打开，按任意键退出...
pause > nul
