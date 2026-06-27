@echo off
echo ========================================
echo   All Novel Can Be Galgame - 启动脚本
echo ========================================
echo.

:: 启动 API 服务器
echo [1/2] 启动 API 服务器 (端口 3002)...
cd /d "%~dp0apps\api"
start "API Server" cmd /c "npx tsx src/index.ts"
timeout /t 4 /nobreak >nul

:: 启动前端开发服务器
echo [2/2] 启动前端开发服务器 (端口 5173)...
cd /d "%~dp0apps\workbench"
start "Workbench" cmd /c "npx vite --port 5173"
timeout /t 3 /nobreak >nul

echo.
echo ========================================
echo   启动完成！
echo   API:  http://localhost:3002
echo   工作台: http://localhost:5173
echo ========================================
echo.
echo 按任意键关闭此窗口...
pause >nul
