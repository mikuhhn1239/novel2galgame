@echo off
echo ========================================
echo   All Novel Can Be Galgame - 停止脚本
echo ========================================
echo.

echo 停止 API 服务器 (端口 3002)...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":3002.*LISTEN"') do taskkill /F /PID %%a >nul 2>&1

echo 停止前端服务器 (端口 5173)...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":5173.*LISTEN"') do taskkill /F /PID %%a >nul 2>&1

echo 停止 Ren'Py 进程...
taskkill /F /IM "renpy*" >nul 2>&1

echo.
echo 所有服务已停止。
