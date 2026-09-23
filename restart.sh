#!/bin/bash
# OntoCare Demo 一键重启脚本

echo "🔄 正在重启 OntoCare Demo..."

# 杀掉旧进程
pkill -f "app.py" 2>/dev/null
sleep 1

# 进入项目目录
cd ~/Downloads/ankercare-mvp

# 设置环境变量
export LLM_API_KEY="sk-20f0de8b48241b6ed0cbebbca660961495261c558a4799c71ad2c8a3da5fc3f"
export VISION_API_KEY="sk-ws-H.PLHDMXE.gBYH.MEUCIAqw6ggy2TgeVUvgE61E8qqodaYi2Hu1LrwxRKoNzJ7YAiEAg7Vty9jkc7lzuNWiyU4HpyTCLacg7JbrzYdxEIgJkUw"

# 启动
python3 app.py
