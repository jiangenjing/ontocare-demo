#!/bin/bash
# OntoCare 一键启动脚本
# 用法：bash run.sh   然后浏览器打开 http://127.0.0.1:5000
cd "$(dirname "$0")"

# 文本主模型（gpt-5.6 @ apiany）
export LLM_API_KEY="sk-20f0de8b48241b6ed0cbebbca660961495261c558a4799c71ad2c8a3eda5fc3f"
# 视觉模型（qwen3-vl @ 百炼）
export VISION_API_KEY="sk-ws-H.PLHDMXE.gBYH.MEUCIAqw6ggy2TgeVUvgE61E8qqodaYi2Hu1LrwxRKoNzJ7YAiEAg7Vty9jkc7lzuNWiyU4HpyTCLacg7JbrzYdxEIgJkUw"

echo "[OntoCare] 安装依赖..."
pip install -q -r requirements.txt 2>/dev/null || pip3 install -q -r requirements.txt

echo "[OntoCare] 启动服务 → http://127.0.0.1:5000"
python3 app.py
