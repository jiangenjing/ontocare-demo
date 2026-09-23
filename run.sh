#!/bin/bash
# OntoCare 一键启动脚本
# 用法：bash run.sh   然后浏览器打开 http://127.0.0.1:5000
cd "$(dirname "$0")"

# 模型密钥、地址和模型名由运行环境提供；不要写进脚本。

echo "[OntoCare] 安装依赖..."
pip install -q -r requirements.txt 2>/dev/null || pip3 install -q -r requirements.txt

echo "[OntoCare] 启动服务 → http://127.0.0.1:5000"
python3 app.py
