# Render 部署指南（5 分钟）

## 前提
- GitHub 账号
- Render 账号（用 GitHub 登录，免费）

## 步骤

### 1. 推代码到 GitHub
1. 新建一个 repo（比如 `ontocare-demo`）
2. 把本目录所有文件推上去（不要推 __pycache__ 和 uploads 里的大文件）

### 2. 在 Render 创建 Web Service
1. 打开 https://dashboard.render.com/
2. New → Web Service → 选你的 GitHub repo
3. 配置：
   - **Name**: `ontocare`（会变成 https://ontocare.onrender.com）
   - **Region**: Singapore（离中国最近）
   - **Branch**: main
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: 自动从 Procfile 读，不用填
   - **Instance Type**: Free

### 3. 设环境变量
在 Render → Environment 里加：
- `LLM_API_KEY` = 你的 apiany key
- `VISION_API_KEY` = 你的百炼 key
- `LLM_BASE_URL` = `https://apiany.org/v1`
- `LLM_MODEL` = `gpt-5.6`

### 4. 部署
点 Deploy，等 2-3 分钟。完成后访问：
`https://ontocare.onrender.com`

## 注意
- 免费版 15 分钟无访问会休眠，首次访问等 30 秒唤醒
- 上传图片的 uploads/ 目录在 Render 是临时的，重启就清空（Demo 不影响）
