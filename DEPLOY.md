# Railway 部署说明

当前公开体验入口为 <https://web-production-94c004.up.railway.app/>，部署在 Railway 项目 `innovative-bravery`。仓库主分支推送不一定意味着服务已经完成部署；以 Railway 的部署记录、对应 commit、启动日志和线上检查为准。

## 仓库服务配置

- Build：`pip install -r requirements.txt`
- Start：仓库 `Procfile` 中的 Gunicorn 命令
- Python：建议使用 3.11 或更新版本
- Worker：当前命令使用单 worker，因为案件状态在进程内存中
- 端口：使用 Railway 注入的 `PORT`

请先在 Railway 确认服务仍连接本仓库的 `main` 分支。不要仅凭这份文档假设项目 ID、部署变量、计费计划或运行中的 commit 没有变化。

## 部署或更新

1. 在本地检查 `git status`，确认没有 `.env`、密钥、个人订单/联系信息或无关上传文件。
2. 提交并推送希望部署的代码到已连接分支。
3. 在 Railway 项目 `innovative-bravery` 查看最新部署。确认它使用刚推送的 commit，等待构建与启动日志完成。
4. 检查 Railway 服务域名仍为 `web-production-94c004.up.railway.app`，打开主页并访问 `/health`。
5. 做一次最小对话检查，再验证产品歧义、订单查无、退款/换货拦截和结束语；确认模型失败时页面显示错误，不切换到固定故事。
6. 记录部署 commit、部署时间、测试结果与回滚版本。若服务未自动部署，检查服务绑定、分支和构建日志，不要通过公开文档粘贴密钥来排障。

## 私密环境变量

按当前后端代码与模型服务配置 Railway Variables。变量名以 `.env.example` 和 `app.py` 为准，可能包括：

- `LLM_BASE_URL`
- `LLM_MODEL`
- `LLM_API_KEY`
- `VISION_BASE_URL`
- `VISION_MODEL`
- `VISION_API_KEY`

只有当前启用的模型能力才需要配置相应变量。模型名和 API 地址应以账户控制台显示值为准；本仓库不写入 API Key。请在 Railway 控制台的私密变量面板录入，不要把真实值放进 README、聊天回复、GitHub Actions 输出或公开 issue。轮换密钥时先更新部署变量并验证，再撤销旧密钥。若密钥曾经提交到 GitHub，应立即轮换；删除当前文件不会清除 Git 历史。

没有设置模型变量时可以验证模拟数据和确定性规则，但这不是真实模型联调。线上模型请求失败、超时或格式异常时必须清晰报告错误，不能回放演示剧本来伪装成功。

## 发布后检查清单

- `/` 返回当前前端；页面版本和右侧状态区正常。
- `/health` 返回健康状态。
- `/api/chat` 可处理文本；request ID、模型状态、证据和规则输出与输入一致。
- 同一会话的连续追问保留产品/问题/已尝试步骤，结束语不会被误识别为新售后问题。
- 订单查无保持 `UNKNOWN`；身份未验证时不声称订单属于当前访客。
- 退款、换货和补发只生成待人工草稿；没有真实写入或“已经受理”的误导性表述。
- 记录线上部署所对应的 Git commit。代码已推送、Railway 已构建、线上行为已验证是三个不同状态，报告时要分开说明。

## 回滚

如果新版本影响演示，优先在 Railway 选择上一个已知正常的部署进行回滚，并记录故障表现与 commit。不要在没有备份或核对的情况下重写 Git 历史。修复后重复关键测试并核对 Railway 记录中的 commit。
