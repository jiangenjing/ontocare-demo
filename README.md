# OntoCare — Anker 售后 Agent 最小可用版（MVP）

> 对应方案里"初始版本开发：导入业务文档后半小时生成可运行 demo"。
> 现在的状态：**业务文档（JSON）+ Decision Ontology 规则引擎已经跑通并验证**；
> LLM 尚未接入（当前用模板回复），留好接口，填 key 即可切换真模型。

## 一句话
大模型只负责"听懂人话"；**质保、经销商、动作资格这些确定性判断，全部由 `app.py` 里的规则函数确定性计算**——LLM 只能在 `allowed_actions` 里选，想越权也做不到。

## 现在已经具备
- **真 LLM**：DeepSeek 生成话术，但只能在后端规则算出的 allowed_actions 里选；
- **双视图前端**：左用户对话（含图片上传 📷），右实时 Decision Trace（情绪/证据/产品/订单/质保/经销商/允许·禁止动作/决策理由）；
- **本体约束检索**：先定产品，再在该产品知识范围内做 TF-IDF 检索，避免跨产品污染；无答案时诚实转人工、不编造；
- **20 个 Golden Case 回归**：`python eval.py`，规则层 100% 通过。

## 怎么跑
```bash
cd ankercare-mvp
pip install flask
export LLM_API_KEY="apiany 的 key"          # 文本主模型 gpt-5.6
export VISION_API_KEY="百炼工作空间 key"     # 视觉 qwen3-vl-flash
python3 app.py                          # 打开 http://127.0.0.1:5000
python3 eval.py                         # 跑回归评测（规则层 20/20）
```
- 文本：gpt-5.6 真生成话术，但只能在后端 `allowed_actions` 里选；
- 视觉：上传故障图 → **qwen3-vl-flash 真读**出 model/sku/error_code → 作为 PROBABLE 证据进 Decision Trace，并要求"以订单复核为准"。
左侧对话，右侧实时显示 Decision Trace（情绪/产品消歧/质保/经销商/允许动作/被禁动作）。

## 已经验证通过的 5 个 Golden Case（规则层）
| Case | 输入 | 系统行为 |
|---|---|---|
| ① S1 Pro 消歧 | "My S1 Pro not sucking" + ORD-2001 | 按订单 SKU 识别成**吸奶器**（不是扫地机），在保+授权，禁止直接退款 |
| ② 订单查无 | ORD-9999 要换货 | warranty=**UNKNOWN**，禁止判"过保"、禁止换货/退款，转而问国家和卖家 |
| ③ 在保+要退款 | ORD-2002 "refund now!!!" | 识别成**愤怒情绪**，订单定位扫地机；`direct_refund` 被规则**直接拦住** |
| ④ 过保 | ORD-2003 | 只能付费维修/转人工，免费换货、退款均不可用 |
| ⑤ 非授权经销商 | ORD-2004（在保但非授权） | 引导联系购买渠道，不进入官方保修 |

## 文件结构
```
ankercare-mvp/
├── app.py              # 后端：规则引擎(Decision Ontology) + LLM 适配层(待接)
├── data/
│   ├── products.json   # 产品+别名（含 S1 Pro=吸奶器&扫地机 同名歧义）
│   ├── orders.json     # 订单：在保/过保/查无/非授权 四种样本
│   ├── dealers.json    # 经销商：Country 精确 + Seller 模糊匹配
│   ├── kb.json         # 排障知识库（按 产品+故障 组织）
│   └── policies.json   # 质保/换货/退款规则、风险等级、情绪策略
└── static/index.html   # 双视图前端：左对话 + 右 Decision Trace
```

## 下一步：接真 LLM（约 30 分钟）
在 `app.py` 的 `build_reply()` 处，把模板替换成一次 LLM 调用：
- 系统提示词里喂入：用户问题 + 检索到的 kb + **后端算出的 allowed_actions**；
- 要求模型"只能从 allowed_actions 里选，并引用证据"；
- 推荐用赛题给的**阿里云百炼**（或任意 OpenAI 兼容接口）；
- 多模态/图片：二期再接 Vision 模型，现在文本跑通即可。

接完 LLM，再做进阶迭代：真向量检索（FAISS/Chroma）→ 接图片识别 → 接 Mock 换货 Tool 生成取件单。
