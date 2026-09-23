"""
Anker Ontology-Driven Service Agent — MVP 后端
================================================
原则：大模型只做模糊理解；质保/经销商/动作资格等确定性判断全部由本文件的
规则函数（Decision Ontology）确定性计算。LLM 无权绕过 allowed_actions。
无 LLM key 时走 MOCK 模式，用模板生成回复，规则引擎照常工作。
运行：pip install flask  &&  python app.py   打开 http://127.0.0.1:5000
"""
import json, os, re, math
from datetime import date
import urllib.request
from flask import Flask, request, jsonify, send_from_directory

# ---------- LLM 适配层（OpenAI 兼容；key 从环境变量读，不写死） ----------
LLM_BASE = os.getenv("LLM_BASE_URL", "https://apiany.org/v1")
LLM_KEY  = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5.6")

def call_llm(system: str, user: str, timeout: int = 15) -> str | None:
    """调 OpenAI 兼容接口；失败/无 key 返回 None（上层回退模板）。"""
    if not LLM_KEY:
        return None
    payload = json.dumps({
        "model": LLM_MODEL,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": 0.3,
        "max_tokens": 400,
    }).encode()
    req = urllib.request.Request(LLM_BASE.rstrip("/") + "/chat/completions",
        data=payload, headers={"Authorization": f"Bearer {LLM_KEY}", "Content-Type": "application/json",
                               "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("[LLM call failed, fallback to template]", e)
        return None

# ---------- 视觉模型(VLM)：读故障图，抽结构化证据 ----------
VLM_BASE  = os.getenv("VISION_BASE_URL", "https://llm-b0dh1t1jaywqp27j.cn-beijing.maas.aliyuncs.com/compatible-mode/v1")
VLM_KEY   = os.getenv("VISION_API_KEY", "")
VLM_MODEL = os.getenv("VISION_MODEL", "qwen3-vl-flash")  # 视觉理解，便宜快

def call_vlm(image_path: str) -> dict | None:
    """把故障图喂给多模态模型，抽 {product_model, sku, error_code, visible_issue, confidence}。
    OpenAI 兼容格式；无 VLM 配置则返回 None（走文本降级）。"""
    if not (VLM_BASE and VLM_KEY and VLM_MODEL) or not os.path.exists(image_path):
        return None
    import base64, mimetypes
    mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    sys_p = ("You inspect a customer's product-fault photo. Return ONLY JSON: "
             "{product_model, sku, error_code, visible_issue, confidence}. "
             "Extract only VISIBLE facts. If not visible, leave empty string. Do not guess.")
    payload = json.dumps({
        "model": VLM_MODEL,
        "messages": [{"role": "system", "content": sys_p},
                     {"role": "user", "content": [
                         {"type": "text", "text": "What product/model/error do you see? JSON only."},
                         {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}]}],
        "temperature": 0.1, "max_tokens": 200,
    }).encode()
    req = urllib.request.Request(VLM_BASE.rstrip("/") + "/chat/completions",
        data=payload, headers={"Authorization": f"Bearer {VLM_KEY}", "Content-Type": "application/json",
                               "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"})
    try:
        with urllib.request.urlopen(req, timeout=40) as r:
            txt = json.loads(r.read())["choices"][0]["message"]["content"]
        # 鲁棒提取：模型可能用 ```json ... ``` 包裹
        s, e = txt.find("{"), txt.rfind("}")
        return json.loads(txt[s:e+1]) if s >= 0 and e > s else None
    except Exception as e:
        print("[VLM failed, fallback to text-only]", e)
        return None

BASE = os.path.dirname(os.path.abspath(__file__))
def load(name):
    with open(os.path.join(BASE, "data", name), encoding="utf-8") as f:
        return json.load(f)

PRODUCTS   = load("products.json")
ORDERS     = load("orders.json")
DEALERS    = load("dealers.json")
KB_LIST    = load("kb.json")
POLICY     = load("policies.json")

# ---------- 本体约束知识库检索（Ontology-Grounded RAG 最小版） ----------
# 只在【已确认产品】的知识范围内做 TF-IDF 检索，避免跨产品/同名产品污染。
class KBIndex:
    def __init__(self, kb_list):
        self.docs, df = [], {}
        for kb in kb_list:
            text = (kb["fault_name"] + " " + " ".join(kb["symptoms"]) + " " + kb.get("cause", "")).lower()
            toks = re.findall(r"[a-z]+", text)
            self.docs.append((kb, toks))
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        self.N, self.df = len(self.docs), df
    def _idf(self, t):
        return math.log((1 + self.N) / (1 + self.df.get(t, 0))) + 1
    def search(self, product_id, query, topk=1):
        q = re.findall(r"[a-z]+", query.lower())
        if not q:
            return []
        scored = []
        for kb, toks in self.docs:
            if kb["product_id"] != product_id:   # 本体约束：只搜当前产品
                continue
            tf = {}
            for x in toks:
                tf[x] = tf.get(x, 0) + 1
            dot = sum(tf.get(qt, 0) * self._idf(qt) for qt in q)
            norm = math.sqrt(sum((v * self._idf(k)) ** 2 for k, v in tf.items())) or 1
            qnorm = math.sqrt(sum(self._idf(qt) ** 2 for qt in q)) or 1
            scored.append((dot / (norm * qnorm), kb))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:topk]

KB = KBIndex(KB_LIST)

app = Flask(__name__, static_folder="static", static_url_path="")

# 内存会话：MVP 用 dict，生产换 DynamoDB / Redis
SESSIONS = {}

# ---------- 1. 感知层：情绪 / 实体 / 意图（mock；接 LLM 后替换这部分） ----------
def llm_understand_intent(text: str, current_product: str | None) -> dict:
    """LLM 前置理解：判断用户这句话是在延续旧问题，还是在说一个新问题。
    返回 {is_new_topic: bool, mentioned_product: str|None, is_irrelevant: bool}"""
    sys = """You are an intent understanding module for Anker after-sales support.
Given the user's message and the current product in the conversation, decide:
1. Is the user starting a NEW topic (different product, completely unrelated question, or greeting)?
2. If they mentioned a product, what is it? (e.g. breast pump, robot vacuum, power bank, earbuds)
3. Is this message completely unrelated to Anker electronics (e.g. asking about a paper box, shipping, etc.)?

Reply in JSON only:
{"is_new_topic": true/false, "mentioned_product": "product name or null", "is_irrelevant": true/false}"""
    user = f"Current product in conversation: {current_product or 'none'}\nUser message: {text}"
    out = call_llm(sys, user, timeout=8)
    if not out:
        # LLM 不可用：保守返回，不重置
        return {"is_new_topic": False, "mentioned_product": None, "is_irrelevant": False}
    try:
        # 提取 JSON
        m = re.search(r'\{[^}]*\}', out, re.DOTALL)
        if m:
            return json.loads(m.group(0))
    except:
        pass
    return {"is_new_topic": False, "mentioned_product": None, "is_irrelevant": False}

def detect_emotion(text: str) -> str:
    t = text.lower()
    if re.search(r"complain|ridiculous|unacceptable|lawyer|better business", t):
        return "COMPLAINT_RISK"
    if re.search(r"!!!|angry|furious|terrible|awful|scam|refund now", t):
        return "ANGRY"
    if re.search(r"urgent|party|tomorrow|asap|in a hurry", t):
        return "ANXIOUS"
    return "NORMAL"

def detect_scope(text: str) -> str:
    """服务边界：超出自助范围的意图直接转人工，不进入业务流程。"""
    t = text.lower()
    if re.search(r"\bpress\b|journalist|media inquiry|corporate|bulk order|wholesale|resell", t):
        return "OUT_OF_SCOPE_BUSINESS"
    if re.search(r"speak to (a )?(human|manager|real person|agent)|talk to (a )?(manager|human)|supervisor|real person|agent please", t):
        return "OUT_OF_SCOPE_HUMAN"
    return "IN_SCOPE"

PROMOTER_WORDS = ["pump", "flange", "breast", "duckbill", "membrane", "nipple"]
ROBOT_WORDS    = ["vacuum", "robot", "brush", "dock", "docking", "charging", "charge", "扫地", "吸尘", "滚刷"]

def resolve_product(text: str, order_sku: str | None) -> dict:
    """本体消歧：同名 S1 Pro → 吸奶器 or 扫地机。订单 SKU 是最强证据，直接定位。"""
    # 订单已明确 SKU：无需再从文本猜
    if order_sku:
        for p in PRODUCTS:
            if p["sku"] == order_sku:
                return {"product": p, "evidence": f"order_sku={order_sku}", "state": "VERIFIED"}
    t = text.lower()
    candidates = [p for p in PRODUCTS if "s1 pro" in t and p["category"] in ("breast_pump", "robot_vacuum")]
    if not candidates:  # 不是同名歧义，按 alias 粗匹配
        for p in PRODUCTS:
            if any(a in t for a in p["aliases"]):
                return {"product": p, "evidence": "alias_match", "state": "PROBABLE"}
        # 检查用户是否提到了产品/购买相关词
        product_words = ["buy", "bought", "product", "item", "order", "买", "箱", "机器", "device", "gadget", "charger", "battery", "vacuum", "pump", "earbud", "speaker", "replacement", "refund", "换货", "退款"]
        if any(w in t for w in product_words):
            return {"product": None, "evidence": "unknown_product", "state": "UNKNOWN"}
        return {"product": None, "evidence": "no_product_mentioned", "state": "UNKNOWN"}

    # 订单 SKU 是最强证据
    if order_sku:
        for p in PRODUCTS:
            if p["sku"] == order_sku:
                return {"product": p, "evidence": f"order_sku={order_sku}", "state": "VERIFIED"}

    # 语义关键词消歧
    if any(w in t for w in PROMOTER_WORDS):
        p = next(x for x in PRODUCTS if x["category"] == "breast_pump")
        return {"product": p, "evidence": "semantic=breast_pump", "state": "PROBABLE"}
    if any(w in t for w in ROBOT_WORDS):
        p = next(x for x in PRODUCTS if x["category"] == "robot_vacuum")
        return {"product": p, "evidence": "semantic=robot_vacuum", "state": "PROBABLE"}
    return {"product": None, "evidence": "ambiguous_S1Pro", "state": "CONFLICTED"}

def match_fault(product_id: str, text: str):
    res = KB.search(product_id, text)
    if res and res[0][0] > 0.08:   # 超过阈值才算命中；低于阈值→知识库无答案→走诚实升级
        return res[0][1]
    return None

# ---------- 2. 数据流：订单 / 经销商（确定性查询） ----------
def lookup_order(order_id: str | None):
    if not order_id or order_id not in ORDERS:
        return {"status": "NOT_FOUND", "warranty": "UNKNOWN"}
    o = ORDERS[order_id]
    wd = date.fromisoformat(o["warranty_until"])
    in_warranty = wd >= date.today()
    within_30d = (date.today() - date.fromisoformat(o["order_date"])).days <= 30
    return {"status": "FOUND", "order": o,
            "warranty": "VALID" if in_warranty else "EXPIRED",
            "warranty_until": o["warranty_until"], "within_30d": within_30d}

def match_dealer(country: str | None, seller: str | None):
    """Country 精确 + Seller 模糊。"""
    if not country or not seller:
        return {"status": "INSUFFICIENT_INFO"}
    sl = seller.strip().lower().replace("-", " ").replace(".", "").replace("  ", " ")
    for d in DEALERS:
        if d["country"].lower() != country.strip().lower():
            continue
        for name in d["seller_names"]:
            nl = name.lower().replace("-", " ").replace(".", "").replace("  ", " ")
            if sl in nl or nl in sl:
                return {"status": "AUTHORIZED" if d["authorized"] else "NOT_AUTHORIZED",
                        "matched": name}
    return {"status": "NO_MATCH"}

# ---------- 3. Decision Ontology：状态 → 合法动作空间（核心） ----------
def allowed_actions(case: dict):
    """返回 (allowed[], forbidden[])。规则全部确定性，LLM 只能在 allowed 里选。"""
    allowed, forbidden, reasons = [], [], []
    prod = case.get("product")
    warranty = case.get("warranty", "UNKNOWN")
    dealer = case.get("dealer", "UNKNOWN")
    ts = case.get("troubleshooting", "NOT_STARTED")

    if not prod:
        allowed += ["ask_product", "search_knowledge"]
        forbidden += ["propose_replacement", "direct_refund"]
        reasons.append("产品未确认：不允许任何售后办理动作")
        return allowed, forbidden, reasons

    allowed += ["search_knowledge", "run_troubleshooting", "escalate_human"]

    # 订单查无：UNKNOWN ≠ EXPIRED，禁止判过保、禁止办理
    if warranty == "UNKNOWN":
        forbidden += ["propose_replacement", "direct_refund", "tell_expired"]
        allowed += ["ask_country", "ask_seller", "request_proof"]
        reasons.append("订单查无：warranty=UNKNOWN，禁止判定过保、拒绝或承诺换货")
        return allowed, forbidden, reasons

    if warranty == "EXPIRED":
        forbidden += ["propose_replacement", "direct_refund"]
        allowed += ["propose_paid_repair"]
        reasons.append("已过保：免费换货/退款均不可用")
        return list(dict.fromkeys(allowed)), forbidden, reasons

    # warranty == VALID
    if ts == "FAILED" and dealer == "AUTHORIZED":
        allowed += ["propose_repair", "propose_replacement"]
        reasons.append("排障失败 + 在保 + 授权经销商：进入换货建议（HIGH，需人工审批）")
    if dealer == "NOT_AUTHORIZED":
        forbidden += ["propose_replacement", "direct_refund"]
        allowed += ["guide_contact_seller"]
        reasons.append("非授权经销商：引导联系购买渠道，不进入官方保修")
    elif case.get("within_30d"):
        allowed += ["direct_refund"]
        reasons.append("下单30天内 + 授权：命中30天无理由，退款可走(MEDIUM，留痕)")
    else:
        forbidden += ["direct_refund"]
        reasons.append("直接退款(CRITICAL)默认禁止，除非人工裁决")
    allowed = list(dict.fromkeys(allowed)); forbidden = list(dict.fromkeys(forbidden))
    return allowed, forbidden, reasons

# ---------- 主流程 ----------
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

UP_DIR = os.path.join(BASE, "uploads")
os.makedirs(UP_DIR, exist_ok=True)
VISION_CACHE = {}  # filename -> vlm 识别结果

@app.route("/api/upload", methods=["POST"])
def upload():
    f = request.files.get("image")
    if not f:
        return jsonify({"ok": False, "error": "no file"}), 400
    fn = f.filename or "upload.jpg"
    save = os.path.join(UP_DIR, fn)
    f.save(save)
    vlm = call_vlm(save)  # 真视觉识别；未配置 VLM 时返回 None
    VISION_CACHE[fn] = vlm
    return jsonify({"ok": True, "filename": fn, "vlm": vlm,
                    "note": "VLM enabled" if vlm else "VLM not configured, text-only fallback"})

@app.route("/api/chat", methods=["POST"])
def chat():
    body = request.get_json(force=True)
    text = body.get("message", "")
    img = body.get("image")  # 形如 "uploaded:fault.jpg"
    img_name = img.split(":", 1)[1] if img else None
    vlm_ev = VISION_CACHE.get(img_name) if img_name else None
    sid = body.get("session_id", "demo")
    case = SESSIONS.get(sid, {"product": None, "evidence": [], "warranty": "UNKNOWN",
                              "dealer": "UNKNOWN", "troubleshooting": "NOT_STARTED"})

    # 从消息里抓订单号；本轮没给则沿用会话里已确认的（多轮记忆）
    m = re.search(r"ORD-\d{4}", text.upper())
    order_id = m.group(0) if m else body.get("order_id") or case.get("order_id")
    if order_id:
        case["order_id"] = order_id
    explicit_oid = bool((m and m.group(0)) or body.get("order_id"))

    emotion = detect_emotion(text)
    scope = detect_scope(text)
    if scope != "IN_SCOPE":
        return jsonify({"reply": "This looks outside what our automated support can handle — I'm connecting you to a human specialist now. No need to repeat your issue, I'll pass the context along.",
                        "cards": [{"title": "Escalated to human", "body": "Reason: out_of_scope. A specialist will pick this up shortly."}],
                        "trace": {"emotion": emotion, "scope": scope, "product": "—", "order": order_id or "未提供",
                                  "warranty": "—", "dealer": "—", "troubleshooting": "—",
                                  "allowed": ["escalate_human"], "forbidden": [],
                                  "reasons": ["服务边界命中：超出自助范围，直接转人工（不强行自动处理）"]}})

    order = lookup_order(order_id) if order_id else {"status": "NOT_FOUND", "warranty": "UNKNOWN"}
    if order["status"] == "FOUND":
        sku = order["order"]["sku"]
        case["warranty"] = order["warranty"]
        d = match_dealer(order["order"]["country"], order["order"]["seller"])
        case["dealer"] = d["status"]
        case["within_30d"] = order.get("within_30d", False)
    else:
        sku = None
        # 仅当本轮明确报了订单号却查无，才把质保标 UNKNOWN；未报则保留上轮已确认状态
        if explicit_oid:
            case["warranty"] = "UNKNOWN"

    pr = resolve_product(text, sku)
    
    # 保护：极短输入（<3字符）或无意义字符，不沿用旧上下文，重置 case
    clean_text = text.strip().lower()
    if len(clean_text) < 3 and not re.search(r'ORD-\d{4}', clean_text.upper()):
        case["product"] = None
        case["warranty"] = "UNKNOWN"
        case["dealer"] = "UNKNOWN"
        case["troubleshooting"] = "NOT_STARTED"
        return jsonify({
            "reply": "Hi! 👋 I'm here to help. Could you tell me more about the issue you're having? "
                     "You can describe the problem, upload a photo, or share your order number (ORD-XXXX).",
            "cards": [],
            "trace": {"emotion": emotion, "scope": scope, "product": "—", "order": order_id or "未提供",
                      "warranty": "—", "dealer": "—", "troubleshooting": "—",
                      "allowed": ["ask_product", "search_knowledge"], "forbidden": ["propose_replacement", "direct_refund"],
                      "reasons": ["输入过短，重置上下文，引导用户描述问题"]}
        })
    
    # LLM 前置理解：判断是不是新话题/无关问题
    intent = llm_understand_intent(text, case.get("product", {}).get("name") if case.get("product") else None)
    
    # 如果是完全无关的问题（比如纸箱子），重置 case 并友好说明
    if intent.get("is_irrelevant") and not intent.get("mentioned_product"):
        case["product"] = None
        case["warranty"] = "UNKNOWN"
        case["dealer"] = "UNKNOWN"
        case["troubleshooting"] = "NOT_STARTED"
        return jsonify({
            "reply": "I'm sorry, but it looks like this might not be about an Anker product — "
                     "we only handle support for Anker electronics (chargers, robot vacuums, earbuds, etc.). "
                     "Could you tell me which product you're having trouble with? 🤔",
            "cards": [],
            "trace": {"emotion": emotion, "scope": scope, "product": "—", "order": order_id or "未提供",
                      "warranty": "—", "dealer": "—", "troubleshooting": "—",
                      "allowed": ["ask_product", "search_knowledge"], "forbidden": ["propose_replacement", "direct_refund"],
                      "reasons": ["LLM 意图识别：用户问题与 Anker 电子产品无关，重置 case，引导正确产品"]}
        })
    
    # 如果是新话题，重置产品和排障状态
    if intent.get("is_new_topic") and intent.get("mentioned_product"):
        case["product"] = None
        case["troubleshooting"] = "NOT_STARTED"
    
    if pr["product"]:
        case["product"] = pr["product"]
        case["product_state"] = pr["state"]
        case["product_unknown"] = False
    elif pr["evidence"] == "no_product_mentioned":
        # 用户没提新产品词，沿用之前的产品（比如"还是不行"）
        pass
    else:
        # unknown_product 或 ambiguous_S1Pro：产品不确定
        case["product"] = None
        case["product_state"] = pr["state"]
        # 只有明确不认识的产品才标记为 unknown
        case["product_unknown"] = (pr["evidence"] == "unknown_product")

    fault = match_fault(case["product"]["id"], text) if case["product"] else None
    if "replace" in text.lower() or "换货" in text:
        case["troubleshooting"] = "FAILED"  # demo：用户主动要求换货视作排障失败

    allowed, forbidden, reasons = allowed_actions(case)

    # 短问候语：先友好回应，再引导说问题
    greetings = ["hi", "hello", "hey", "hii", "hiya", "你好", "哈喽", "嗨", "早上好", "下午好", "晚上好", "在吗"]
    if text.strip().lower() in greetings or (len(text.strip()) <= 6 and any(g in text.lower() for g in greetings)):
        return jsonify({
            "reply": "Hi there! 👋 I'm OntoCare, Anker's after-sales assistant. What seems to be the problem? "
                     "You can tell me the product issue, upload a photo of the error, or share your order number (ORD-XXXX).",
            "cards": [],
            "trace": {"emotion": emotion, "scope": scope, "product": "—", "order": order_id or "未提供",
                      "warranty": "—", "dealer": "—", "troubleshooting": "—",
                      "allowed": ["ask_product", "search_knowledge"], "forbidden": ["propose_replacement", "direct_refund"],
                      "reasons": ["用户问候，先友好回应，引导描述问题"]}
        })

    # 1) 确定性后端先给出结构化卡片 + 模板兜底
    reply, cards = build_reply(case, emotion, order_id, order, fault, allowed, forbidden)
    # 2) 再让 LLM 在【合法动作】内把话术写自然；失败则用模板
    system = ("You are OntoCare, Anker's after-sales agent for overseas customers. "
              "STRICT RULES: recommend ONLY actions in [ALLOWED]; NEVER promise anything in [FORBIDDEN]. "
              "Do not guess warranty or dealer status — if evidence says UNKNOWN, say so and ask. "
              "Be concise, warm, and match the user's emotion. Do NOT repeat troubleshooting steps already shown as cards. "
              "Always reply in English. "
              "If you don't know the answer, say so and offer a human. Reply in the user's language.")
    kb_hint = f"{fault['fault_name']}: " + " | ".join(fault["steps"]) if fault else "none yet"
    if vlm_ev:
        img_hint = (f"VLM read from photo: model={vlm_ev.get('product_model','')}, sku={vlm_ev.get('sku','')}, "
                    f"error_code={vlm_ev.get('error_code','')}, visible_issue={vlm_ev.get('visible_issue','')}. "
                    f"This is PROBABLE evidence; confirm with the order if possible.")
    else:
        img_hint = ("User attached a photo but no VLM result (text-only). Ask for model/error code if unclear."
                    if img_name else "No photo attached.")
    user_prompt = (f"User said: {text}\n{img_hint}\nEmotion: {emotion}\n"
                   f"Product: {case['product']['name'] if case['product'] else 'not yet identified'}\n"
                   f"Order: {order_id or 'none'} ({order['status']})\n"
                   f"Warranty: {case['warranty']} | Dealer: {case['dealer']} | Troubleshooting: {case['troubleshooting']}\n"
                   f"Fault KB: {kb_hint}\n"
                   f"ALLOWED actions: {allowed}\nFORBIDDEN actions: {forbidden}\nReasons: {reasons}\n"
                   f"Write the short customer-facing reply now.")
    llm_reply = call_llm(system, user_prompt)
    if llm_reply:
        reply = llm_reply  # 话术用 LLM；卡片和 trace 仍是确定性后端产物

    trace = {
        "emotion": emotion,
        "evidence": pr["evidence"],
        "image": (f"{img_name} · VLM:{vlm_ev.get('error_code','?')}/{vlm_ev.get('visible_issue','?')}"
                  if vlm_ev else (img_name or "无")),
        "product": case["product"]["name"] if case["product"] else "未确定",
        "order": order_id or "未提供",
        "warranty": case["warranty"],
        "dealer": case["dealer"],
        "troubleshooting": case["troubleshooting"],
        "allowed": allowed, "forbidden": forbidden, "reasons": reasons,
    }
    SESSIONS[sid] = case
    return jsonify({"reply": reply, "cards": cards, "trace": trace})

def tone(emotion: str) -> str:
    """情绪→话术策略：不是换个温柔说法，而是改变服务节奏。"""
    return {
        "COMPLAINT_RISK": "I completely understand your frustration, and I'm taking this seriously — ",
        "ANGRY": "I hear you, and I'm sorry this happened. Let's cut to the solution — ",
        "ANXIOUS": "No need to worry, we'll sort this out quickly. ",
    }.get(emotion, "")

def nss_card():
    return {"title": "How did we do?", "body": "Your case is being handled. Please rate this interaction after it's resolved (NSS)."}

def build_reply(case, emotion, order_id, order, fault, allowed, forbidden):
    cards = []
    if not case["product"]:
        # 检查是用户没提产品，还是提了不认识的，还是同名歧义
        product_state = case.get("product_state", "UNKNOWN")
        product_unknown = case.get("product_unknown", False)
        if product_unknown:
            return (tone(emotion) + "Sorry about that — we don't carry that product. "
                    "Anker sells chargers, power banks, earbuds, robot vacuums, breast pumps and smart home devices. "
                    "Could you tell me which Anker product you need help with?", cards)
        return (tone(emotion) + "I want to make sure I help with the right product — which one is it? "
                "For example the eufy breast pump S1 Pro or the robot vacuum S1 Pro?", cards)
    p = case["product"]
    if order["status"] == "NOT_FOUND" and case["warranty"] == "UNKNOWN":
        return (tone(emotion) + f"I couldn't find order {order_id or 'your order'} in the system. That doesn't mean you're out of warranty — "
                f"could you tell me the country and the store/website where you bought it?", cards)
    if case["warranty"] == "EXPIRED":
        return (tone(emotion) + f"I checked your {p['name']}: it is past the standard warranty. I can still help you arrange a paid repair "
                f"or connect you to a specialist.", cards)
    if case["dealer"] == "NOT_AUTHORIZED":
        return (tone(emotion) + f"Thanks — the seller on this order is not in our authorized dealer list. Please reach out to the store you "
                f"bought it from for warranty; I'll also loop in a human to double-check.", cards)
    if fault:
        # 未知错误码 / 知识库无答案：诚实升级，绝不编造排障步骤
        if fault.get("after_failed") == "honest_escalate_no_fabrication":
            cards.append({"title": "Unknown error code → human", "body":
                          f"{fault['fault_name']} is not in our troubleshooting knowledge base. "
                          f"I will NOT guess a fix — escalating to a specialist who knows this model."})
            return (tone(emotion) + f"Thanks for the details. This {fault['fault_name']} isn't something I have a verified fix for, "
                    f"so rather than guess, I'm connecting you to a human who can look at it properly.", cards)
        steps = "\n".join(f"{i+1}. {s}" for i, s in enumerate(fault["steps"]))
        cards.append({"title": f"Troubleshooting: {fault['fault_name']}", "body": steps})
        return (tone(emotion) + f"Sorry about that 🙏 Based on your {p['name']}, please try these steps:\n{steps}\n"
                f"Did this fix it? (If not, I'll check warranty and replacement options.)", cards)
    if "propose_replacement" in allowed:
        cards.append({"title": "Replacement proposed (HIGH RISK)", "body":
                      "Eligible: troubleshooting failed + in-warranty + authorized dealer.\n"
                      "→ Pending human approval; a replacement order will be created after sign-off."})
        cards.append(nss_card())
        return (tone(emotion) + "I've reviewed your case and a replacement looks eligible. Because this is a high-risk action, a human "
                "agent will approve it and issue the replacement order shortly.", cards)
    if "direct_refund" in allowed:
        cards.append(nss_card())
        return (tone(emotion) + "You're within our 30-day return window from an authorized seller, so I've initiated a refund. "
                "You'll receive a confirmation by email shortly.", cards)
    return (tone(emotion) + f"Thanks for contacting us about your {p['name']}. Tell me a bit more about the issue and I'll guide you step by step.", cards)

if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False, threaded=True)
