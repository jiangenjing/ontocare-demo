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
LLM_BASE = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_KEY  = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

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

# ---------- 1. 感知层：LLM 统一理解（产品/情绪/新话题/问候/无关） ----------
def llm_perceive(text: str, current_product_name: str | None) -> dict:
    """LLM 前置感知：用户每说一句话，先让 LLM 判断 5 件事，输出结构化 JSON。
    LLM 只做"感知"，不做"决策"——决策还是规则引擎做。
    返回 {
      is_greeting, is_new_topic, is_irrelevant,
      product_hint: breast_pump/robot_vacuum/power_bank/earbuds/camera/other/null,
      emotion: normal/anxious/angry/complaint_risk,
      confidence, reasoning
    }"""
    sys = """You are the PERCEPTION module for Anker OntoCare after-sales agent.
You ONLY perceive and understand the user's message; you do NOT make business decisions.

Given the user's message and the current product in conversation, return JSON with these fields:
- "is_greeting": true ONLY if the message contains a greeting and NOTHING else (no problem, no request). If a greeting is followed by ANY issue or request, set it to false.
- "is_new_topic": true if the user starts a DIFFERENT product/issue, or a completely unrelated question
- "is_irrelevant": true if the message has NOTHING to do with Anker electronics (e.g. paper box, clothes, weather, random gibberish)
- "logistics_damage": true if the customer's PACKAGE/parcel/box/outer packaging arrived damaged, crushed, torn or wet in shipping (a transit problem, NOT a product malfunction). e.g. "快递破了", "box arrived crushed"
- "product_hint": one of "breast_pump", "robot_vacuum", "power_bank", "earbuds", "camera", "speaker", "other", or null if unclear
- "emotion": one of "normal", "anxious" (urgent, in a hurry, event soon), "angry" (furious, terrible, awful, !!!), "complaint_risk" (will complain, lawyer, BBB, ridiculous)
- "confidence": 0.0-1.0 for your product/emotion judgment
- "reasoning": one short sentence

Important disambiguation for "S1 Pro":
- Words like "sucking", "suction problem on pump", "breast", "flange", "milk" → breast_pump
- Words like "robot", "vacuum", "roller brush", "dock", "won't navigate" → robot_vacuum
- "not sucking" with S1 Pro → breast_pump (a breast pump sucks milk)
- "robot died" → robot_vacuum

Reply with ONLY valid JSON, no markdown, no extra text."""
    user = f"Current product in conversation: {current_product_name or 'none'}\nUser message: {text}"
    out = call_llm(sys, user, timeout=12)
    default = {"is_greeting": False, "is_new_topic": False, "is_irrelevant": False,
               "logistics_damage": False,
               "product_hint": None, "emotion": None, "confidence": 0.0, "reasoning": "LLM unavailable"}
    if not out:
        return default
    try:
        s, e = out.find("{"), out.rfind("}")
        if s >= 0 and e > s:
            data = json.loads(out[s:e+1])
            for k, v in default.items():
                data.setdefault(k, v)
            return data
    except Exception as ex:
        print("[perceive JSON parse failed]", ex, out[:100])
    return default

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

# ---------- 物流 / 外包装损坏（独立于产品故障：走破损登记 / 补发 / 索赔） ----------
LOGISTICS_ZH = ["快递", "包裹", "外包装", "包装", "纸箱", "外箱", "面单", "封箱", "运输"]
DAMAGE_ZH    = ["破", "损", "坏", "碎", "扁", "塌", "压", "裂", "湿", "漏"]
def detect_logistics_damage(text: str) -> bool:
    """识别"收到的包裹/外包装在运输中破损"。这是物流问题，不应直接定位成某产品故障。"""
    t = text.lower()
    # 中文：物流词 + 破损词
    if any(w in t for w in LOGISTICS_ZH) and any(b in t for b in DAMAGE_ZH):
        return True
    # 英文：package/parcel/box/transit/delivery + damaged/crushed/torn/broken
    if re.search(r"packag|parcel|shipping box|\bbox\b|in transit|delivery|courier|envelope", t) and \
       re.search(r"damag|crush|torn|broken|dented|destroyed|ripped|wet|leak", t):
        return True
    return False


PROMOTER_WORDS = ["pump", "flange", "breast", "duckbill", "membrane", "nipple",
                  "suck", "sucks", "sucking", "suction", "milk", "leak", "漏奶", "吸奶"]
ROBOT_WORDS    = ["vacuum", "robot", "brush", "dock", "docking", "navigate", "扫地", "吸尘", "滚刷",
                  "robot vacuum", "mop", "sweep"]

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
def allowed_actions(case: dict, asked_refund: bool = False, asked_replacement: bool = False):
    """返回 (allowed[], forbidden[], reasons[])。规则全部确定性，LLM 只能在 allowed 里选。
    asked_* 用于"理由降噪"：用户没提退款/换货时，动作仍在 forbidden 兜底，但不显示
    相关 CRITICAL/HIGH 理由，避免右栏出现与当前诉求无关的噪音。"""
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
    can_replace = (ts == "FAILED" and dealer == "AUTHORIZED")
    if can_replace:
        allowed += ["propose_repair", "propose_replacement"]
        reasons.append("排障失败 + 在保 + 授权经销商：可进入换货建议（HIGH，需人工审批）")
    else:
        forbidden += ["propose_replacement"]  # 安全兜底（无论用户是否提及）
        if asked_replacement and dealer != "NOT_AUTHORIZED":
            if ts != "FAILED":
                reasons.append("您申请换货：需先完成排障且失败，当前先引导排障")
            else:
                reasons.append("暂不满足换货条件，已转人工核验")
    if dealer == "NOT_AUTHORIZED":
        forbidden += ["propose_replacement", "direct_refund"]
        allowed += ["guide_contact_seller"]
        reasons.append("非授权经销商：引导联系购买渠道，不进入官方保修")
    elif case.get("within_30d"):
        allowed += ["direct_refund"]
        reasons.append("下单30天内 + 授权：命中30天无理由，退款可走（MEDIUM，留痕）")
    else:
        forbidden += ["direct_refund"]  # 安全兜底
        if asked_refund:
            reasons.append("您申请退款：不满足30天/授权条件，直接退款需人工裁决")
        # 用户没提退款 → 不显示该理由（右栏降噪）
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

def fresh_case() -> dict:
    """全新的本体 Case State（新话题/无关问题时重置）。"""
    return {"product": None, "evidence": [], "warranty": "UNKNOWN",
            "dealer": "UNKNOWN", "troubleshooting": "NOT_STARTED",
            "tool_trace": [], "llm_fail_count": 0}

def trace_tool(case: dict, action: str, inputs: dict, result: str):
    """记录一次工具/查询调用（可审计）：动作名 + 输入 + 结果。"""
    case.setdefault("tool_trace", []).append({
        "action": action,
        "input": {k: str(v)[:60] for k, v in inputs.items() if v},
        "result": str(result)[:80],
    })

def _trace_view(emotion, scope, case, order_id, allowed, forbidden, reasons,
                evidence=None, img_name=None, vlm_ev=None):
    """构造右栏 Decision Trace 视图（含工具审计）。"""
    product = case.get("product") if case else None
    t = {
        "emotion": emotion,
        "scope": scope,
        "product": product["name"] if product else "—",
        "order": order_id or "未提供",
        "warranty": case.get("warranty", "—") if case else "—",
        "dealer": case.get("dealer", "—") if case else "—",
        "troubleshooting": case.get("troubleshooting", "—") if case else "—",
        "allowed": allowed, "forbidden": forbidden, "reasons": reasons,
        "tool_trace": case.get("tool_trace", []) if case else [],
    }
    if evidence is not None:
        t["evidence"] = evidence
    if img_name or vlm_ev:
        t["image"] = (f"{img_name} · VLM:{vlm_ev.get('error_code','?')}/{vlm_ev.get('visible_issue','?')}"
                      if vlm_ev else (img_name or "无"))
    return t

@app.route("/api/chat", methods=["POST"])
def chat():
    body = request.get_json(force=True)
    text = (body.get("message", "") or "").strip()
    img = body.get("image")  # 形如 "uploaded:fault.jpg"
    img_name = img.split(":", 1)[1] if img else None
    vlm_ev = VISION_CACHE.get(img_name) if img_name else None
    sid = body.get("session_id", "demo")
    case = SESSIONS.get(sid, fresh_case())

    # 从消息里抓订单号；本轮没给则沿用会话里已确认的（多轮记忆）
    m = re.search(r"ORD-\d{4}", text.upper())
    order_id = m.group(0) if m else body.get("order_id") or case.get("order_id")
    if order_id:
        case["order_id"] = order_id
    explicit_oid = bool((m and m.group(0)) or body.get("order_id"))

    # ===== L1 感知层：LLM 统一理解（产品/情绪/新话题/问候/无关）=====
    cur_name = case.get("product", {}).get("name") if case.get("product") else None
    perception = llm_perceive(text, cur_name)

    # 情绪：LLM 判断优先，LLM 失败/不确定再用正则兜底
    emotion = None
    if perception.get("emotion") and perception["emotion"] != "normal":
        emotion = perception["emotion"].upper()
    if not emotion:
        emotion = detect_emotion(text)

    scope = detect_scope(text)

    # 问候：LLM 判断优先，再用关键词兜底
    greetings = ["hi", "hello", "hey", "hii", "hiya", "你好", "哈喽", "嗨", "早上好", "下午好", "晚上好", "在吗"]
    is_greeting = perception.get("is_greeting") or (
        text.lower() in greetings or (len(text) <= 6 and any(g in text.lower() for g in greetings)))
    # 兜底：问候后若带实质诉求（破损/故障/退款等），不算纯问候，必须进入业务流程
    has_substance = bool(re.search(
        r"破|损|坏|碎|裂|漏|退|换|修|refund|replace|repair|broken|damag|crush|torn|not work|doesn't|charge|"
        r"leak|offline|issue|problem|fault|error|stop|dead", text.lower()))
    if has_substance:
        is_greeting = False

    # 极短/乱码输入（<3字符且非订单号、非问候）：重置，引导描述
    clean = text.lower()
    if len(clean) < 3 and not is_greeting and not re.search(r'ORD-\d{4}', clean.upper()):
        case = fresh_case()
        SESSIONS[sid] = case
        return jsonify({
            "reply": "Hi! I'm here to help. Could you tell me a bit more? You can describe the problem, "
                     "upload a photo of the error, or share your order number (ORD-XXXX).",
            "cards": [],
            "trace": _trace_view(emotion, scope, None, order_id,
                                 ["ask_product", "search_knowledge"],
                                 ["propose_replacement", "direct_refund"],
                                 ["输入过短/不完整：重置上下文，引导用户描述问题"], None)})

    # 问候：先友好回应，不进入业务流程
    if is_greeting:
        return jsonify({
            "reply": "Hi there! I'm OntoCare, Anker's after-sales assistant. What seems to be the problem? "
                     "You can tell me the product issue, upload a photo of the error, or share your order number (ORD-XXXX).",
            "cards": [],
            "trace": _trace_view(emotion, scope, None, order_id,
                                 ["ask_product", "search_knowledge"],
                                 ["propose_replacement", "direct_refund"],
                                 ["用户问候：先友好回应，引导描述问题"], None)})

    # 服务边界：企业/媒体/批量等超范围 → 直接转人工
    if scope != "IN_SCOPE":
        return jsonify({"reply": "This looks outside what our automated support can handle — I'm connecting you to a human specialist now. "
                                 "No need to repeat your issue, I'll pass the context along.",
                        "cards": [{"title": "Escalated to human", "body": "Reason: out of scope. A specialist will pick this up shortly."}],
                        "trace": _trace_view(emotion, scope, None, order_id,
                                             ["escalate_human"], [],
                                             ["服务边界命中：超出自助范围，直接转人工"], None)})

    # ===== 新话题 / 无关问题：重置本体状态（解决"纸箱子串台"）=====
    is_irrelevant = perception.get("is_irrelevant")
    is_new_topic = perception.get("is_new_topic")
    product_hint = perception.get("product_hint")

    # 完全无关（纸箱子/衣服/天气/乱码）：重置，友好说明我们只做 Anker 电子产品
    if is_irrelevant and product_hint in (None, "other"):
        case = fresh_case()
        SESSIONS[sid] = case
        return jsonify({
            "reply": "Thanks for reaching out! Just to clarify, I only support Anker electronics such as chargers, "
                     "power banks, earbuds, robot vacuums and breast pumps — we don't sell other items like that. "
                     "Which Anker product can I help you with?",
            "cards": [],
            "trace": _trace_view(emotion, scope, None, order_id,
                                 ["ask_product", "search_knowledge"],
                                 ["propose_replacement", "direct_refund"],
                                 ["LLM 感知：问题与 Anker 电子产品无关，重置 Case，引导正确产品"], None)})

    # 新话题且提到了（可能的）产品：重置产品与排障，重新走识别
    if is_new_topic and product_hint and product_hint != "other":
        case["product"] = None
        case["troubleshooting"] = "NOT_STARTED"
        case["product_state"] = None

    # ===== 物流 / 外包装损坏：独立流程（破损登记 + 补发/索赔），不定位产品 =====
    logistics_damage = bool(perception.get("logistics_damage")) or detect_logistics_damage(text)
    if logistics_damage:
        case["product"] = None
        case["troubleshooting"] = "NOT_STARTED"
        case["warranty"] = "UNKNOWN"
        cards = []
        if order_id:
            od = lookup_order(order_id)
            trace_tool(case, "lookup_order", {"order_id": order_id}, od["status"])
            if od["status"] == "FOUND":
                cards = [{"title": "Logistics damage reported", "body":
                          f"Order {order_id}: outer packaging arrived damaged.\n"
                          "→ Shipping-damage claim logged; arranging inspection & reshipment.\n"
                          "Please upload photos of the outer box + shipping label."}, nss_card()]
                reply = (f"I'm sorry your order {order_id} arrived with damaged packaging. I've logged a "
                         "shipping-damage claim and we'll inspect it and arrange a reshipment. Please upload "
                         "photos of the outer box and the shipping label to speed this up.")
                allowed = ["report_logistics_damage", "arrange_reshipment", "request_photos", "escalate_human"]
                forbidden = ["direct_refund"]
                reasons = [f"物流破损：订单 {order_id} 已登记，安排核查与补发/索赔（需外包装+面单照片）"]
            else:
                reply = ("I'm sorry about the damaged package. I couldn't find that order yet — could you "
                         "confirm your order number and upload photos of the outer box and shipping label?")
                allowed = ["ask_order", "request_photos", "report_logistics_damage"]
                forbidden = ["direct_refund", "propose_replacement"]
                reasons = ["物流破损但订单未核实：先确认订单号与外包装/面单照片"]
        else:
            reply = ("I'm really sorry your package arrived damaged. To open a shipping-damage claim, could you "
                     "share your order number and upload a photo of the outer box (and the shipping label)? I'll take it from there.")
            allowed = ["ask_order", "request_photos", "report_logistics_damage"]
            forbidden = ["direct_refund", "propose_replacement"]
            reasons = ["物流破损：先收集订单号 + 外包装/面单照片，再登记补发/索赔"]
        trace = _trace_view(emotion, scope, case, order_id, allowed, forbidden, reasons, None)
        SESSIONS[sid] = case
        return jsonify({"reply": reply, "cards": cards, "trace": trace})

    # ===== L2 数据流：订单 / 经销商（确定性查询，留痕）=====
    order = lookup_order(order_id) if order_id else {"status": "NOT_FOUND", "warranty": "UNKNOWN"}
    if order_id:
        trace_tool(case, "lookup_order", {"order_id": order_id}, order["status"])
    if order["status"] == "FOUND":
        sku = order["order"]["sku"]
        case["warranty"] = order["warranty"]
        d = match_dealer(order["order"]["country"], order["order"]["seller"])
        trace_tool(case, "verify_dealer",
                   {"country": order["order"]["country"], "seller": order["order"]["seller"]},
                   d["status"])
        case["dealer"] = d["status"]
        case["within_30d"] = order.get("within_30d", False)
    else:
        sku = None
        if explicit_oid:
            case["warranty"] = "UNKNOWN"

    # ===== L3 本体：产品消歧（订单 SKU 最强；LLM 感知 + 关键词兜底）=====
    pr = resolve_product(text, sku)

    # 若规则消歧没结果，但 LLM 感知明确给了品类，用感知结果兜底（PROBABLE）
    if not pr["product"] and pr["state"] in ("CONFLICTED", "UNKNOWN") and product_hint:
        hint_map = {"breast_pump": "breast_pump", "robot_vacuum": "robot_vacuum",
                    "power_bank": "power_bank", "earbuds": "earbuds",
                    "camera": "camera", "speaker": "speaker"}
        cat = hint_map.get(product_hint)
        if cat:
            for p in PRODUCTS:
                if p["category"] == cat:
                    pr = {"product": p, "evidence": f"llm_perception={cat}", "state": "PROBABLE"}
                    break

    if pr["product"]:
        case["product"] = pr["product"]
        case["product_state"] = pr["state"]
        case["product_unknown"] = False
    elif pr["evidence"] == "no_product_mentioned":
        # 用户没提新产品词，沿用之前的产品（如"还是不行""ok"）
        pass
    else:
        case["product"] = None
        case["product_state"] = pr["state"]
        case["product_unknown"] = (pr["evidence"] == "unknown_product")

    # 知识库检索（本体约束），留痕
    fault = match_fault(case["product"]["id"], text) if case["product"] else None
    if case["product"]:
        trace_tool(case, "search_knowledge",
                   {"product": case["product"]["name"], "query": text[:40]},
                   fault["fault_name"] if fault else "no match")

    if "replace" in text.lower() or "换货" in text:
        case["troubleshooting"] = "FAILED"  # demo：用户主动要求换货视作排障失败

    # ===== L4 决策：Decision Ontology 计算合法动作空间 =====
    low = text.lower()
    asked_refund = bool(re.search(r"refund|money back|退款|退钱", low))
    asked_replacement = bool(re.search(r"replace|replacement|exchange|换货|换新", low))
    allowed, forbidden, reasons = allowed_actions(case, asked_refund, asked_replacement)

    # 1) 确定性后端先给结构化卡片 + 模板兜底
    reply, cards = build_reply(case, emotion, order_id, order, fault, allowed, forbidden)
    # 2) 再让 LLM 在【合法动作】内把话术写自然；失败则用模板
    system = ("You are OntoCare, Anker's after-sales agent for overseas customers. "
              "STRICT RULES: recommend ONLY actions in [ALLOWED]; NEVER promise anything in [FORBIDDEN]. "
              "Do not guess warranty or dealer status — if evidence says UNKNOWN, say so and ask. "
              "Be concise, warm, and match the user's emotion. Do NOT repeat troubleshooting steps already shown as cards. "
              "Always reply in English. If you don't know the answer, say so and offer a human.")
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
        reply = llm_reply
        case["llm_fail_count"] = 0
    else:
        # LLM 失败：模板兜底，累计失败次数；连续 2 次主动提示转人工
        case["llm_fail_count"] = case.get("llm_fail_count", 0) + 1
        if case["llm_fail_count"] >= 2:
            reply = ("I'm having a little trouble on my side right now. Would you like me to connect you "
                     "with a human specialist instead? They'll have the full context, so you won't need to repeat anything.")

    trace = _trace_view(emotion, scope, case, order_id, allowed, forbidden, reasons,
                        pr["evidence"], img_name, vlm_ev)
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
