"""Small dialogue state tracker for the demo's bounded after-sales tasks.

The case stores claims and verified fixture facts separately. This module
tracks conversation flow; it never authorizes a refund or replacement.
"""
import re


PHASES = ("CONSULTATION", "VERIFY", "TROUBLESHOOT", "REVIEW", "CLOSED")
END_WORDS = {
    "没", "没有", "没了", "没有了", "不用了", "谢谢", "谢谢你", "先这样",
    "就这样", "结束", "没问题了", "没有其他问题了", "thanks", "thank you",
    "no thanks", "that's all", "all good", "no more questions",
    "不用了谢谢", "谢谢不用了", "没有了谢谢", "没了谢谢",
    "再见", "拜拜", "回头见", "goodbye", "bye", "see you",
}


def _normalize(text):
    return re.sub(r"[\s\u3000，。！？!?,.～~]+", "", text.strip().lower())


def is_ending(text):
    """Match closing utterances while allowing a farewell after a short sentence."""
    normalized = _normalize(text)
    if normalized in END_NORMALIZED:
        return True
    if not re.search(r"(?:再见|拜拜|回头见|goodbye|bye|seeyou)$", normalized):
        return False
    # A farewell after an active after-sales question is still part of that request.
    service_terms = (
        "安克", "anker", "订单", "产品", "型号", "质保", "保修", "退款", "退货",
        "换货", "故障", "坏了", "充电", "吸奶", "扫地", "报错", "维修",
        "order", "product", "model", "warranty", "refund", "return", "replace",
        "repair", "broken", "charging", "error", "how", "what", "can i",
    )
    return not any(term in normalized for term in service_terms)


END_NORMALIZED = {_normalize(word) for word in END_WORDS}


def redact_for_memory(text):
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[email]", text)
    text = re.sub(r"(?<!\d)1[3-9]\d{9}(?!\d)", "[phone]", text)
    return text[:300]


def remember_turn(case, user_text, reply):
    case.setdefault("history", []).append({
        "user": redact_for_memory(user_text),
        "assistant": redact_for_memory(reply),
    })
    case["history"] = case["history"][-4:]


def step_reference(text, model_step=None):
    match = re.search(r"第\s*([一二三四1-4])\s*(?:步|条)|\bstep\s*([1-4])\b", text, re.I)
    if match:
        digit = match.group(1) or match.group(2)
        return {"一": 1, "二": 2, "三": 3, "四": 4}.get(digit, int(digit) if digit.isdigit() else None)
    if re.search(r"那(?:个|一)步|刚才.*步骤|that step", text, re.I) and type(model_step) is int:
        return model_step if 1 <= model_step <= 4 else None
    return None


def slots_view(case, order, fault):
    product = case.get("product")
    slots = {
        "product": {"value": product["name"] if product else None,
                    "source": case.get("product_state", "UNKNOWN")},
        "order_id": {"value": case.get("order_id"), "source": "USER_CLAIM"},
        "order_record": {"value": order["status"], "source": "TEST_FIXTURE"},
        "symptom": {"value": redact_for_memory(case["last_issue"]) if case.get("last_issue") else None,
                    "source": "USER_CLAIM"},
        "knowledge": {"value": fault["fault"] if fault else None,
                      "source": "DEMO_KB" if fault else "NONE"},
        "requested_action": {"value": case.get("requested_action"), "source": "USER_CLAIM"},
        "purchase_proof": {"value": None, "source": "NOT_VERIFIED"},
    }
    missing = []
    if not product and not case.get("logistics_damage"):
        missing.append("产品型号")
    if not case.get("last_issue") and not case.get("logistics_damage") and not case.get("requested_action"):
        missing.append("具体故障")
    if case.get("requested_action") and order["status"] != "FOUND":
        missing.append("订单号与购买凭证")
    if case.get("requested_action") and order["status"] == "FOUND":
        missing.append("订单归属与购买凭证")
    if case.get("logistics_damage"):
        missing.append("外箱与面单照片")
    return slots, missing


def phase_for(case, order, fault):
    if case.get("phase") == "CLOSED":
        return "CLOSED"
    if case.get("logistics_damage"):
        return "VERIFY"
    if case.get("requested_action"):
        return "REVIEW" if case.get("product") and order["status"] == "FOUND" else "VERIFY"
    if not case.get("product"):
        return "CONSULTATION"
    if fault:
        return "TROUBLESHOOT"
    return "VERIFY"
