"""Small, deterministic service workflow inspired by ServiceMind's routing and traces.

These are work items, not claims that a separate agent or external tool ran.
"""
import re


def work_items(text, case, order, fault, scope):
    t = text.lower()
    items = []
    if scope != "IN_SCOPE":
        return ["human_handoff"]
    if case.get("logistics_damage"):
        return ["shipping_damage_review", "order_check"]
    if t in ("hi", "hello", "hey", "你好", "嗨", "在吗", "早上好", "下午好", "晚上好"):
        return ["greeting"]
    if case.get("product_state") == "CONFLICTED" or not case.get("product"):
        items.append("confirm_product")
    if re.search(r"退款|退货|refund|return", t):
        items.append("refund_review")
    if re.search(r"换货|更换|replace|replacement|exchange", t):
        items.append("replacement_review")
    if case.get("order_id") or re.search(r"订单|质保|保修|order|warranty", t):
        items.append("order_check")
    if fault or re.search(r"不吸|充不上|故障|坏了|报错|不工作|not sucking|not charging|won't charge|dead|offline|error|problem", t):
        items.append("diagnosis")
    return items or ["clarify_issue"]


def tool_trace(name, status, source, result):
    """Safe summary only: never emit a fixture's email, address or other PII."""
    return {"tool": name, "status": status, "source": source, "result": result}


def handoff(case, order, fault, items, reasons):
    """A review draft; no ticket, refund or replacement is created."""
    return {
        "status": "DRAFT_NOT_SUBMITTED",
        "product": case["product"]["name"] if case.get("product") else "待确认",
        "order_id": case.get("order_id") or "未提供",
        "order_status": order["status"],
        "ownership": "NOT_VERIFIED",
        "issue": (case.get("last_issue") or "待补充")[:240],
        "knowledge": fault["fault"] if fault else "未命中",
        "requests": [i for i in items if i in ("refund_review", "replacement_review", "human_handoff", "shipping_damage_review")],
        "missing": [
            label for needed, label in (
                (not case.get("product") and not case.get("logistics_damage"), "产品型号"),
                (order["status"] != "FOUND", "购买凭证和订单信息"),
                (True, "订单归属核验"),
                (case.get("date_claim_pending"), "更正日期的支持凭证"),
                (case.get("logistics_damage"), "外包装与面单照片"),
            ) if needed
        ],
        "blocked_reasons": reasons,
    }
