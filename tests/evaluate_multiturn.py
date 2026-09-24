"""Replay four multi-turn customer journeys against the local rule path.

This deliberately disables model calls. PASS means the listed safety/context
checks pass; it is not a real-model success rate or customer resolution rate.
"""
import csv
import os
import sys
from unittest.mock import patch
from flask import g

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import app


SCENARIOS = {
    "pump_followups": [
        "我的 S1 Pro 不吸了", "是吸奶器", "第二步没看懂", "我照做了还是不吸",
        "我的订单 ORD-2001，能换货吗？", "刚才说的第二步是什么？",
    ],
    "order_switch": [
        "ORD-2002 robot won't charge", "hi", "第二步没看懂",
        "其实我的订单是 ORD-2001", "那现在是不是在保？",
    ],
    "shipping_followups": [
        "ORD-2002 我的快递外包装破了", "需要拍什么照片？",
        "我现在发不了图，你已经安排补发了吗？",
    ],
    "product_conflict": [
        "ORD-2001 my robot vacuum won't charge", "还是充不上电",
        "你刚才说的是哪个产品？",
    ],
}


def check(scenario, turn, data):
    t, reply = data["trace"], data["reply"]
    if scenario == "pump_followups":
        return [
            lambda: t["product_state"] == "CONFLICTED",
            lambda: "Breast Pump" in t["product"] and t["knowledge_source"] == "suction_failure",
            lambda: "Check duckbill valve for cracks." in reply and "你之前反馈尝试无效" not in reply,
            lambda: t["user_reported_failure"] and t["troubleshooting"] == "USER_REPORTED_FAILED",
            lambda: t["order"] == "ORD-2001" and t["user_reported_failure"]
            and t["handoff"]["status"] == "DRAFT_NOT_SUBMITTED"
            and "propose_replacement" in t["forbidden"],
            lambda: t["order"] == "ORD-2001" and t["user_reported_failure"]
            and "Check duckbill valve for cracks." in reply,
        ][turn - 1]()
    if scenario == "order_switch":
        return [
            lambda: "Robot Vacuum" in t["product"] and t["order"] == "ORD-2002",
            lambda: "Robot Vacuum" in t["product"] and t["order"] == "ORD-2002",
            lambda: "Clear the dock path" in reply and t["order"] == "ORD-2002",
            lambda: "Breast Pump" in t["product"] and t["order"] == "ORD-2001",
            lambda: "模拟订单" in reply and "订单归属尚未核验" in reply,
        ][turn - 1]()
    if scenario == "shipping_followups":
        return [
            lambda: t["product_state"] == "NOT_REQUIRED_LOGISTICS" and "尚未登记索赔或安排补发" in reply,
            lambda: t["product_state"] == "NOT_REQUIRED_LOGISTICS"
            and "外包装破了" in t["handoff"]["issue"],
            lambda: t["product_state"] == "NOT_REQUIRED_LOGISTICS"
            and "无法提供照片" in reply and "尚未登记索赔或安排补发" in reply,
        ][turn - 1]()
    return [
        lambda: t["product_state"] == "CONFLICTED" and t["warranty"] == "PENDING_RECHECK",
        lambda: t["product_state"] == "CONFLICTED" and t["warranty"] == "PENDING_RECHECK",
        lambda: "Robot Vacuum" in reply and "Breast Pump" in reply
        and t["warranty"] == "PENDING_RECHECK",
    ][turn - 1]()


def main(path, real=False):
    if real and not (app.LLM_KEY and app.LLM_MODEL):
        print("NOT_CONFIGURED: a real model key and model are required")
        return 1
    if not real:
        app.LLM_KEY = ""
        app.LLM_MODEL = ""
    app.SESSIONS.clear()
    client = app.app.test_client()
    rows = []
    for scenario, messages in SCENARIOS.items():
        for turn, message in enumerate(messages, 1):
            if real:
                response = client.post("/api/chat", json={
                    "message": message, "session_id": "replay-" + scenario, "language": "zh",
                })
            else:
                def test_adapter(system, user, timeout=15):
                    g.model_status = "TEST_STUB"
                    if "intent understanding module" in system:
                        return '{"is_new_topic": false, "mentioned_product": null, "is_irrelevant": false, "step_number": null}'
                    return "你好，我也可以继续帮你处理安克产品问题。"
                with patch.object(app, "call_llm", side_effect=test_adapter):
                    response = client.post("/api/chat", json={
                        "message": message, "session_id": "replay-" + scenario, "language": "zh",
                    })
            data = response.get_json() or {}
            trace = data.get("trace") or {}
            expected_model = "OK" if real else "TEST_STUB"
            passed = response.status_code == 200 and trace.get("model_status") == expected_model and check(scenario, turn, data)
            rows.append({"scenario": scenario, "turn": turn, "question": message,
                         "status": "PASS" if passed else "FAIL", "reply": data.get("reply", ""),
                         "product_state": trace.get("product_state", ""),
                         "order": trace.get("order", ""), "warranty": trace.get("warranty", ""),
                         "troubleshooting": trace.get("troubleshooting", ""),
                         "reported_failure": trace.get("user_reported_failure", ""),
                         "model_status": trace.get("model_status", data.get("model_status", "")),
                         "request_id": data.get("request_id", "")})
    if path:
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    passed = sum(row["status"] == "PASS" for row in rows)
    label = "real-model" if real else "local rule"
    print(f"Multi-turn {label} replay: {passed}/{len(rows)} PASS")
    return int(passed != len(rows))


if __name__ == "__main__":
    real = "--real" in sys.argv[1:]
    paths = [arg for arg in sys.argv[1:] if arg != "--real"]
    raise SystemExit(main(paths[0] if paths else None, real))
