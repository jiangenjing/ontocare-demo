"""Run the existing 20 Golden questions through /api/chat, with no external model.

Usage: python tests/evaluate_endpoint.py [results.csv]
The fixture assertions remain rule checks. Additional reply and evidence checks
probe the actual route, but this is not a real-model task-completion score.
"""
import contextlib
import csv
import io
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.chdir(os.path.dirname(os.path.dirname(__file__)))
with contextlib.redirect_stdout(io.StringIO()):
    cases = runpy.run_path("eval_safe.py")["CASES"]
import app

app.LLM_KEY = ""
client = app.app.test_client()
rows = []
for index, item in enumerate(cases, 1):
    name, message, order_id, expected = item[:4]
    message = message or "E999 error"
    if index == 12:
        message += " I tried everything"
    response = client.post("/api/chat", json={
        "message": message, "order_id": order_id,
        "session_id": "golden-%02d" % index,
        "language": "en" if message.isascii() else "zh",
    })
    data = response.get_json() or {}
    trace = data.get("trace") or {}
    issues = []
    if response.status_code != 200:
        issues.append("HTTP %s" % response.status_code)
    if not data.get("reply"):
        issues.append("empty reply")
    if not trace.get("request_id") or len(trace.get("tool_steps", [])) != 4:
        issues.append("missing source trace")
    for field, want in expected.items():
        if field == "_fault":
            if trace.get("knowledge_source") != "unknown_error":
                issues.append("unknown code knowledge mismatch")
        elif field == "product":
            actual = next((p["id"] for p in app.PRODUCTS if p["name"] == trace.get("product")), None)
            if actual != want:
                issues.append("product %s != %s" % (actual, want))
        elif field in ("allowed", "forbidden"):
            for action in want:
                if index == 12 and action == "prepare_replacement_review":
                    # User's report of failure is not verified troubleshooting.
                    if action in trace.get("allowed", []):
                        issues.append("unverified failure authorized replacement")
                elif action not in trace.get(field, []):
                    issues.append("%s missing %s" % (field, action))
        elif trace.get(field) != want:
            issues.append("%s %s != %s" % (field, trace.get(field), want))
    if "direct_refund" not in trace.get("forbidden", []):
        issues.append("refund write not blocked")
    if any(s in data.get("reply", "").lower() for s in ("已到账", "已退款", "refund completed", "replacement completed")):
        issues.append("false completion claim")
    rows.append({"case": name, "status": "PASS" if not issues else "FAIL",
                 "issues": "; ".join(issues), "reply": data.get("reply", ""),
                 "work_items": ",".join(trace.get("work_items", [])),
                 "order_status": trace.get("order_status", ""),
                 "warranty": trace.get("warranty", "")})

passed = sum(r["status"] == "PASS" for r in rows)
print("Endpoint Golden: %s/%s PASS" % (passed, len(rows)))
for row in rows:
    if row["status"] != "PASS":
        print(row["case"], row["issues"])
if len(sys.argv) > 1:
    with open(sys.argv[1], "w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
sys.exit(0 if passed == len(rows) else 1)
