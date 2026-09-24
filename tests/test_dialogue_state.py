"""Conversation-state checks against the same route used by the browser."""
import unittest
from unittest.mock import patch
from flask import g
import app
from dialogue import is_ending, step_reference


class DialogueState(unittest.TestCase):
    def setUp(self):
        app.SESSIONS.clear()
        app.LLM_KEY = ""
        self.client = app.app.test_client()

    def ask(self, text, **extra):
        response = self.client.post("/api/chat", json={"message": text,
                                    "session_id": "dst", **extra})
        self.assertEqual(response.status_code, 200)
        return response.get_json()

    def test_phase_slots_and_no_repeated_order_question(self):
        first = self.ask("ORD-2002 robot won't charge")
        self.assertEqual(first["trace"]["phase"], "TROUBLESHOOT")
        self.assertEqual(first["trace"]["slots"]["order_id"]["value"], "ORD-2002")
        self.assertEqual(first["trace"]["slots"]["order_record"]["source"], "TEST_FIXTURE")
        self.assertEqual(first["trace"]["memory_turns"], 1)
        review = self.ask("能换货吗？")
        self.assertEqual(review["trace"]["phase"], "REVIEW")
        self.assertEqual(review["trace"]["slots"]["requested_action"]["value"], "REPLACEMENT_REVIEW")
        self.assertIn("订单归属与购买凭证", review["trace"]["missing_slots"])
        self.assertNotIn("请提供订单号", review["reply"])
        self.assertEqual(review["trace"]["memory_turns"], 2)

    def test_ending_keeps_case_and_requires_explicit_new_case(self):
        self.ask("ORD-2002 robot won't charge")
        with patch.object(app, "call_llm", side_effect=AssertionError("closing called model")):
            closed = self.ask("没")
            self.assertEqual(closed["trace"]["phase"], "CLOSED")
            self.assertEqual(closed["trace"]["order"], "ORD-2002")
            again = self.ask("谢谢")
            self.assertEqual(again["trace"]["phase"], "CLOSED")
            another = self.ask("我的充电宝坏了")
            self.assertEqual(another["trace"]["phase"], "CLOSED")
        fresh = self.ask("我的充电宝坏了", new_case=True)
        self.assertNotEqual(fresh["trace"]["phase"], "CLOSED")
        self.assertEqual(fresh["trace"]["order"], "未提供")

    def test_short_negative_is_not_any_sentence_containing_no(self):
        self.assertTrue(is_ending("没。"))
        self.assertTrue(is_ending("不用了，谢谢"))
        self.assertFalse(is_ending("没有订单"))
        self.assertFalse(is_ending("没有发票，能换货吗"))
        self.assertEqual(step_reference("刚才的第3步是什么"), 3)
        self.assertEqual(step_reference("那一步呢", 2), 2)

    def test_model_receives_bounded_history_without_contact_details(self):
        prompts = []

        def fake_call(system, user, timeout=15):
            g.model_status = "OK"
            prompts.append(user)
            return '{"is_new_topic": false, "mentioned_product": null, "is_irrelevant": false, "step_number": null}'

        with patch.object(app, "call_llm", side_effect=fake_call):
            self.ask("ORD-2002 robot won't charge. 邮箱 test@example.com 手机 13812345678")
            self.ask("第二步是什么？")
        self.assertEqual(len(prompts), 2)
        self.assertIn("recent_turns", prompts[1])
        self.assertIn("[email]", prompts[1])
        self.assertIn("[phone]", prompts[1])
        self.assertNotIn("test@example.com", prompts[1])
        self.assertNotIn("13812345678", prompts[1])

    def test_greeting_and_day_question_do_not_change_case(self):
        case = app.SESSIONS.setdefault("dst", {"product": None, "warranty": "UNKNOWN",
                    "dealer": "UNKNOWN", "troubleshooting": "NOT_STARTED",
                    "phase": "CONSULTATION", "history": []})
        case.update(order_id="ORD-2002", last_issue="robot won't charge", phase="TROUBLESHOOT")
        prompts = []

        def fake_call(system, user, timeout=15):
            g.model_status = "OK"
            prompts.append((system, user))
            return "今天是星期四。还需要我帮你查安克产品问题吗？"

        with patch.object(app, "call_llm", side_effect=fake_call):
            result = self.ask("hi 今天周几？")

        self.assertIn("星期四", result["reply"])
        self.assertEqual(result["trace"]["phase"], "TROUBLESHOOT")
        self.assertEqual(result["trace"]["order"], "ORD-2002")
        self.assertEqual(result["trace"]["memory_turns"], 0)
        self.assertNotIn("ORD-2002", prompts[0][1])
        self.assertNotIn("robot won't charge", prompts[0][1])

    def test_model_classified_side_question_preserves_case_and_history(self):
        case = app.SESSIONS.setdefault("dst", {"product": None, "warranty": "UNKNOWN",
                    "dealer": "UNKNOWN", "troubleshooting": "NOT_STARTED",
                    "phase": "CONSULTATION", "history": []})
        case.update(order_id="ORD-2002", last_issue="robot won't charge", phase="TROUBLESHOOT",
                    history=[{"user": "robot won't charge", "assistant": "Try step one."}])
        calls = []

        def fake_call(system, user, timeout=15):
            g.model_status = "OK"
            calls.append(user)
            if len(calls) == 1:
                return '{"is_new_topic": true, "mentioned_product": null, "is_irrelevant": true, "step_number": null}'
            return "今天天气我无法实时查询。需要的话，我可以继续帮你处理安克售后问题。"

        with patch.object(app, "call_llm", side_effect=fake_call):
            result = self.ask("今天附近天气怎么样？")

        self.assertIn("天气", result["reply"])
        self.assertEqual(result["trace"]["phase"], "TROUBLESHOOT")
        self.assertEqual(result["trace"]["order"], "ORD-2002")
        self.assertEqual(result["trace"]["memory_turns"], 1)
        self.assertEqual(len(case["history"]), 1)
        self.assertNotIn("ORD-2002", calls[1])
        self.assertNotIn("robot won't charge", calls[1])


if __name__ == "__main__":
    unittest.main()
