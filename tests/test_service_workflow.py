"""Checks for the transferred workflow on the real Flask route, without a model key."""
import unittest
import app


class ServiceWorkflow(unittest.TestCase):
    def setUp(self):
        app.SESSIONS.clear()
        app.LLM_KEY = ""
        self.client = app.app.test_client()

    def ask(self, message, session="flow", **fields):
        response = self.client.post("/api/chat", json={"message": message,
                                   "session_id": session, **fields})
        self.assertEqual(response.status_code, 200)
        return response.get_json()

    def test_composite_request_keeps_diagnosis_and_review(self):
        result = self.ask("ORD-2002 robot won't charge and I want a refund", language="en")
        trace = result["trace"]
        self.assertIn("diagnosis", trace["work_items"])
        self.assertIn("refund_review", trace["work_items"])
        self.assertEqual(trace["knowledge_source"], "not_charging")
        self.assertTrue(any("troubleshooting" in card["title"].lower()
                            for card in result["cards"]))
        self.assertIn("direct_refund", trace["forbidden"])
        self.assertEqual(trace["handoff"]["status"], "DRAFT_NOT_SUBMITTED")
        self.assertIn("订单归属核验", trace["handoff"]["missing"])

    def test_conflicting_order_and_product_remains_unverified_across_turns(self):
        first = self.ask("ORD-2001 my robot vacuum won't charge")
        self.assertEqual(first["trace"]["product_state"], "CONFLICTED")
        self.assertEqual(first["trace"]["warranty"], "PENDING_RECHECK")
        second = self.ask("It still does not work")
        self.assertEqual(second["trace"]["product_state"], "CONFLICTED")
        self.assertEqual(second["trace"]["warranty"], "PENDING_RECHECK")
        third = self.ask("Sorry, it is my breast pump")
        self.assertIn("Breast Pump", third["trace"]["product"])

    def test_tool_trace_has_sources_but_no_customer_details(self):
        result = self.ask("ORD-2002 robot won't charge")
        trace = result["trace"]
        self.assertEqual([s["tool"] for s in trace["tool_steps"]],
                         ["resolve_product", "lookup_order", "search_knowledge", "evaluate_policy"])
        self.assertNotIn("user_2002@example.com", str(result))
        self.assertNotIn("Hauptstr", str(result))

    def test_handoff_is_draft_and_never_claims_real_transfer(self):
        result = self.ask("I want to speak to a manager about my refund", language="en")
        trace = result["trace"]
        self.assertEqual(trace["work_items"], ["human_handoff"])
        self.assertEqual(trace["handoff"]["status"], "DRAFT_NOT_SUBMITTED")
        self.assertIn("not transferred", result["reply"])

    def test_separate_sessions_do_not_share_context(self):
        self.ask("ORD-2002 robot won't charge", session="a")
        other = self.ask("Hello", session="b")
        self.assertEqual(other["trace"]["order_status"], "NOT_PROVIDED")
        self.assertEqual(other["trace"]["product"], "未确定")

    def test_shipping_damage_is_review_only_and_keeps_context(self):
        first = self.ask("ORD-2002 我的快递外包装破了")
        trace = first["trace"]
        self.assertEqual(trace["work_items"], ["shipping_damage_review", "order_check"])
        self.assertEqual(trace["product_state"], "NOT_REQUIRED_LOGISTICS")
        self.assertIn("arrange_reshipment", trace["forbidden"])
        self.assertEqual(trace["handoff"]["status"], "DRAFT_NOT_SUBMITTED")
        self.assertNotIn("产品型号", trace["handoff"]["missing"])
        self.assertEqual(trace["tool_steps"][2]["status"], "SKIPPED")
        self.assertIn("尚未登记", first["reply"])
        second = self.ask("面单照片可以补充")
        self.assertEqual(second["trace"]["order"], "ORD-2002")
        self.assertIn("shipping_damage_review", second["trace"]["work_items"])

    def test_plain_greeting_keeps_case_and_mixed_greeting_handles_issue(self):
        self.ask("ORD-2002 robot won't charge")
        hello = self.ask("hi", language="en")
        self.assertEqual(hello["trace"]["order"], "ORD-2002")
        self.assertEqual(hello["trace"]["work_items"], ["greeting"])
        mixed = self.ask("hi, robot won't charge", language="en")
        self.assertIn("diagnosis", mixed["trace"]["work_items"])
        self.assertNotEqual(mixed["trace"]["work_items"], ["greeting"])

    def test_unrequested_refund_reason_is_hidden_but_action_still_blocked(self):
        result = self.ask("ORD-2002 robot won't charge")
        self.assertIn("direct_refund", result["trace"]["forbidden"])
        self.assertFalse(any("退款" in reason for reason in result["trace"]["reasons"]))

    def test_repeated_followups_keep_fault_step_and_unverified_failure(self):
        first = self.ask("我的 S1 Pro 不吸了")
        self.assertEqual(first["trace"]["product_state"], "CONFLICTED")
        confirmed = self.ask("是吸奶器")
        self.assertEqual(confirmed["trace"]["knowledge_source"], "suction_failure")
        step = self.ask("第二步没看懂")
        self.assertEqual(step["trace"]["troubleshooting"], "NEEDS_GUIDANCE")
        self.assertIn("Check duckbill valve for cracks.", step["reply"])
        self.assertNotIn("你之前反馈尝试无效", step["reply"])
        failed = self.ask("我照做了还是不吸")
        self.assertEqual(failed["trace"]["troubleshooting"], "USER_REPORTED_FAILED")
        self.assertTrue(failed["trace"]["user_reported_failure"])
        review = self.ask("我的订单 ORD-2001，能换货吗？")
        self.assertEqual(review["trace"]["troubleshooting"], "USER_REPORTED_FAILED")
        self.assertTrue(review["trace"]["user_reported_failure"])
        self.assertEqual(review["trace"]["handoff"]["status"], "DRAFT_NOT_SUBMITTED")
        self.assertIn("不吸", review["trace"]["handoff"]["issue"])
        self.assertIn("propose_replacement", review["trace"]["forbidden"])
        again = self.ask("刚才说的第二步是什么？")
        self.assertIn("Check duckbill valve for cracks.", again["reply"])
        self.assertTrue(again["trace"]["user_reported_failure"])
        self.assertEqual(again["trace"]["troubleshooting"], "USER_REPORTED_FAILED")
        self.assertEqual(again["trace"]["order"], "ORD-2001")

    def test_shipping_photo_followups_do_not_become_product_fault(self):
        self.ask("ORD-2002 我的快递外包装破了")
        photo = self.ask("需要拍什么照片？")
        self.assertEqual(photo["trace"]["product_state"], "NOT_REQUIRED_LOGISTICS")
        self.assertIn("外包装破了", photo["trace"]["handoff"]["issue"])
        self.assertEqual(photo["trace"]["knowledge_source"], "NONE")
        reship = self.ask("我现在发不了图，你已经安排补发了吗？")
        self.assertIn("尚未登记索赔或安排补发", reship["reply"])
        self.assertIn("无法提供照片", reship["reply"])
        self.assertIn("arrange_reshipment", reship["trace"]["forbidden"])

    def test_new_order_preserves_fault_but_switching_orders_resets_it(self):
        self.ask("我的 S1 Pro 不吸了")
        self.ask("是吸奶器")
        self.ask("我照做了还是不吸")
        first_order = self.ask("订单是 ORD-2001")
        self.assertEqual(first_order["trace"]["troubleshooting"], "USER_REPORTED_FAILED")
        switched = self.ask("其实订单是 ORD-2002")
        self.assertEqual(switched["trace"]["troubleshooting"], "NOT_STARTED")
        self.assertFalse(switched["trace"]["user_reported_failure"])
        self.assertIn("Robot Vacuum", switched["trace"]["product"])

    def test_order_attached_to_conflicting_prior_product_stays_unverified(self):
        self.ask("我的 S1 Pro 扫地机充不上电")
        order = self.ask("订单是 ORD-2001")
        self.assertEqual(order["trace"]["product_state"], "CONFLICTED")
        self.assertEqual(order["trace"]["warranty"], "PENDING_RECHECK")

    def test_warranty_followup_qualifies_fixture_and_ownership(self):
        self.ask("ORD-2002 robot won't charge")
        reply = self.ask("那现在是不是在保？")
        self.assertIn("模拟订单", reply["reply"])
        self.assertIn("订单归属尚未核验", reply["reply"])
        self.assertEqual(reply["trace"]["order"], "ORD-2002")

    def test_conflict_followup_names_claim_and_fixture_separately(self):
        self.ask("ORD-2001 my robot vacuum won't charge")
        self.ask("还是充不上电")
        reply = self.ask("你刚才说的是哪个产品？")
        self.assertIn("Robot Vacuum", reply["reply"])
        self.assertIn("Breast Pump", reply["reply"])
        self.assertEqual(reply["trace"]["warranty"], "PENDING_RECHECK")


if __name__ == "__main__":
    unittest.main()
