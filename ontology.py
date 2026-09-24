"""Small executable checks for the M0 service ontology and policy bundle."""


def validate_relation(ontology, relation, domain, range_):
    spec = ontology["relationships"].get(relation)
    if not spec or spec["domain"] != domain or spec["range"] != range_:
        raise ValueError("Ontology relation mismatch: %s(%s, %s)" % (relation, domain, range_))


def validate_fixture_relations(ontology, orders, products):
    validate_ontology(ontology)
    product_skus = {product["sku"] for product in products}
    for product in products:
        validate_relation(ontology, "INSTANCE_OF", "SKU", "Product")
    for order in orders.values():
        validate_relation(ontology, "PLACED", "Customer", "Order")
        validate_relation(ontology, "CONTAINS", "Order", "SKU")
        validate_relation(ontology, "SOLD_BY", "Order", "Seller")
        if order.get("sku") not in product_skus:
            raise ValueError("Order fixture references an unknown SKU")
        if not order.get("seller") or not order.get("country") or not order.get("customer_email"):
            raise ValueError("Order fixture is missing relationship evidence")


def validate_ontology(ontology):
    """Reject malformed ontology declarations before the app serves decisions."""
    entities = set(ontology.get("entities", []))
    if not entities or not ontology.get("version"):
        raise ValueError("Ontology must define entities and a version")
    for relation, spec in ontology.get("relationships", {}).items():
        if spec.get("domain") not in entities or spec.get("range") not in entities:
            raise ValueError("Ontology relation has an unknown entity: %s" % relation)
    if not ontology.get("case_states") or not ontology.get("actions"):
        raise ValueError("Ontology must define case states and actions")


def validate_policy_links(ontology, policy):
    """Ensure runtime policy actions have explicit ontology links and valid predicates."""
    validate_relation(ontology, "APPLIES_TO", "Policy", "Action")
    for action in ("prepare_return_review", "prepare_replacement_review"):
        if action not in ontology["actions"] or action not in policy.get("actions", {}):
            raise ValueError("Policy action is missing from ontology: %s" % action)
        spec = policy["actions"][action]
        if spec.get("auto") is not True or spec.get("effect") != "draft_only":
            raise ValueError("High-risk policy action must be draft-only: %s" % action)
        for condition in spec.get("preconditions", []):
            if condition not in ACTION_PREDICATES:
                raise ValueError("Unknown policy precondition: %s" % condition)


def validate_phase(ontology, phase):
    if phase not in ontology["case_states"]:
        raise ValueError("Unknown ontology case state: %s" % phase)
    return phase


def validate_actions(ontology, actions):
    unknown = set(actions) - set(ontology["actions"])
    if unknown:
        raise ValueError("Unknown ontology action(s): %s" % ", ".join(sorted(unknown)))


ACTION_PREDICATES = {
    "return_requested": lambda case: case.get("requested_action") in ("RETURN_REVIEW", "REFUND_REVIEW"),
    "replacement_requested": lambda case: case.get("requested_action") == "REPLACEMENT_REVIEW",
    "order_found": lambda case: case.get("order_status") == "FOUND",
    "within_30days": lambda case: case.get("within_30d") is True,
    "warranty_valid": lambda case: case.get("warranty") == "VALID",
    "dealer_authorized": lambda case: case.get("dealer") == "AUTHORIZED",
    "troubleshooting_failed": lambda case: case.get("troubleshooting") == "FAILED",
}


def policy_allows(policy, action, case):
    """Evaluate named policy predicates from a fixed Python whitelist; never eval JSON."""
    spec = policy.get("actions", {}).get(action)
    if not spec or spec.get("auto") is not True:
        return False
    for condition in spec.get("preconditions", []):
        predicate = ACTION_PREDICATES.get(condition)
        if predicate is None or not predicate(case):
            return False
    return True
