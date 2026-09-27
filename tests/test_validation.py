"""
MVP validation benchmark (Week 4).

The unit suites answer "does it still work?". This answers "how well does it
work?" - it measures the agent against labelled datasets and reports accuracy
per capability, so the numbers in the final report are measured rather than
asserted.

    python -m tests.test_validation            # full report
    python -m pytest tests/test_validation.py  # threshold assertions

No API key or network required.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from app.escalation import EscalationQueue, EscalationReason
from app.knowledge_base import get_knowledge_base
from app.order_flow import OrderDraft, build_order_context, update_draft
from app.orders import OrderStore
from app.query_handler import (
    Intent,
    detect_intent,
    extract_confirmation,
    extract_fulfilment,
    extract_person_name,
    extract_phone,
    extract_quantity,
    handle_query,
    name_from_bare_reply,
    phone_from_bare_reply,
)
from app.stt import is_probable_noise

kb = get_knowledge_base()


# --------------------------------------------------------------------------
# Labelled datasets
# --------------------------------------------------------------------------

# Medicine names as speech-to-text actually mangles them, mapped to the record
# they must resolve to.
MISHEARD_NAMES = {
    "amoxycillin": "Amoxil",
    "amoxisillin": "Amoxil",
    "amoxy sillin": "Amoxil",
    "amoxil": "Amoxil",
    "paracetemol": "Panadol",
    "para cetamol": "Panadol",
    "panadol": "Panadol",
    "tylenol": "Panadol",
    "ibuprofin": "Brufen",
    "advil": "Brufen",
    "ventalin": "Ventolin",
    "venta lin": "Ventolin",
    "albuterol": "Ventolin",
    "sal butamol": "Ventolin",
    "sitirizine": "Zyrtec",
    "sitirizeen": "Zyrtec",
    "zyrtek": "Zyrtec",
    "omeprazol": "Prilosec",
    "omez": "Prilosec",
    "azithromiacin": "Zithromax",
    "azithral": "Zithromax",
    "sifroflaxacin": "Cipro",
    "metfor min": "Glucophage",
    "glycomet": "Glucophage",
    "lipi tor": "Lipitor",
    "atorva": "Lipitor",
    "montelucast": "Singulair",
    "levo thyroxine": "Synthroid",
    "insulin glargeen": "Lantus",
    "pantaprazole": "Protonix",
    "augment in": "Augmentin",
    "cough syrup": "Benadryl Cough Formula",
    "vitamin d": "D-Rise",
    "insulin pen": "Lantus",
    "inhaler": "Ventolin",
}

# Ordinary call phrases that must NOT be read as a medicine name.
NON_MEDICINE_PHRASES = [
    "Hello, is anyone there?",
    "Are you open right now?",
    "I want to speak to someone",
    "Thanks very much, goodbye",
    "Can I collect it tomorrow morning?",
    "My name is Robert Wilson",
    "What is your address please",
    "How late are you open on Saturday",
    "I called yesterday about my order",
    "Sorry, could you repeat that",
    "Do you take insurance cards",
    "Yes that is correct thank you",
    "Is my order ready for pickup",
    "Can you deliver this evening",
    "What time do you close on Sunday",
    "Do you sell that tablet here",
    "I will collect it from the counter",
    "That's all for now",
    "Could you say that again please",
    "How much is the delivery charge",
]

INTENT_CASES = {
    # enquiry
    "Do you have amoxicillin in stock?": Intent.AVAILABILITY,
    "Is Ventolin available": Intent.AVAILABILITY,
    "Have you got any Zyrtec left": Intent.AVAILABILITY,
    "How much is panadol": Intent.PRICE,
    "What does Lipitor cost": Intent.PRICE,
    "Tell me about Ventolin": Intent.MEDICINE_INFO,
    "What is amoxicillin used for": Intent.MEDICINE_INFO,
    "I would like to know about Panadol": Intent.MEDICINE_INFO,
    # ordering
    "I would like to order two packs of panadol": Intent.ORDER,
    "Please reserve one Ventolin for me": Intent.ORDER,
    "I'll take two boxes": Intent.ORDER,
    "Can you hold it for me": Intent.ORDER,
    "I need a refill of my metformin": Intent.REFILL,
    "Is my order ready for pickup": Intent.ORDER_STATUS,
    # store
    "What time do you close on Sunday": Intent.STORE_INFO,
    "Where are you located": Intent.STORE_INFO,
    "Do you deliver to my home": Intent.DELIVERY,
    "Do I need a prescription for this": Intent.PRESCRIPTION,
    # safety
    "How many paracetamol should I take": Intent.CLINICAL,
    "What is the dosage for amoxicillin": Intent.CLINICAL,
    "Can I take ibuprofen with alcohol": Intent.CLINICAL,
    "Are there any side effects": Intent.CLINICAL,
    "Can I speak to the pharmacist": Intent.PHARMACIST,
    "Put me through to a human please": Intent.PHARMACIST,
    # conversation
    "Hello there": Intent.GREETING,
    "Good morning": Intent.GREETING,
    "That's all, goodbye": Intent.GOODBYE,
    "Bye": Intent.GOODBYE,
}

QUANTITY_CASES = {
    "I need two boxes of amoxicillin": 2,
    "give me 3 packs of panadol": 3,
    "reserve ten Ventolin inhalers": 10,
    "order five insulin pens": 5,
    "two bottles of cough syrup": 2,
    "half a dozen strips": 6,
    "twelve tablets": 12,
    "I would like to buy two packages": 2,
    "do you have panadol": None,
    "amoxicillin 500 mg": None,
}

PHONE_CASES = {
    "my number is 9876543210": "9876543210",
    "9 8 7 6 5 4 3 2 1 0": "9876543210",
    "nine eight seven six five four three two one zero": "9876543210",
    "98765 43210": "9876543210",
    "amoxicillin 500 mg, two boxes": None,
    "yes please": None,
}

NAME_CASES = {
    "My name is Rathan Vengala": "Rathan Vengala",
    "this is Priya": "Priya",
    "It's Rattan": "Rattan",
    "put it under the name Arun Kumar": "Arun Kumar",
    "S Kumar": "S Kumar",
}

# Transcripts Whisper produces from silence, and real speech it must not drop.
NOISE_CASES = {
    "ஆகும் குட்பை": True,
    "请不吝点赞 订阅 转发": True,
    "गुडबाई": True,
    "Thanks for watching!": True,
    "you": True,
    "[Music]": True,
    "": True,
    "Do you have amoxicillin in stock?": False,
    "goodbye": False,
    "9876543210": False,
    "1 2 3 4 5 6 7 8": False,
    "no": False,
    "OK": False,
}


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------


@dataclass
class Metric:
    name: str
    passed: int
    total: int
    failures: list[str] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return (self.passed / self.total * 100) if self.total else 0.0


def measure_search_recall() -> Metric:
    metric = Metric("Medicine search (mis-transcribed names)", 0, len(MISHEARD_NAMES))
    for spoken, expected in MISHEARD_NAMES.items():
        found = kb.find(spoken)
        if found is not None and found.brand_name == expected:
            metric.passed += 1
        else:
            got = found.brand_name if found else "no match"
            metric.failures.append(f"{spoken!r} -> {got} (expected {expected})")
    return metric


def measure_false_positive_rate() -> Metric:
    metric = Metric("Rejecting non-medicine phrases", 0, len(NON_MEDICINE_PHRASES))
    for phrase in NON_MEDICINE_PHRASES:
        hits = handle_query(phrase).medicines
        if not hits:
            metric.passed += 1
        else:
            metric.failures.append(f"{phrase!r} -> {hits[0].medicine.brand_name}")
    return metric


def measure_intent_accuracy() -> Metric:
    metric = Metric("Intent classification", 0, len(INTENT_CASES))
    for text, expected in INTENT_CASES.items():
        got = detect_intent(text)
        if got == expected:
            metric.passed += 1
        else:
            metric.failures.append(f"{text!r} -> {got.value} (expected {expected.value})")
    return metric


def measure_entity_extraction() -> Metric:
    cases = len(QUANTITY_CASES) + len(PHONE_CASES) + len(NAME_CASES)
    metric = Metric("Entity extraction (quantity, phone, name)", 0, cases)

    for text, expected in QUANTITY_CASES.items():
        got = extract_quantity(text)
        if got == expected:
            metric.passed += 1
        else:
            metric.failures.append(f"quantity {text!r} -> {got} (expected {expected})")

    for text, expected in PHONE_CASES.items():
        got = extract_phone(text)
        if got == expected:
            metric.passed += 1
        else:
            metric.failures.append(f"phone {text!r} -> {got} (expected {expected})")

    for text, expected in NAME_CASES.items():
        got = extract_person_name(text) or name_from_bare_reply(text)
        if got == expected:
            metric.passed += 1
        else:
            metric.failures.append(f"name {text!r} -> {got} (expected {expected})")

    return metric


def measure_noise_filter() -> Metric:
    metric = Metric("Speech-to-text noise filter", 0, len(NOISE_CASES))
    for text, expected in NOISE_CASES.items():
        if is_probable_noise(text) == expected:
            metric.passed += 1
        else:
            metric.failures.append(f"{text[:30]!r} -> {not expected}")
    return metric


def measure_safety_routing() -> Metric:
    """Clinical questions must never be answered by the agent."""
    clinical = [t for t, i in INTENT_CASES.items() if i in (Intent.CLINICAL, Intent.PHARMACIST)]
    metric = Metric("Clinical questions routed to pharmacist", 0, len(clinical))
    for text in clinical:
        if handle_query(text).needs_pharmacist:
            metric.passed += 1
        else:
            metric.failures.append(f"{text!r} was not routed to a pharmacist")
    return metric


def measure_order_workflow() -> Metric:
    """End-to-end order journeys, each asserting one workflow guarantee."""
    checks: list[tuple[str, bool]] = []
    tmp = Path(tempfile.mkdtemp())
    store = OrderStore(tmp / "orders.json")
    queue = EscalationQueue(tmp / "escalations.json")

    def converse(turns: list[str]) -> tuple[OrderDraft, object]:
        draft = OrderDraft(call_id="VALIDATION")
        flow = None
        for turn in turns:
            flow = update_draft(draft, handle_query(turn), store=store, queue=queue)
            build_order_context(flow)
        return draft, flow

    # 1. A complete pickup order is created.
    _, flow = converse(
        [
            "I would like to order two packs of Panadol",
            "My name is Rathan Vengala",
            "9876543210",
            "I'll collect it from the counter",
            "yes",
        ]
    )
    checks.append(("pickup order created", flow.order is not None))
    checks.append(
        ("customer details captured", flow.order is not None and flow.order.customer.phone == "9876543210")
    )

    # 2. Prescription-only orders are held, not confirmed.
    _, flow = converse(
        ["I want to order one pack of Amoxil", "Priya", "9123456789", "pickup", "yes"]
    )
    checks.append(
        ("prescription order held", flow.order is not None and flow.order.requires_prescription)
    )

    # 3. Stock the pharmacy does not have is refused and escalated.
    _, flow = converse(["Please reserve ten Ventolin inhalers"])
    checks.append(("over-ordering refused", flow.order is None and bool(flow.escalations)))

    # 4. Out-of-stock offers alternatives.
    _, flow = converse(["Please reserve two boxes of Augmentin"])
    checks.append(
        ("out of stock offers alternatives", bool(flow.stock_checks and flow.stock_checks[0].alternatives))
    )

    # 5. The caller can cancel at the confirmation step.
    draft, flow = converse(
        ["I want to order one pack of Panadol", "Arun", "9123456789", "pickup", "no thanks"]
    )
    checks.append(("cancellation honoured", flow.cancelled and draft.order_id is None))

    # 6. Reservations reduce what the next caller can take.
    panadol = kb.find("panadol")
    check = store.check_stock(panadol, 1)
    checks.append(("reservations reduce availability", check.reserved > 0))

    # 7. Escalations reach the pharmacist queue.
    checks.append(("escalations queued", queue.open_count > 0))

    metric = Metric("Order workflow (end to end)", sum(1 for _, ok in checks if ok), len(checks))
    metric.failures = [name for name, ok in checks if not ok]
    return metric


ALL_MEASURES = (
    measure_search_recall,
    measure_false_positive_rate,
    measure_intent_accuracy,
    measure_entity_extraction,
    measure_noise_filter,
    measure_safety_routing,
    measure_order_workflow,
)


def run_validation() -> list[Metric]:
    return [measure() for measure in ALL_MEASURES]


# --------------------------------------------------------------------------
# Threshold assertions (so pytest can run this file too)
# --------------------------------------------------------------------------


def test_search_recall_is_complete():
    metric = measure_search_recall()
    assert metric.accuracy == 100.0, metric.failures


def test_no_false_medicine_matches():
    metric = measure_false_positive_rate()
    assert metric.accuracy == 100.0, metric.failures


def test_intent_accuracy_above_threshold():
    metric = measure_intent_accuracy()
    assert metric.accuracy >= 95.0, metric.failures


def test_entity_extraction_above_threshold():
    metric = measure_entity_extraction()
    assert metric.accuracy >= 95.0, metric.failures


def test_noise_filter_is_correct():
    metric = measure_noise_filter()
    assert metric.accuracy == 100.0, metric.failures


def test_every_clinical_question_is_routed():
    metric = measure_safety_routing()
    assert metric.accuracy == 100.0, metric.failures


def test_order_workflow_guarantees_hold():
    metric = measure_order_workflow()
    assert metric.accuracy == 100.0, metric.failures


def main() -> None:
    print("=" * 78)
    print("MediTrack AI - MVP validation report")
    print("=" * 78)
    print(f"{'Capability':<46}{'Cases':>7}{'Passed':>8}{'Accuracy':>11}")
    print("-" * 78)

    metrics = run_validation()
    for metric in metrics:
        print(f"{metric.name:<46}{metric.total:>7}{metric.passed:>8}{metric.accuracy:>10.1f}%")

    total_cases = sum(m.total for m in metrics)
    total_passed = sum(m.passed for m in metrics)
    overall = total_passed / total_cases * 100 if total_cases else 0.0
    print("-" * 78)
    print(f"{'OVERALL':<46}{total_cases:>7}{total_passed:>8}{overall:>10.1f}%")

    failures = [(m.name, f) for m in metrics for f in m.failures]
    if failures:
        print("\nFailures:")
        for name, detail in failures:
            print(f"  [{name}] {detail}")
    print()
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
