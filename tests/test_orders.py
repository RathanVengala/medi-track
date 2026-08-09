"""
Tests for the Week 3 order workflow: customer capture, inventory checking,
order records and pharmacist escalation.

Every test uses a temporary store, so running these never touches
data/orders.json or data/escalations.json. No API key or network needed.

    python -m tests.test_orders
    python -m pytest tests/test_orders.py -q
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from app.escalation import EscalationQueue, EscalationReason, EscalationStatus, Priority
from app.knowledge_base import get_knowledge_base
from app.order_flow import (
    OrderDraft,
    Slot,
    build_order_context,
    fallback_order_reply,
    should_start_order,
    update_draft,
)
from app.orders import (
    Customer,
    Fulfilment,
    OrderError,
    OrderItem,
    OrderStatus,
    OrderStore,
)
from app.query_handler import (
    Intent,
    detect_intent,
    extract_confirmation,
    extract_fulfilment,
    extract_person_name,
    extract_phone,
    extract_quantity,
    handle_query,
)

kb = get_knowledge_base()


def fresh():
    """A store and queue backed by throwaway files."""
    tmp = Path(tempfile.mkdtemp())
    return OrderStore(tmp / "orders.json"), EscalationQueue(tmp / "escalations.json")


def converse(turns, store=None, queue=None, call_id="TEST"):
    """Feed turns through the order flow and return (draft, last flow)."""
    store = store or fresh()[0]
    queue = queue or fresh()[1]
    draft = OrderDraft(call_id=call_id)
    flow = None
    for turn in turns:
        result = handle_query(turn)
        flow = update_draft(draft, result, store=store, queue=queue)
        build_order_context(flow)  # sets draft.asked_for, as the live path does
    return draft, flow


# --- customer information capture -------------------------------------------


def test_extract_person_name():
    assert extract_person_name("My name is Rathan Vengala") == "Rathan Vengala"
    assert extract_person_name("this is Priya") == "Priya"
    assert extract_person_name("put it under the name Arun Kumar") == "Arun Kumar"
    assert extract_person_name("do you have panadol") is None


def test_extract_phone():
    assert extract_phone("my number is 9876543210") == "9876543210"
    assert extract_phone("9 8 7 6 5 4 3 2 1 0") == "9876543210"
    assert extract_phone("nine eight seven six five four three two one zero") == "9876543210"
    # Strengths and quantities must never be mistaken for a phone number.
    assert extract_phone("amoxicillin 500 mg, two boxes") is None


def test_phone_captured_from_a_bare_reply():
    """
    Answering "what's your number?" with just digits must work, including the
    short/partial numbers Whisper produces from fast dictation.
    """
    store, queue = fresh()
    draft, _ = converse(
        [
            "I want to order one pack of Panadol",
            "It's Rathan",
            "123456789",
        ],
        store,
        queue,
    )
    assert draft.name == "Rathan"
    assert draft.phone == "123456789"
    assert draft.next_slot() == Slot.FULFILMENT


def test_spaced_out_digits_are_captured():
    store, queue = fresh()
    draft, _ = converse(
        ["I want to order one pack of Panadol", "Arun", "9 8 7 6 5 4 3 2 1 0"],
        store,
        queue,
    )
    assert draft.phone == "9876543210"


def test_extract_fulfilment():
    assert extract_fulfilment("I'll collect it from the counter") == "pickup"
    assert extract_fulfilment("can you deliver it") == "delivery"
    assert extract_fulfilment("send it to my home") == "delivery"
    assert extract_fulfilment("I want panadol") is None


def test_extract_confirmation():
    assert extract_confirmation("yes please") is True
    assert extract_confirmation("no, cancel that") is False
    assert extract_confirmation("what time do you close") is None


def test_quantity_units_cover_every_dispensing_form():
    """A missing unit silently becomes an order for one pack."""
    assert extract_quantity("reserve ten Ventolin inhalers") == 10
    assert extract_quantity("order five insulin pens") == 5
    assert extract_quantity("two bottles of cough syrup") == 2
    assert extract_quantity("half a dozen strips") == 6


def test_order_intent_detection():
    assert detect_intent("I would like to order two packs of panadol") == Intent.ORDER
    assert detect_intent("please reserve one for me") == Intent.ORDER
    assert detect_intent("I'll take two") == Intent.ORDER
    assert detect_intent("can I speak to the pharmacist") == Intent.PHARMACIST
    # An enquiry is still an enquiry.
    assert detect_intent("do you have panadol in stock") == Intent.AVAILABILITY


# --- inventory checking ------------------------------------------------------


def test_stock_check_reflects_reservations():
    store, _ = fresh()
    ventolin = kb.find("ventolin")  # 4 on hand
    assert store.check_stock(ventolin, 4).can_fulfil

    store.create_order(
        Customer("A", "9000000000"), [OrderItem.from_medicine(ventolin, 3)]
    )
    after = store.check_stock(ventolin, 4)
    assert not after.can_fulfil
    assert after.available == 1
    assert after.reserved == 3
    assert after.shortfall == 3


def test_medicines_file_is_never_mutated():
    """On-hand stock is the pharmacy's record; reservations live in orders."""
    store, _ = fresh()
    panadol = kb.find("panadol")
    before = panadol.stock_qty
    store.create_order(Customer("A", "9000000000"), [OrderItem.from_medicine(panadol, 5)])
    assert kb.find("panadol").stock_qty == before


def test_overselling_is_blocked():
    store, _ = fresh()
    ventolin = kb.find("ventolin")
    store.create_order(Customer("A", "9000000000"), [OrderItem.from_medicine(ventolin, 3)])
    try:
        store.create_order(Customer("B", "9000000001"), [OrderItem.from_medicine(ventolin, 2)])
    except OrderError as exc:
        assert "available" in str(exc)
    else:
        raise AssertionError("second order should not have been accepted")


def test_cancelled_orders_release_their_stock():
    store, _ = fresh()
    ventolin = kb.find("ventolin")
    order = store.create_order(
        Customer("A", "9000000000"), [OrderItem.from_medicine(ventolin, 4)]
    )
    assert store.check_stock(ventolin, 1).available == 0
    store.cancel(order.id, "caller changed their mind")
    assert store.check_stock(ventolin, 4).can_fulfil


# --- order records -----------------------------------------------------------


def test_order_totals_and_persistence():
    store, _ = fresh()
    panadol = kb.find("panadol")
    order = store.create_order(
        Customer("Rathan", "9876543210"), [OrderItem.from_medicine(panadol, 2)]
    )
    assert order.total == round(panadol.price * 2, 2)
    assert order.status == OrderStatus.CONFIRMED.value

    # Survives a reload from disk.
    reloaded = OrderStore(store.path)
    assert reloaded.get(order.id) is not None
    assert reloaded.get(order.id).customer.name == "Rathan"


def test_delivery_fee_applied_below_the_free_threshold():
    store, _ = fresh()
    cheap = kb.find("electral")  # well under the free-delivery threshold
    order = store.create_order(
        Customer("A", "9000000000", Fulfilment.DELIVERY.value, "12 MG Road"),
        [OrderItem.from_medicine(cheap, 1)],
    )
    assert order.delivery_fee > 0
    assert order.total == round(order.subtotal + order.delivery_fee, 2)


def test_prescription_order_is_held_not_confirmed():
    store, _ = fresh()
    amoxil = kb.find("amoxil")
    order = store.create_order(Customer("A", "9000000000"), [OrderItem.from_medicine(amoxil, 1)])
    assert order.requires_prescription
    assert order.status == OrderStatus.AWAITING_PRESCRIPTION.value


def test_order_ids_are_short_enough_to_read_aloud():
    store, _ = fresh()
    panadol = kb.find("panadol")
    order = store.create_order(Customer("A", "9000000000"), [OrderItem.from_medicine(panadol, 1)])
    assert order.id.startswith("ORD-")
    assert len(order.id.split("-")[-1]) == 4


# --- the conversation flow ---------------------------------------------------


def test_full_order_conversation_creates_a_record():
    store, queue = fresh()
    draft, flow = converse(
        [
            "I would like to order two packs of Panadol",
            "My name is Rathan Vengala",
            "My number is 9876543210",
            "I'll collect it from the counter",
            "Yes, please place the order",
        ],
        store,
        queue,
    )
    assert flow.order is not None
    order = flow.order
    assert order.customer.name == "Rathan Vengala"
    assert order.customer.phone == "9876543210"
    assert order.customer.fulfilment == Fulfilment.PICKUP.value
    assert order.items[0].quantity == 2
    assert store.get(order.id) is not None


def test_flow_asks_for_one_detail_at_a_time():
    store, queue = fresh()
    draft = OrderDraft(call_id="T")
    expected = [Slot.NAME, Slot.PHONE, Slot.FULFILMENT, Slot.CONFIRM]
    turns = [
        "I want to order one pack of Panadol",
        "My name is Priya",
        "9123456789",
        "pickup please",
    ]
    for turn, slot in zip(turns, expected):
        flow = update_draft(draft, handle_query(turn), store=store, queue=queue)
        context = build_order_context(flow)
        assert draft.next_slot() == slot, f"after {turn!r}: {draft.next_slot()}"
        assert "NEXT:" in context


def test_delivery_branch_asks_for_an_address():
    store, queue = fresh()
    draft, _ = converse(
        [
            "I want to order one pack of Panadol",
            "My name is Arun",
            "9123456789",
            "please deliver it",
        ],
        store,
        queue,
    )
    assert draft.next_slot() == Slot.ADDRESS


def test_caller_can_cancel_at_confirmation():
    store, queue = fresh()
    draft, flow = converse(
        [
            "I want to order one pack of Panadol",
            "My name is Arun",
            "9123456789",
            "pickup",
            "no, cancel that",
        ],
        store,
        queue,
    )
    assert flow.cancelled
    assert draft.order_id is None
    assert store.list_orders() == []


def test_medicine_carries_over_from_the_previous_turn():
    """'I'd like to order two packs' after asking about Panadol."""
    enquiry = handle_query("Do you have Panadol in stock?")
    order_turn = handle_query("Great, I would like to order two packs please")
    assert not order_turn.medicines  # no name in this sentence
    assert should_start_order(order_turn, None, enquiry.medicines)

    store, queue = fresh()
    draft = OrderDraft(call_id="T")
    flow = update_draft(
        draft, order_turn, store=store, queue=queue, carried_medicines=enquiry.medicines
    )
    assert draft.items and draft.items[0].brand_name == "Panadol"
    assert draft.items[0].quantity == 2


def test_enquiry_alone_does_not_start_an_order():
    assert not should_start_order(handle_query("Do you have Panadol?"), None, None)
    assert not should_start_order(handle_query("How much is Panadol?"), None, None)


# --- pharmacist escalation ---------------------------------------------------


def test_insufficient_stock_escalates():
    store, queue = fresh()
    draft, flow = converse(["Please reserve ten Ventolin inhalers"], store, queue)
    assert flow.escalations
    assert draft.order_id is None
    items = queue.list_escalations(reason=EscalationReason.INSUFFICIENT_STOCK.value)
    assert items and items[0].status == EscalationStatus.OPEN.value


def test_out_of_stock_escalates_with_alternatives():
    store, queue = fresh()
    _, flow = converse(["Please reserve two boxes of Augmentin"], store, queue)
    assert flow.escalations
    check = flow.stock_checks[0]
    assert check.available == 0
    assert "Amoxil" in check.message


def test_stock_problem_escalates_only_once_per_call():
    store, queue = fresh()
    converse(
        ["Please reserve ten Ventolin inhalers", "My name is Arun", "9123456789"],
        store,
        queue,
    )
    assert len(queue.list_escalations()) == 1


def test_prescription_order_raises_an_escalation():
    store, queue = fresh()
    _, flow = converse(
        [
            "I want to order one pack of Amoxil",
            "My name is Priya",
            "9123456789",
            "pickup",
            "yes",
        ],
        store,
        queue,
    )
    assert flow.order is not None
    assert flow.order.escalation_ids
    reasons = {e.reason for e in queue.list_escalations()}
    assert EscalationReason.PRESCRIPTION_REQUIRED.value in reasons


def test_escalation_priority_ordering():
    _, queue = fresh()
    queue.raise_escalation(EscalationReason.OUT_OF_STOCK, "low priority item")
    queue.raise_escalation(EscalationReason.CLINICAL_QUESTION, "dosage question")
    items = queue.list_escalations()
    assert items[0].priority == Priority.HIGH.value
    assert items[0].reason == EscalationReason.CLINICAL_QUESTION.value


def test_escalations_persist_and_can_be_resolved():
    _, queue = fresh()
    item = queue.raise_escalation(EscalationReason.CLINICAL_QUESTION, "dosage question")
    assert queue.open_count == 1
    queue.update_status(item.id, EscalationStatus.RESOLVED, "pharmacist called back")
    assert queue.open_count == 0
    reloaded = EscalationQueue(queue.path)
    assert reloaded.get(item.id).status == EscalationStatus.RESOLVED.value


def test_fallback_replies_are_speakable():
    store, queue = fresh()
    draft = OrderDraft(call_id="T")
    for turn in ("I want to order one pack of Panadol", "My name is Arun", "9123456789"):
        flow = update_draft(draft, handle_query(turn), store=store, queue=queue)
        build_order_context(flow)
        reply = fallback_order_reply(flow)
        assert reply and "\n" not in reply and len(reply) < 320


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {test.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
