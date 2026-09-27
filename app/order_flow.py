"""
Order-taking conversation flow (Week 3).

A phone order needs several facts the caller supplies across many turns, in
whatever order they feel like. This module keeps a per-call draft, decides
which detail is still missing, and hands the LLM an explicit instruction about
what to ask next - so the wording stays natural while the sequence stays
deterministic.

    Request -> Availability Check -> Order Creation -> Confirmation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

from app.escalation import (
    SPOKEN_REASON,
    EscalationQueue,
    EscalationReason,
    get_escalation_queue,
)
from app.knowledge_base import PharmacyKnowledgeBase, get_knowledge_base
from app.orders import (
    Customer,
    Fulfilment,
    InvalidQuantity,
    Order,
    OrderError,
    OrderItem,
    OrderStore,
    StockCheck,
    get_order_store,
    validate_quantity,
)
from app.query_handler import (
    QUANTITY_ISSUE_NOTE,
    QUANTITY_ISSUE_REPLY,
    Intent,
    QuantityIssue,
    QueryResult,
    extract_address,
    name_from_bare_reply,
    phone_from_bare_reply,
)

logger = logging.getLogger("meditrack.order_flow")


class Slot(str, Enum):
    """The next detail the agent needs from the caller."""

    MEDICINE = "medicine"
    QUANTITY = "quantity"
    NAME = "name"
    PHONE = "phone"
    FULFILMENT = "fulfilment"
    ADDRESS = "address"
    CONFIRM = "confirm"
    DONE = "done"


# What the agent should ask for each outstanding slot.
SLOT_PROMPTS = {
    Slot.MEDICINE: "Ask which medicine the caller would like to order.",
    Slot.QUANTITY: "Ask how many packs they would like.",
    Slot.NAME: "Ask for the name to put the order under.",
    Slot.PHONE: "Ask for a contact phone number.",
    Slot.FULFILMENT: "Ask whether they want to collect it from the counter or have it delivered.",
    Slot.ADDRESS: "Ask for the delivery address.",
    Slot.CONFIRM: "Read the order summary back and ask them to confirm.",
}


@dataclass
class DraftItem:
    medicine_id: str
    brand_name: str
    quantity: int = 1
    stock_ok: bool = True
    stock_message: str = ""
    # True when the caller did say a pack count but it had to be refused
    # (negative, zero, or over the limit). The default of one pack is then NOT
    # their answer, and the flow must ask again before going any further.
    needs_quantity: bool = False


@dataclass
class OrderDraft:
    """A partially collected order, held for the duration of one call."""

    call_id: str = ""
    items: list[DraftItem] = field(default_factory=list)
    name: str | None = None
    phone: str | None = None
    fulfilment: str | None = None
    address: str | None = None
    asked_for: Slot | None = None
    confirmed: bool = False
    cancelled: bool = False
    order_id: str | None = None
    escalation_ids: list[str] = field(default_factory=list)
    # Medicines already escalated on this call, so a stock problem is queued
    # once rather than on every subsequent turn.
    escalated_medicines: list[str] = field(default_factory=list)

    @property
    def is_active(self) -> bool:
        return bool(self.items) and not self.confirmed and not self.cancelled

    @property
    def is_complete(self) -> bool:
        return self.next_slot() == Slot.DONE

    def find_item(self, medicine_id: str) -> DraftItem | None:
        return next((i for i in self.items if i.medicine_id == medicine_id), None)

    def next_slot(self) -> Slot:
        """First missing detail, in the order a person would ask for them."""
        if not self.items:
            return Slot.MEDICINE
        if any(item.needs_quantity for item in self.items):
            return Slot.QUANTITY
        if self.name is None:
            return Slot.NAME
        if self.phone is None:
            return Slot.PHONE
        if self.fulfilment is None:
            return Slot.FULFILMENT
        if self.fulfilment == Fulfilment.DELIVERY.value and not self.address:
            return Slot.ADDRESS
        if not self.confirmed:
            return Slot.CONFIRM
        return Slot.DONE

    def summary(self, kb: PharmacyKnowledgeBase | None = None) -> list[str]:
        kb = kb or get_knowledge_base()
        lines: list[str] = []
        subtotal = 0.0
        for item in self.items:
            medicine = kb.get(item.medicine_id)
            if medicine is None:
                continue
            line_total = medicine.price * item.quantity
            subtotal += line_total
            lines.append(
                f"  {item.quantity} x {medicine.brand_name} ({medicine.generic_name} "
                f"{medicine.strength}, {medicine.pack_size}) = {kb.format_price(line_total)}"
            )
        lines.append(f"  Subtotal: {kb.format_price(subtotal)}")
        if self.name:
            lines.append(f"  Name: {self.name}")
        if self.phone:
            lines.append(f"  Phone: {self.phone}")
        if self.fulfilment:
            lines.append(f"  Fulfilment: {self.fulfilment}")
        if self.address:
            lines.append(f"  Address: {self.address}")
        return lines


@dataclass
class FlowResult:
    """What the flow did with this turn, rendered into the LLM context."""

    draft: OrderDraft
    started: bool = False
    order: Order | None = None
    stock_checks: list[StockCheck] = field(default_factory=list)
    escalations: list[str] = field(default_factory=list)
    error: str | None = None
    cancelled: bool = False
    # Set when this turn contained a pack count that had to be refused, so the
    # agent re-asks for the quantity instead of assuming one pack.
    quantity_issue: QuantityIssue | None = None


# Intents that mean "I want to buy this", as opposed to just asking about it.
_ORDER_INTENTS = {Intent.ORDER, Intent.REFILL}


def wants_to_order(result: QueryResult) -> bool:
    """True when the caller is asking to buy, not just asking about it."""
    return result.intent in _ORDER_INTENTS or result.confirmation is True


def should_start_order(
    result: QueryResult,
    draft: OrderDraft | None,
    carried_medicines: list | None = None,
) -> bool:
    """
    Decide whether this turn is the start of an order.

    `carried_medicines` is what was being discussed on earlier turns, so
    "I'd like to order two packs" works when the name was said a turn ago.
    """
    if draft is not None and draft.is_active:
        return True
    medicines = result.medicines or (carried_medicines or [])
    if not medicines:
        return False
    if result.intent in _ORDER_INTENTS:
        return True
    if result.confirmation is True and carried_medicines:
        return True
    # "I need two boxes of X" - a quantity turns an enquiry into a purchase.
    if result.intent == Intent.AVAILABILITY and result.quantity:
        return True
    return False


def update_draft(
    draft: OrderDraft,
    result: QueryResult,
    kb: PharmacyKnowledgeBase | None = None,
    store: OrderStore | None = None,
    queue: EscalationQueue | None = None,
    carried_medicines: list | None = None,
) -> FlowResult:
    """
    Fold one customer turn into the draft, then create the order once every
    detail is present and the caller has confirmed.
    """
    kb = kb or get_knowledge_base()
    store = store or get_order_store()
    queue = queue or get_escalation_queue()

    flow = FlowResult(draft=draft, quantity_issue=result.quantity_issue)
    asked = draft.asked_for

    # Only fall back to an earlier turn's medicine while the basket is empty,
    # so a later "yes" cannot silently re-add an item.
    mentioned = result.medicines
    if not mentioned and not draft.items and carried_medicines:
        mentioned = carried_medicines

    # --- cancellation ---------------------------------------------------
    if result.confirmation is False and asked == Slot.CONFIRM:
        draft.cancelled = True
        flow.cancelled = True
        logger.info("Order draft cancelled by caller on call %s", draft.call_id)
        return flow

    # --- medicines and quantities ---------------------------------------
    for extracted in mentioned:
        medicine = extracted.medicine
        existing = draft.find_item(medicine.id)
        if existing:
            if result.quantity:
                existing.quantity = result.quantity
                existing.needs_quantity = False
            elif result.quantity_issue:
                existing.needs_quantity = True
        else:
            draft.items.append(
                DraftItem(
                    medicine_id=medicine.id,
                    brand_name=medicine.brand_name,
                    quantity=result.quantity or 1,
                    # "999 packs of Panadol" must not quietly become one pack.
                    needs_quantity=result.quantity_issue is not None,
                )
            )
            flow.started = True

    # A bare number answers "how many packs?".
    if not mentioned and draft.items:
        if result.quantity:
            draft.items[-1].quantity = result.quantity
            draft.items[-1].needs_quantity = False
        elif result.quantity_issue and asked == Slot.QUANTITY:
            draft.items[-1].needs_quantity = True

    # --- customer details ------------------------------------------------
    if result.person_name:
        draft.name = result.person_name
    elif asked == Slot.NAME:
        draft.name = name_from_bare_reply(result.text)

    if result.phone:
        draft.phone = result.phone
    elif asked == Slot.PHONE:
        draft.phone = phone_from_bare_reply(result.text)

    if result.fulfilment:
        draft.fulfilment = result.fulfilment
    elif asked == Slot.FULFILMENT and result.confirmation is not None:
        # "Yes" to "shall we deliver it?" - treat as delivery.
        draft.fulfilment = Fulfilment.DELIVERY.value if result.confirmation else Fulfilment.PICKUP.value

    if draft.fulfilment == Fulfilment.DELIVERY.value and not draft.address:
        address = extract_address(result.text)
        if address and asked in (Slot.ADDRESS, Slot.FULFILMENT):
            draft.address = address

    # --- inventory check -------------------------------------------------
    flow.stock_checks = _check_stock(draft, kb, store)
    for check in flow.stock_checks:
        item = draft.find_item(check.medicine.id)
        if item:
            item.stock_ok = check.can_fulfil
            item.stock_message = check.message

    blocked = [c for c in flow.stock_checks if not c.can_fulfil]
    if blocked:
        for check in blocked:
            if check.medicine.id in draft.escalated_medicines:
                continue
            draft.escalated_medicines.append(check.medicine.id)
            reason = (
                EscalationReason.OUT_OF_STOCK
                if check.available <= 0
                else EscalationReason.INSUFFICIENT_STOCK
            )
            escalation = queue.raise_escalation(
                reason=reason,
                summary=check.message,
                call_id=draft.call_id,
                customer_name=draft.name,
                customer_phone=draft.phone,
                transcript=result.text,
            )
            draft.escalation_ids.append(escalation.id)
            flow.escalations.append(escalation.id)
        # Nothing can be reserved, so stop collecting details.
        return flow

    # --- confirmation ----------------------------------------------------
    if asked == Slot.CONFIRM and result.confirmation is True:
        draft.confirmed = True

    if draft.confirmed and draft.order_id is None:
        flow.order = _create_order(draft, kb, store, queue, result)
        if flow.order is None:
            draft.confirmed = False
            flow.error = "Stock changed before the order could be created."

    return flow


def _check_stock(
    draft: OrderDraft, kb: PharmacyKnowledgeBase, store: OrderStore
) -> list[StockCheck]:
    checks: list[StockCheck] = []
    for item in draft.items:
        medicine = kb.get(item.medicine_id)
        if medicine is None:
            continue
        checks.append(store.check_stock(medicine, item.quantity, kb=kb))
    return checks


def _create_order(
    draft: OrderDraft,
    kb: PharmacyKnowledgeBase,
    store: OrderStore,
    queue: EscalationQueue,
    result: QueryResult,
) -> Order | None:
    items: list[OrderItem] = []
    for draft_item in draft.items:
        medicine = kb.get(draft_item.medicine_id)
        if medicine is None:
            continue
        # Belt and braces: next_slot() already stops the flow reaching
        # confirmation with an unanswered quantity, but the order store is the
        # last place a bad count can still be caught before it is written.
        try:
            quantity = validate_quantity(draft_item.quantity)
        except InvalidQuantity as exc:
            logger.error("Refusing to build an order item: %s", exc)
            return None
        items.append(OrderItem.from_medicine(medicine, quantity))

    customer = Customer(
        name=draft.name or "Unnamed caller",
        phone=draft.phone or "",
        fulfilment=draft.fulfilment or Fulfilment.PICKUP.value,
        address=draft.address,
    )

    try:
        order = store.create_order(customer=customer, items=items, call_id=draft.call_id, kb=kb)
    except OrderError as exc:
        logger.error("Order creation failed: %s", exc)
        return None

    draft.order_id = order.id

    # Prescription-only and refrigerated items always need a pharmacist.
    if order.requires_prescription:
        escalation = queue.raise_escalation(
            reason=EscalationReason.PRESCRIPTION_REQUIRED,
            summary=(
                "Prescription needed for "
                + ", ".join(i.brand_name for i in order.items if i.prescription_required)
            ),
            call_id=draft.call_id,
            customer_name=customer.name,
            customer_phone=customer.phone,
            order_id=order.id,
            transcript=result.text,
        )
        store.attach_escalation(order.id, escalation.id)
        draft.escalation_ids.append(escalation.id)

    for item in order.items:
        medicine = kb.get(item.medicine_id)
        if medicine and "refrigerat" in medicine.storage.lower():
            escalation = queue.raise_escalation(
                reason=EscalationReason.COLD_CHAIN,
                summary=f"{medicine.brand_name} is a cold chain item and needs pharmacist handling",
                call_id=draft.call_id,
                customer_name=customer.name,
                customer_phone=customer.phone,
                order_id=order.id,
            )
            store.attach_escalation(order.id, escalation.id)
            draft.escalation_ids.append(escalation.id)

    return order


# --------------------------------------------------------------------------
# Rendering for the LLM
# --------------------------------------------------------------------------


def build_order_context(flow: FlowResult, kb: PharmacyKnowledgeBase | None = None) -> str:
    """Render the order state as an instruction block for the LLM."""
    kb = kb or get_knowledge_base()
    draft = flow.draft
    lines = ["ORDER WORKFLOW STATE (follow this exactly; it drives what you say next)"]

    if flow.cancelled:
        lines.append("- The caller cancelled the order. Confirm it is cancelled and offer more help.")
        return "\n".join(lines)

    if flow.order is not None:
        order = flow.order
        lines.append(f"- ORDER CREATED. Order number: {order.id}")
        lines.append(
            "- Read the order number back as its four digits only, one at a time "
            f"(say \"{' '.join(order.id.split('-')[-1])}\"). Do not spell out the ORD prefix."
        )
        for item in order.items:
            lines.append(
                f"  {item.quantity} x {item.brand_name} {item.strength} "
                f"({item.pack_size}) = {kb.format_price(item.line_total)}"
            )
        if order.delivery_fee:
            lines.append(f"  Delivery fee: {kb.format_price(order.delivery_fee)}")
        lines.append(f"  Total payable: {kb.format_price(order.total)}")
        lines.append(f"  Status: {order.status}")
        lines.append(f"  For: {order.customer.name}, {order.customer.phone}, {order.customer.fulfilment}")
        if order.customer.address:
            lines.append(f"  Address: {order.customer.address}")
        if order.requires_prescription:
            lines.append(
                "- This order needs a valid prescription before we can dispense it. Tell the "
                "caller a pharmacist will confirm once the prescription is received."
            )
        if order.escalation_ids:
            lines.append(f"  Pharmacist follow-up raised: {', '.join(order.escalation_ids)}")
        return "\n".join(lines)

    if flow.error:
        lines.append(f"- Order could not be created: {flow.error}. Apologise and offer a pharmacist callback.")
        return "\n".join(lines)

    blocked = [c for c in flow.stock_checks if not c.can_fulfil]
    if blocked:
        lines.append("- Cannot reserve the requested quantity:")
        for check in blocked:
            lines.append(f"  {check.message}")
        lines.append(
            "- Tell the caller what is available, offer the listed alternatives, and say a "
            "pharmacist will follow up."
        )
        return "\n".join(lines)

    if draft.items:
        lines.append("- Order in progress:")
        lines.extend(draft.summary(kb))

    if flow.quantity_issue:
        lines.append(f"- {QUANTITY_ISSUE_NOTE[flow.quantity_issue]}")

    next_slot = draft.next_slot()
    draft.asked_for = next_slot if next_slot != Slot.DONE else None
    if next_slot in SLOT_PROMPTS:
        lines.append(f"- NEXT: {SLOT_PROMPTS[next_slot]}")
        lines.append("- Ask for one detail at a time. Do not invent details the caller has not given.")
    return "\n".join(lines)


def fallback_order_reply(flow: FlowResult, kb: PharmacyKnowledgeBase | None = None) -> str | None:
    """Deterministic reply for the order flow when no LLM is available."""
    kb = kb or get_knowledge_base()
    draft = flow.draft

    if flow.cancelled:
        return "No problem, I've cancelled that. Is there anything else I can help with?"

    if flow.order is not None:
        order = flow.order
        parts = [
            f"Your order number is {order.id}.",
            f"The total is {kb.format_price(order.total)}.",
        ]
        if order.requires_prescription:
            parts.append("We'll need a valid prescription before we can dispense it.")
        return " ".join(parts)

    blocked = [c for c in flow.stock_checks if not c.can_fulfil]
    if blocked:
        return blocked[0].message + " A pharmacist will follow up with you."

    if flow.quantity_issue:
        return QUANTITY_ISSUE_REPLY[flow.quantity_issue]

    next_slot = draft.next_slot()
    prompts = {
        Slot.MEDICINE: "Which medicine would you like to order?",
        Slot.QUANTITY: "How many packs would you like?",
        Slot.NAME: "Can I take the name for the order?",
        Slot.PHONE: "And a contact number, please?",
        Slot.FULFILMENT: "Would you like to collect it, or shall we deliver?",
        Slot.ADDRESS: "What's the delivery address?",
        Slot.CONFIRM: _confirm_sentence(draft, kb),
    }
    return prompts.get(next_slot)


def _confirm_sentence(draft: OrderDraft, kb: PharmacyKnowledgeBase) -> str:
    items = ", ".join(f"{i.quantity} {i.brand_name}" for i in draft.items)
    total = sum(
        (kb.get(i.medicine_id).price * i.quantity) for i in draft.items if kb.get(i.medicine_id)
    )
    return (
        f"So that's {items} for {draft.name}, {kb.format_price(total)}. "
        "Shall I place the order?"
    )


# What the caller should be told for each status.
_STATUS_MEANING = {
    "confirmed": "confirmed and being prepared",
    "awaiting_prescription": "waiting for a valid prescription before we can dispense it",
    "escalated": "with the pharmacist for review",
    "ready_for_pickup": "ready to collect at the counter",
    "out_for_delivery": "out for delivery",
    "completed": "completed",
    "cancelled": "cancelled",
}


def build_status_context(
    result: QueryResult,
    store: OrderStore | None = None,
    kb: PharmacyKnowledgeBase | None = None,
) -> str:
    """
    Answer "is my order ready?" from the order records.

    Looks the order up by the number the caller reads back, falling back to
    their phone number.
    """
    store = store or get_order_store()
    kb = kb or get_knowledge_base()

    order = None
    if result.order_reference:
        order = store.get(result.order_reference)
    if order is None and result.phone:
        matches = store.list_orders(phone=result.phone)
        order = matches[0] if matches else None

    lines = ["ORDER STATUS LOOKUP"]
    if order is None:
        lines.append("- No matching order found.")
        lines.append(
            "- Ask the caller for their order number or the phone number the order was "
            "placed under. Do not invent an order."
        )
        return "\n".join(lines)

    lines.append(f"- Order {order.id} for {order.customer.name}")
    for item in order.items:
        lines.append(f"  {item.quantity} x {item.brand_name} {item.strength}")
    lines.append(f"  Total: {kb.format_price(order.total)}")
    lines.append(f"  Status: {order.status} - {_STATUS_MEANING.get(order.status, order.status)}")
    lines.append(f"  Fulfilment: {order.customer.fulfilment}")
    if order.status == "awaiting_prescription":
        lines.append("- Remind the caller we need a valid prescription before dispensing.")
    lines.append("- Tell the caller the status in one short sentence.")
    return "\n".join(lines)


def fallback_status_reply(
    result: QueryResult,
    store: OrderStore | None = None,
    kb: PharmacyKnowledgeBase | None = None,
) -> str:
    store = store or get_order_store()
    kb = kb or get_knowledge_base()
    order = None
    if result.order_reference:
        order = store.get(result.order_reference)
    if order is None and result.phone:
        matches = store.list_orders(phone=result.phone)
        order = matches[0] if matches else None
    if order is None:
        return "I couldn't find that order. Could you give me the order number or the phone number it was placed under?"
    return (
        f"Order {order.id} for {order.customer.name} is "
        f"{_STATUS_MEANING.get(order.status, order.status)}."
    )


def escalation_context(reason: EscalationReason, escalation_id: str) -> str:
    """Context block for an escalation raised outside the order flow."""
    return (
        "PHARMACIST ESCALATION RAISED\n"
        f"- Reference: {escalation_id}\n"
        f"- Reason: {SPOKEN_REASON.get(reason, 'a pharmacist needs to help')}\n"
        "- Tell the caller you are passing this to the pharmacist on duty, and that they "
        "will follow up. Do not attempt to answer it yourself."
    )
