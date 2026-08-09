"""
Order records and inventory checking (Week 3).

Turns a confirmed enquiry into a persisted order, and answers the question
"can we actually supply this?" by comparing the requested quantity against
stock on hand minus everything already reserved by open orders.

Stock in data/medicines.json is never mutated - it is the on-hand count from
the pharmacy's own system. Reservations live in data/orders.json, so
availability is always `on_hand - reserved` and the formulary stays the single
source of truth for what the shelf holds.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

from app.knowledge_base import DATA_DIR, Medicine, PharmacyKnowledgeBase, get_knowledge_base

logger = logging.getLogger("meditrack.orders")

ORDERS_FILE = DATA_DIR / "orders.json"

# Order numbers start here so the first one reads as "one zero zero one" on the
# phone rather than a string of leading zeros.
_SEQUENCE_START = 1000


class OrderStatus(str, Enum):
    DRAFT = "draft"
    CONFIRMED = "confirmed"
    AWAITING_PRESCRIPTION = "awaiting_prescription"
    ESCALATED = "escalated"
    READY_FOR_PICKUP = "ready_for_pickup"
    OUT_FOR_DELIVERY = "out_for_delivery"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


# Statuses that still hold stock back from other customers.
OPEN_STATUSES = frozenset(
    {
        OrderStatus.DRAFT,
        OrderStatus.CONFIRMED,
        OrderStatus.AWAITING_PRESCRIPTION,
        OrderStatus.ESCALATED,
        OrderStatus.READY_FOR_PICKUP,
        OrderStatus.OUT_FOR_DELIVERY,
    }
)


class Fulfilment(str, Enum):
    PICKUP = "pickup"
    DELIVERY = "delivery"


class OrderError(Exception):
    pass


@dataclass
class Customer:
    name: str
    phone: str
    fulfilment: str = Fulfilment.PICKUP.value
    address: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OrderItem:
    medicine_id: str
    brand_name: str
    generic_name: str
    strength: str
    pack_size: str
    quantity: int
    unit_price: float
    prescription_required: bool

    @property
    def line_total(self) -> float:
        return round(self.unit_price * self.quantity, 2)

    @classmethod
    def from_medicine(cls, medicine: Medicine, quantity: int) -> "OrderItem":
        return cls(
            medicine_id=medicine.id,
            brand_name=medicine.brand_name,
            generic_name=medicine.generic_name,
            strength=medicine.strength,
            pack_size=medicine.pack_size,
            quantity=quantity,
            unit_price=medicine.price,
            prescription_required=medicine.prescription_required,
        )

    def to_dict(self) -> dict:
        data = asdict(self)
        data["line_total"] = self.line_total
        return data


@dataclass
class Order:
    id: str
    created_at: str
    updated_at: str
    call_id: str
    customer: Customer
    items: list[OrderItem]
    subtotal: float
    delivery_fee: float
    total: float
    status: str
    requires_prescription: bool
    escalation_ids: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "call_id": self.call_id,
            "customer": self.customer.to_dict(),
            "items": [i.to_dict() for i in self.items],
            "subtotal": self.subtotal,
            "delivery_fee": self.delivery_fee,
            "total": self.total,
            "status": self.status,
            "requires_prescription": self.requires_prescription,
            "escalation_ids": list(self.escalation_ids),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "Order":
        return cls(
            id=raw["id"],
            created_at=raw["created_at"],
            updated_at=raw.get("updated_at", raw["created_at"]),
            call_id=raw.get("call_id", ""),
            customer=Customer(**raw["customer"]),
            items=[
                OrderItem(**{k: v for k, v in item.items() if k != "line_total"})
                for item in raw["items"]
            ],
            subtotal=raw["subtotal"],
            delivery_fee=raw.get("delivery_fee", 0.0),
            total=raw["total"],
            status=raw["status"],
            requires_prescription=raw.get("requires_prescription", False),
            escalation_ids=list(raw.get("escalation_ids", [])),
            notes=list(raw.get("notes", [])),
        )

    @property
    def is_open(self) -> bool:
        return OrderStatus(self.status) in OPEN_STATUSES

    def summary_line(self, kb: PharmacyKnowledgeBase | None = None) -> str:
        kb = kb or get_knowledge_base()
        items = ", ".join(f"{i.quantity} x {i.brand_name}" for i in self.items)
        return f"{self.id}: {items} for {self.customer.name} - {kb.format_price(self.total)} ({self.status})"


# --------------------------------------------------------------------------
# Inventory checking
# --------------------------------------------------------------------------


@dataclass
class StockCheck:
    """Answer to 'can we supply N packs of this right now?'"""

    medicine: Medicine
    requested: int
    on_hand: int
    reserved: int
    available: int
    can_fulfil: bool
    message: str
    alternatives: list[Medicine] = field(default_factory=list)

    @property
    def shortfall(self) -> int:
        return max(0, self.requested - self.available)

    def to_dict(self) -> dict:
        return {
            "medicine_id": self.medicine.id,
            "brand_name": self.medicine.brand_name,
            "requested": self.requested,
            "on_hand": self.on_hand,
            "reserved": self.reserved,
            "available": self.available,
            "can_fulfil": self.can_fulfil,
            "shortfall": self.shortfall,
            "message": self.message,
            "alternatives": [a.brand_name for a in self.alternatives],
        }


# --------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------


class OrderStore:
    """JSON-backed order records. Safe for the single-process demo server."""

    def __init__(self, path: Path | str = ORDERS_FILE) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._orders: dict[str, Order] = {}
        self._sequence = 0
        self.load()

    # -- persistence ------------------------------------------------------

    def load(self) -> None:
        if not self.path.exists():
            self._orders = {}
            self._sequence = _SEQUENCE_START
            return
        with self.path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        self._orders = {row["id"]: Order.from_dict(row) for row in raw.get("orders", [])}
        self._sequence = int(raw.get("sequence", _SEQUENCE_START + len(self._orders)))
        logger.info("Loaded %d orders from %s", len(self._orders), self.path)

    def _save(self) -> None:
        payload = {
            "sequence": self._sequence,
            "updated_at": _now(),
            "orders": [o.to_dict() for o in self._orders.values()],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, self.path)

    # -- reads ------------------------------------------------------------

    def get(self, order_id: str) -> Order | None:
        return self._orders.get(order_id.upper().strip())

    def list_orders(
        self, status: str | None = None, phone: str | None = None
    ) -> list[Order]:
        orders = sorted(self._orders.values(), key=lambda o: o.created_at, reverse=True)
        if status:
            orders = [o for o in orders if o.status == status]
        if phone:
            digits = "".join(ch for ch in phone if ch.isdigit())
            orders = [
                o
                for o in orders
                if digits and digits in "".join(ch for ch in o.customer.phone if ch.isdigit())
            ]
        return orders

    def reserved_quantity(self, medicine_id: str) -> int:
        """Packs held by orders that have not been completed or cancelled."""
        return sum(
            item.quantity
            for order in self._orders.values()
            if order.is_open
            for item in order.items
            if item.medicine_id == medicine_id
        )

    # -- inventory --------------------------------------------------------

    def check_stock(
        self,
        medicine: Medicine,
        quantity: int = 1,
        kb: PharmacyKnowledgeBase | None = None,
    ) -> StockCheck:
        kb = kb or get_knowledge_base()
        reserved = self.reserved_quantity(medicine.id)
        available = max(0, medicine.stock_qty - reserved)
        can_fulfil = quantity <= available

        if available <= 0:
            message = f"{medicine.brand_name} is out of stock."
        elif can_fulfil:
            message = (
                f"{quantity} pack{'s' if quantity != 1 else ''} of {medicine.brand_name} "
                f"can be reserved, {available} available."
            )
        else:
            message = (
                f"Only {available} pack{'s' if available != 1 else ''} of "
                f"{medicine.brand_name} available, {quantity} requested."
            )

        alternatives: list[Medicine] = []
        if not can_fulfil:
            alternatives = [
                alt
                for alt in kb.alternatives_for(medicine)
                if alt.stock_qty - self.reserved_quantity(alt.id) >= quantity
            ]
            if alternatives:
                message += " Alternatives available: " + ", ".join(
                    a.brand_name for a in alternatives
                ) + "."

        return StockCheck(
            medicine=medicine,
            requested=quantity,
            on_hand=medicine.stock_qty,
            reserved=reserved,
            available=available,
            can_fulfil=can_fulfil,
            message=message,
            alternatives=alternatives,
        )

    # -- writes -----------------------------------------------------------

    def _next_id(self) -> str:
        # Short and speakable - a date-stamped id reads back as a 30-syllable
        # mouthful on a phone call. The date lives in created_at instead.
        self._sequence += 1
        return f"ORD-{self._sequence:04d}"

    def create_order(
        self,
        customer: Customer,
        items: list[OrderItem],
        call_id: str = "",
        status: OrderStatus = OrderStatus.CONFIRMED,
        notes: list[str] | None = None,
        kb: PharmacyKnowledgeBase | None = None,
    ) -> Order:
        """
        Record a confirmed order. Re-checks stock first, so two callers cannot
        both reserve the last pack.
        """
        if not items:
            raise OrderError("An order needs at least one item.")
        kb = kb or get_knowledge_base()

        with self._lock:
            for item in items:
                medicine = kb.get(item.medicine_id)
                if medicine is None:
                    raise OrderError(f"Unknown medicine: {item.medicine_id}")
                check = self.check_stock(medicine, item.quantity, kb=kb)
                if not check.can_fulfil:
                    raise OrderError(check.message)

            subtotal = round(sum(i.line_total for i in items), 2)
            delivery_fee = self._delivery_fee(customer, subtotal, kb)
            requires_prescription = any(i.prescription_required for i in items)

            if requires_prescription and status == OrderStatus.CONFIRMED:
                status = OrderStatus.AWAITING_PRESCRIPTION

            now = _now()
            order = Order(
                id=self._next_id(),
                created_at=now,
                updated_at=now,
                call_id=call_id,
                customer=customer,
                items=items,
                subtotal=subtotal,
                delivery_fee=delivery_fee,
                total=round(subtotal + delivery_fee, 2),
                status=status.value,
                requires_prescription=requires_prescription,
                notes=list(notes or []),
            )
            self._orders[order.id] = order
            self._save()

        logger.info("Created order %s (%s)", order.id, order.status)
        return order

    @staticmethod
    def _delivery_fee(customer: Customer, subtotal: float, kb: PharmacyKnowledgeBase) -> float:
        if customer.fulfilment != Fulfilment.DELIVERY.value:
            return 0.0
        delivery = kb.pharmacy.get("delivery", {})
        if not delivery.get("available"):
            return 0.0
        if subtotal >= float(delivery.get("free_above", 0)):
            return 0.0
        return float(delivery.get("fee", 0.0))

    def update_status(self, order_id: str, status: OrderStatus, note: str = "") -> Order:
        with self._lock:
            order = self.get(order_id)
            if order is None:
                raise OrderError(f"Order not found: {order_id}")
            order.status = status.value
            order.updated_at = _now()
            if note:
                order.notes.append(note)
            self._save()
        logger.info("Order %s -> %s", order.id, order.status)
        return order

    def attach_escalation(self, order_id: str, escalation_id: str) -> None:
        with self._lock:
            order = self.get(order_id)
            if order is None:
                return
            if escalation_id not in order.escalation_ids:
                order.escalation_ids.append(escalation_id)
                order.updated_at = _now()
                self._save()

    def cancel(self, order_id: str, reason: str = "") -> Order:
        return self.update_status(order_id, OrderStatus.CANCELLED, reason)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# Module-level singleton
# --------------------------------------------------------------------------

_store: OrderStore | None = None


def get_order_store() -> OrderStore:
    global _store
    if _store is None:
        _store = OrderStore()
    return _store
