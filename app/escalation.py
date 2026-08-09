"""
Pharmacist escalation queue (Week 3).

Anything the AI must not decide on its own becomes a queued item for the
pharmacist on duty: clinical questions, prescription-only orders, stock the
agent cannot supply, cold-chain items, and explicit requests for a human.

Escalations are persisted to data/escalations.json so the pharmacist can work
the queue after the call has ended.
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

from app.knowledge_base import DATA_DIR

logger = logging.getLogger("meditrack.escalation")

ESCALATIONS_FILE = DATA_DIR / "escalations.json"


class EscalationReason(str, Enum):
    CLINICAL_QUESTION = "clinical_question"
    PRESCRIPTION_REQUIRED = "prescription_required"
    OUT_OF_STOCK = "out_of_stock"
    INSUFFICIENT_STOCK = "insufficient_stock"
    COLD_CHAIN = "cold_chain"
    UNRECOGNISED_MEDICINE = "unrecognised_medicine"
    CUSTOMER_REQUEST = "customer_request"


class EscalationStatus(str, Enum):
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


class Priority(str, Enum):
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


# Clinical questions and cold-chain items are what a pharmacist most needs to
# see first; everything else can wait for the queue to be worked in order.
_PRIORITY = {
    EscalationReason.CLINICAL_QUESTION: Priority.HIGH,
    EscalationReason.COLD_CHAIN: Priority.HIGH,
    EscalationReason.CUSTOMER_REQUEST: Priority.HIGH,
    EscalationReason.PRESCRIPTION_REQUIRED: Priority.NORMAL,
    EscalationReason.INSUFFICIENT_STOCK: Priority.NORMAL,
    EscalationReason.OUT_OF_STOCK: Priority.LOW,
    EscalationReason.UNRECOGNISED_MEDICINE: Priority.LOW,
}

# What the agent should tell the caller for each reason.
SPOKEN_REASON = {
    EscalationReason.CLINICAL_QUESTION: "a pharmacist needs to answer this one",
    EscalationReason.PRESCRIPTION_REQUIRED: "this medicine needs a valid prescription",
    EscalationReason.OUT_OF_STOCK: "we are out of stock and need to reorder it",
    EscalationReason.INSUFFICIENT_STOCK: "we do not have that many packs on the shelf",
    EscalationReason.COLD_CHAIN: "this item is refrigerated and handled by the pharmacist",
    EscalationReason.UNRECOGNISED_MEDICINE: "I could not find that medicine in our system",
    EscalationReason.CUSTOMER_REQUEST: "you asked to speak with the pharmacist",
}


@dataclass
class Escalation:
    id: str
    created_at: str
    call_id: str
    reason: str
    priority: str
    summary: str
    status: str = EscalationStatus.OPEN.value
    customer_name: str | None = None
    customer_phone: str | None = None
    order_id: str | None = None
    transcript: str | None = None
    resolution: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> "Escalation":
        return cls(**raw)

    def summary_line(self) -> str:
        who = self.customer_name or "caller"
        return f"[{self.priority.upper()}] {self.id} - {self.reason} - {who}: {self.summary}"


class EscalationQueue:
    """JSON-backed queue of items awaiting the pharmacist on duty."""

    def __init__(self, path: Path | str = ESCALATIONS_FILE) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._items: dict[str, Escalation] = {}
        self._sequence = 0
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self._items = {}
            self._sequence = 0
            return
        with self.path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        self._items = {row["id"]: Escalation.from_dict(row) for row in raw.get("escalations", [])}
        self._sequence = int(raw.get("sequence", len(self._items)))
        logger.info("Loaded %d escalations from %s", len(self._items), self.path)

    def _save(self) -> None:
        payload = {
            "sequence": self._sequence,
            "updated_at": _now(),
            "escalations": [e.to_dict() for e in self._items.values()],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, self.path)

    def _next_id(self) -> str:
        self._sequence += 1
        return f"ESC-{datetime.now().strftime('%Y%m%d')}-{self._sequence:04d}"

    def raise_escalation(
        self,
        reason: EscalationReason,
        summary: str,
        call_id: str = "",
        customer_name: str | None = None,
        customer_phone: str | None = None,
        order_id: str | None = None,
        transcript: str | None = None,
    ) -> Escalation:
        with self._lock:
            item = Escalation(
                id=self._next_id(),
                created_at=_now(),
                call_id=call_id,
                reason=reason.value,
                priority=_PRIORITY.get(reason, Priority.NORMAL).value,
                summary=summary,
                customer_name=customer_name,
                customer_phone=customer_phone,
                order_id=order_id,
                transcript=transcript,
            )
            self._items[item.id] = item
            self._save()
        logger.info("Escalation %s raised (%s, %s)", item.id, item.reason, item.priority)
        return item

    def get(self, escalation_id: str) -> Escalation | None:
        return self._items.get(escalation_id.upper().strip())

    def list_escalations(
        self, status: str | None = None, reason: str | None = None
    ) -> list[Escalation]:
        order = {Priority.HIGH.value: 0, Priority.NORMAL.value: 1, Priority.LOW.value: 2}
        items = sorted(
            self._items.values(),
            key=lambda e: (order.get(e.priority, 3), e.created_at),
        )
        if status:
            items = [e for e in items if e.status == status]
        if reason:
            items = [e for e in items if e.reason == reason]
        return items

    def update_status(
        self, escalation_id: str, status: EscalationStatus, resolution: str = ""
    ) -> Escalation:
        with self._lock:
            item = self.get(escalation_id)
            if item is None:
                raise KeyError(f"Escalation not found: {escalation_id}")
            item.status = status.value
            if resolution:
                item.resolution = resolution
            self._save()
        return item

    @property
    def open_count(self) -> int:
        return sum(1 for e in self._items.values() if e.status == EscalationStatus.OPEN.value)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


_queue: EscalationQueue | None = None


def get_escalation_queue() -> EscalationQueue:
    global _queue
    if _queue is None:
        _queue = EscalationQueue()
    return _queue
