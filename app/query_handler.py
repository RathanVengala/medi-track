"""
Customer query handling (Week 2).

Takes a transcribed customer utterance and turns it into structured
information the LLM can be grounded on:

    raw text -> intent + entities -> knowledge base lookup -> context block

The context block is injected into the LLM turn so replies are built from
real formulary data instead of the model's imagination.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum

from app.knowledge_base import (
    AvailabilityResult,
    Medicine,
    PharmacyKnowledgeBase,
    SearchHit,
    get_knowledge_base,
    normalize,
)

logger = logging.getLogger("meditrack.query")


class Intent(str, Enum):
    GREETING = "greeting"
    AVAILABILITY = "availability"
    PRICE = "price"
    MEDICINE_INFO = "medicine_info"
    PRESCRIPTION = "prescription"
    REFILL = "refill"
    ORDER_STATUS = "order_status"
    DELIVERY = "delivery"
    STORE_INFO = "store_info"
    SYMPTOM = "symptom"
    CLINICAL = "clinical"
    GOODBYE = "goodbye"
    UNKNOWN = "unknown"


# Checked in order; the first pattern group that matches wins. Clinical
# questions are tested first because they must always route to a pharmacist.
_INTENT_PATTERNS: tuple[tuple[Intent, tuple[str, ...]], ...] = (
    (
        Intent.CLINICAL,
        (
            r"\bhow (much|many|often)\b.*\b(take|dose|dosage)\b",
            r"\bdosage\b",
            r"\bdose\b",
            r"\bsafe to (take|use)\b",
            r"\b(interact|interaction)\b",
            r"\bside effects?\b",
            r"\bpregnan(t|cy)\b",
            r"\bbreast ?feed",
            r"\bwith (alcohol|my other)\b",
            r"\bcan i take\b.*\b(with|and)\b",
            r"\bwhat.s wrong with me\b",
            r"\bdiagnos",
        ),
    ),
    (
        Intent.GOODBYE,
        (r"\b(goodbye|bye|that.s all|thats all|hang up|nothing else|no thank you)\b",),
    ),
    # Unambiguous "tell me about it" phrasings are matched before the purchase
    # and price patterns, so "I would like to know about X" is not read as an
    # order. The looser "what is X" patterns stay near the bottom, where they
    # cannot steal "what is your address" from STORE_INFO.
    (
        Intent.MEDICINE_INFO,
        (
            r"\btell me about\b",
            r"\bknow about\b",
            r"\bwhat.* used for\b",
            r"\bwhat does .* do\b",
            r"\bmore (information|info|details) (about|on)\b",
        ),
    ),
    (
        Intent.PRICE,
        (
            r"\bhow much\b",
            r"\b(price|cost|costs|charge|rate)\b",
            r"\bexpensive\b",
        ),
    ),
    # Refill and order-status are matched before availability, because "I need a
    # refill" would otherwise be read as a purchase request.
    (
        Intent.REFILL,
        (r"\brefill\b", r"\brepeat prescription\b", r"\brenew\b", r"\btop up my\b"),
    ),
    (
        Intent.ORDER_STATUS,
        (
            r"\b(my|the) order\b",
            r"\border (status|number|ready)\b",
            r"\bis it ready\b",
            r"\bready for (pickup|pick up|collection)\b",
        ),
    ),
    (
        Intent.AVAILABILITY,
        (
            r"\b(do you|have you|got any|you got)\b.*\b(have|got|stock|carry)\b",
            r"\bin stock\b",
            r"\bavailable\b",
            r"\bavailability\b",
            r"\bdo you (have|carry|stock|sell)\b",
            r"\bis there any\b",
            r"\bany .* left\b",
            # Purchase requests are answered the same way: can we supply it?
            r"\bi (need|want)\b",
            r"\b(can|could) i (get|have|buy)\b",
            r"\bi.d like\b",
            r"\bi would like\b",
            r"\bgive me\b",
            r"\blooking for\b",
        ),
    ),
    (
        Intent.DELIVERY,
        (r"\bdeliver", r"\bshipping\b", r"\bhome delivery\b", r"\bdrop it off\b"),
    ),
    (
        Intent.PRESCRIPTION,
        (
            r"\bprescription\b",
            r"\bwithout a script\b",
            r"\bover the counter\b",
            r"\bdoctor.s note\b",
        ),
    ),
    (
        Intent.STORE_INFO,
        (
            r"\b(what time|when) (do|are) you\b",
            r"\b(opening|closing) (time|hours)\b",
            r"\bare you open\b",
            r"\byour (address|location|timing|hours|phone)\b",
            r"\bwhere are you\b",
            r"\bhow late\b",
        ),
    ),
    (
        Intent.MEDICINE_INFO,
        (
            r"\bwhat is\b",
            r"\bwhat.s\b",
            r"\bwhat.* used for\b",
            r"\btell me about\b",
            r"\bwhat does .* do\b",
        ),
    ),
    (
        Intent.SYMPTOM,
        (
            r"\bfor (a |my )?(headache|fever|cough|cold|pain|allergy|acidity|rash)\b",
            r"\bi have (a )?(headache|fever|cough|cold|sore throat|rash)\b",
            r"\bsomething for\b",
        ),
    ),
    # Checked last, so "Hi, do you have amoxicillin?" is classified by the real
    # request rather than by the greeting it opens with.
    (
        Intent.GREETING,
        (r"^(hi|hello|hey|good (morning|afternoon|evening))\b",),
    ),
)

_FORMS = (
    "tablet",
    "tablets",
    "capsule",
    "capsules",
    "syrup",
    "injection",
    "inhaler",
    "cream",
    "gel",
    "drops",
    "sachet",
    "sachets",
    "powder",
    "pen",
)

_WORD_NUMBERS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "a couple of": 2,
    "a few": 3,
    "half a dozen": 6,
    "a dozen": 12,
}

# Words that never start a medicine name — skipped when generating n-grams.
# Deliberately excludes words that DO appear in product names ("cold", "cough",
# "vitamin", "gel", "pen"), which must stay searchable.
_NGRAM_STOPWORDS = {
    "a", "about", "afternoon", "am", "an", "and", "any", "anyone", "anything",
    "are", "at", "available", "be", "box", "boxes", "but", "buy", "by", "call",
    "called", "can", "card", "cards", "carry", "close", "closed", "closes",
    "closing", "collect", "correct", "cost",
    "costs", "could", "day", "do", "does", "evening", "for", "friday", "from",
    "get", "give", "got", "has", "have", "hello", "help", "hi", "how", "i", "if",
    "in", "insurance", "is", "it", "just", "late", "left", "like", "many", "me",
    "monday", "month", "more", "morning", "much", "my", "name", "need", "night",
    "number", "of", "okay", "on", "one", "open", "opening", "opens", "or",
    "order", "pack", "packs", "phone",
    "please", "price", "repeat", "saturday", "some", "someone", "something",
    "sorry", "speak", "stock", "strip", "sunday", "tablet", "tablets", "take",
    "taking", "thank",
    "thanks", "that", "the", "there", "they", "this", "thursday", "to", "today",
    "tomorrow", "tonight", "tuesday", "want", "was", "we", "wednesday", "week",
    "what", "when", "where", "which", "will", "with", "would", "yes",
    "yesterday", "you", "your",
}

_MAX_NGRAM = 3
# All correctly-heard medicine names in the formulary score >= 0.83, including
# badly transcribed ones, so this rejects near-miss English words safely.
_ENTITY_THRESHOLD = 0.78


@dataclass
class ExtractedMedicine:
    """A medicine mentioned in the utterance, resolved against the KB."""

    spoken_text: str
    medicine: Medicine
    score: float
    match_type: str


@dataclass
class QueryResult:
    text: str
    intent: Intent
    medicines: list[ExtractedMedicine] = field(default_factory=list)
    availability: list[AvailabilityResult] = field(default_factory=list)
    suggestions: list[SearchHit] = field(default_factory=list)
    quantity: int | None = None
    strength: str | None = None
    form: str | None = None
    needs_pharmacist: bool = False

    @property
    def matched(self) -> bool:
        return bool(self.medicines)

    @property
    def primary(self) -> Medicine | None:
        return self.medicines[0].medicine if self.medicines else None


# --------------------------------------------------------------------------
# Intent + entity extraction
# --------------------------------------------------------------------------


def detect_intent(text: str) -> Intent:
    lowered = normalize(text)
    if not lowered:
        return Intent.UNKNOWN
    for intent, patterns in _INTENT_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, lowered):
                return intent
    return Intent.UNKNOWN


def extract_quantity(text: str) -> int | None:
    lowered = normalize(text)
    match = re.search(
        r"\b(\d{1,3})\s*(tablets?|capsules?|packs?|boxes|box|strips?|bottles?|sachets?|tubes?|pieces?)\b",
        lowered,
    )
    if match:
        return int(match.group(1))
    for word, value in _WORD_NUMBERS.items():
        if re.search(
            rf"\b{re.escape(word)}\s+(tablets?|capsules?|packs?|boxes|box|strips?|bottles?|sachets?|tubes?)\b",
            lowered,
        ):
            return value
    # Bare "I need 2" style requests.
    match = re.search(r"\b(?:need|want|give me|order)\s+(\d{1,3})\b", lowered)
    return int(match.group(1)) if match else None


def extract_strength(text: str) -> str | None:
    match = re.search(
        r"\b(\d+(?:\.\d+)?)\s*(mg|mcg|milligrams?|micrograms?|ml|g|iu|percent|%)\b",
        normalize(text),
    )
    if not match:
        return None
    unit = match.group(2)
    unit = {"milligram": "mg", "milligrams": "mg", "microgram": "mcg", "micrograms": "mcg"}.get(
        unit, unit
    )
    return f"{match.group(1)} {unit}"


def extract_form(text: str) -> str | None:
    lowered = normalize(text)
    for form in _FORMS:
        if re.search(rf"\b{form}\b", lowered):
            return form.rstrip("s")
    return None


def _ngrams(tokens: list[str], max_n: int = _MAX_NGRAM) -> list[tuple[int, int, str]]:
    """All (start, end, text) n-grams that do not begin with a stopword."""
    grams: list[tuple[int, int, str]] = []
    for size in range(min(max_n, len(tokens)), 0, -1):
        for start in range(len(tokens) - size + 1):
            window = tokens[start : start + size]
            if window[0] in _NGRAM_STOPWORDS:
                continue
            grams.append((start, start + size, " ".join(window)))
    return grams


def extract_medicines(
    text: str,
    kb: PharmacyKnowledgeBase | None = None,
    threshold: float = _ENTITY_THRESHOLD,
) -> list[ExtractedMedicine]:
    """
    Find every medicine name mentioned in the utterance.

    Scans longest-first n-grams so "amoxicillin five hundred" resolves as one
    entity, and drops overlapping matches so a name is only reported once.
    """
    kb = kb or get_knowledge_base()
    tokens = normalize(text).split()
    if not tokens:
        return []

    candidates: list[tuple[int, int, ExtractedMedicine]] = []
    for start, end, gram in _ngrams(tokens):
        hits = kb.search(gram, limit=1, threshold=threshold, include_uses=False)
        if hits:
            hit = hits[0]
            candidates.append(
                (start, end, ExtractedMedicine(gram, hit.medicine, hit.score, hit.match_type))
            )

    # Prefer higher scores, then longer spans, then earlier position.
    candidates.sort(key=lambda c: (-c[2].score, -(c[1] - c[0]), c[0]))

    taken_positions: set[int] = set()
    seen_ids: set[str] = set()
    resolved: list[tuple[int, ExtractedMedicine]] = []
    for start, end, extracted in candidates:
        span = set(range(start, end))
        if span & taken_positions or extracted.medicine.id in seen_ids:
            continue
        taken_positions |= span
        seen_ids.add(extracted.medicine.id)
        resolved.append((start, extracted))

    resolved.sort(key=lambda r: r[0])
    return [extracted for _, extracted in resolved]


# --------------------------------------------------------------------------
# Top-level handler
# --------------------------------------------------------------------------


def handle_query(text: str, kb: PharmacyKnowledgeBase | None = None) -> QueryResult:
    """Run the full customer query -> knowledge base search pipeline."""
    kb = kb or get_knowledge_base()

    intent = detect_intent(text)
    medicines = extract_medicines(text, kb=kb)

    result = QueryResult(
        text=text,
        intent=intent,
        medicines=medicines,
        quantity=extract_quantity(text),
        strength=extract_strength(text),
        form=extract_form(text),
        needs_pharmacist=intent == Intent.CLINICAL,
    )

    # Availability is resolved whenever a medicine was named at all - the
    # caller usually wants to know if we have it even when they asked about
    # price or what it is used for.
    if medicines:
        result.availability = [kb.check_availability(m.spoken_text) for m in medicines]
    elif intent in (Intent.SYMPTOM, Intent.AVAILABILITY, Intent.MEDICINE_INFO, Intent.UNKNOWN):
        result.suggestions = kb.search_by_use(text, limit=3)

    logger.info(
        "Query handled: intent=%s medicines=%s quantity=%s",
        intent.value,
        [m.medicine.brand_name for m in medicines],
        result.quantity,
    )
    return result


# --------------------------------------------------------------------------
# Grounding context for the LLM
# --------------------------------------------------------------------------


def build_context(result: QueryResult, kb: PharmacyKnowledgeBase | None = None) -> str:
    """
    Render the knowledge base findings as a plain-text block for the LLM.

    Everything the model is allowed to state as fact appears here.
    """
    kb = kb or get_knowledge_base()
    lines = [
        "PHARMACY KNOWLEDGE BASE RESULT (use only these facts; do not invent stock, prices or medical advice)",
        f"Detected intent: {result.intent.value}",
    ]
    if result.quantity:
        lines.append(f"Requested quantity: {result.quantity}")
    if result.strength:
        lines.append(f"Requested strength: {result.strength}")
    if result.form:
        lines.append(f"Requested form: {result.form}")

    if result.needs_pharmacist:
        lines.append(
            "NOTE: this is a clinical question. Do not answer it. Offer to connect the "
            "caller with the pharmacist on duty."
        )

    if result.availability:
        for avail in result.availability:
            med = avail.medicine
            if med is None:
                lines.append(f"- Searched '{avail.query}': {avail.message}")
                continue
            lines.append(f"- Medicine: {med.display_name} [{med.form}]")
            lines.append(f"  Heard as: '{avail.query}'")
            lines.append(f"  Availability: {avail.message}")
            lines.append(f"  Price: {kb.price_text(med)}")
            lines.append(
                "  Prescription required: "
                + ("yes" if med.prescription_required else "no, available over the counter")
            )
            lines.append(f"  Used for: {med.common_uses}")
            lines.append(f"  Storage: {med.storage}")
            lines.append(f"  Shelf location: {med.shelf_location}")
            if avail.alternatives:
                alts = ", ".join(
                    f"{a.brand_name} ({'in stock' if a.in_stock else 'out of stock'})"
                    for a in avail.alternatives
                )
                lines.append(f"  Alternatives: {alts}")
    elif result.suggestions:
        lines.append("- No exact medicine name recognised. Closest stocked options:")
        for hit in result.suggestions:
            med = hit.medicine
            lines.append(
                f"  * {med.brand_name} ({med.generic_name} {med.strength}) - {med.category} - "
                f"{kb.price_text(med)} - {med.stock_sentence()}"
            )
    else:
        lines.append("- No medicine matched this query in the formulary.")

    if result.intent in (Intent.STORE_INFO, Intent.DELIVERY, Intent.PRESCRIPTION):
        lines.append("STORE PROFILE:")
        lines.append(kb.store_summary())

    return "\n".join(lines)


def fallback_answer(result: QueryResult, kb: PharmacyKnowledgeBase | None = None) -> str:
    """
    A deterministic, knowledge-base-only reply.

    Used when no LLM provider is configured, and as a safety net if the LLM
    call fails, so the demo never dead-ends on a missing API key.
    """
    kb = kb or get_knowledge_base()
    pharmacy_name = kb.pharmacy.get("name", "the pharmacy")

    if result.needs_pharmacist:
        return (
            "That one is best answered by our pharmacist. Let me connect you with "
            f"{kb.pharmacy.get('pharmacist_on_duty', 'the pharmacist on duty')}."
        )

    if result.intent == Intent.GREETING:
        return f"Thanks for calling {pharmacy_name}. How can I help you today?"

    if result.intent == Intent.GOODBYE:
        return f"Thanks for calling {pharmacy_name}. Have a great day!"

    if result.intent == Intent.STORE_INFO and not result.matched:
        hours = kb.pharmacy.get("hours", {})
        return (
            f"We're open {hours.get('monday')} Monday to Friday, {hours.get('saturday')} "
            f"on Saturday, and {hours.get('sunday')} on Sunday."
        )

    if result.intent == Intent.DELIVERY and not result.matched:
        delivery = kb.pharmacy.get("delivery", {})
        return (
            f"Yes, we deliver within {delivery.get('radius_miles')} miles. It's free above "
            f"{kb.format_price(delivery.get('free_above', 0))}, and same day if you order "
            f"before {delivery.get('same_day_cutoff')}."
        )

    if result.availability:
        avail = result.availability[0]
        med = avail.medicine
        if med is None:
            return avail.message
        parts = [avail.message]
        if result.intent == Intent.PRICE or med.in_stock:
            parts.append(f"It's {kb.price_text(med)}.")
        if med.prescription_required:
            parts.append("It's prescription-only, so we'll need a valid prescription.")
        return " ".join(parts)

    if result.suggestions:
        names = ", ".join(h.medicine.brand_name for h in result.suggestions)
        return f"I didn't catch that medicine name. For that we stock {names}. Which one would you like?"

    return (
        "I didn't quite catch the medicine name. Could you spell it out or tell me "
        "what it's for?"
    )
