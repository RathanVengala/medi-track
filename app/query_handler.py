

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum

from app.config import settings
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
    ORDER = "order"
    PRICE = "price"
    MEDICINE_INFO = "medicine_info"
    PRESCRIPTION = "prescription"
    REFILL = "refill"
    ORDER_STATUS = "order_status"
    DELIVERY = "delivery"
    STORE_INFO = "store_info"
    SYMPTOM = "symptom"
    CLINICAL = "clinical"
    PHARMACIST = "pharmacist"
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
            # Combining two medicines is an interaction question however it is
            # phrased. "Can I mix amoxicillin with paracetamol" used to fall
            # through every group to UNKNOWN, which walked straight past the
            # clinical gate - the one route that must never be missed.
            r"\b(mix|mixing|combine|combining)\b.*\b(with|and)\b",
            r"\b(mix|mixing|combine|combining)\b.*\b(these|them|both|two|together|medicines?|medications?|tablets?|pills?)\b",
            r"\btake\b.*\b(together|at the same time|alongside|along with)\b",
            r"\b(together with|alongside)\b.*\b(medicine|medication|tablets?|pills?)\b",
            r"\bsafe\b.*\b(together|with)\b",
            r"\b(interfere|interferes|react|reacts)\b.*\bwith\b",
            r"\ballergic to\b",
            r"\bcontraindicat",
            r"\boverdose\b",
            r"\btoo (many|much)\b.*\b(tablets?|pills?|take|taken)\b",
            r"\b(how long|how many days)\b.*\b(take|use|continue|course)\b",
            r"\b(empty stomach|before food|after food|with food)\b",
            r"\bis it (safe|okay|ok)\b",
            r"\bcan i (give|use) (it|this|them)\b.*\b(child|baby|kid|toddler|infant)\b",
        ),
    ),
    # An explicit request for a human outranks everything except a clinical
    # question, which is already being handed to a pharmacist anyway.
    (
        Intent.PHARMACIST,
        (
            r"\b(speak|talk|connect) (to|with|me to)\b.*\b(pharmacist|chemist|human|person|someone|staff)\b",
            r"\b(put me through|transfer me)\b",
            r"\bis there (a|any) (real )?(person|human)\b",
            r"\bi want to speak to\b",
            # "I need human help" was classified AVAILABILITY, because the
            # generic "i need" purchase pattern matched it first. A request for
            # a person is an escalation whatever verb introduces it.
            r"\bhuman (help|being|agent|operator|assistance|support)\b",
            r"\b(real|actual|live) (person|human|agent)\b",
            r"\b(need|want|get me|give me)\b.*\b(human|pharmacist|chemist)\b",
            r"\b(customer (service|support)|operator|receptionist|supervisor|manager)\b",
            r"\bnot a (bot|robot|machine|computer)\b",
            r"\b(bot|robot|machine|computer|ai)\b.*\b(human|person|real)\b",
            r"\bsomeone (real|human)\b",
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
            # Deliberately narrow: a bare "the order" also appears in "place the
            # order", which is a confirmation, not a status question.
            r"\b(is|are|was|has) (my|the) order\b",
            r"\b(check|track|chase) (on )?(my|the) order\b",
            r"\border (status|number|ready)\b",
            r"\bstatus of (my|the) order\b",
            r"\bis it ready\b",
            r"\bready for (pickup|pick up|collection)\b",
        ),
    ),
    # An explicit purchase instruction, as opposed to "do you have it?". Matched
    # before AVAILABILITY so the order flow starts rather than a stock answer.
    (
        Intent.ORDER,
        (
            r"\b(place|make) an order\b",
            r"\bto order\b",
            r"\border (some|a|an|the|me)\b",
            r"\breserve\b",
            r"\bset (it|one|some|them) aside\b",
            r"\bput (it|one|some|them) aside\b",
            r"\bhold (it|one|some|them) for me\b",
            r"\bi.ll take\b",
            r"\bi will take\b",
            r"\bcan you (order|reserve|hold)\b",
            r"\bbook (it|me|one)\b",
        ),
    ),
    # Checked before AVAILABILITY: "do I need a prescription for this" would
    # otherwise be swallowed by the "i need" purchase pattern below.
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
        Intent.STORE_INFO,
        (
            r"\b(what time|when) (do|are) you\b",
            r"\b(opening|closing) (time|hours)\b",
            r"\bare you open\b",
            r"\byour (address|location|timing|hours|phone)\b",
            r"\bwhere are you\b",
            r"\bhow late\b",
            # Shop policy, not a medicine. "What is your return policy" used to
            # reach the loose "what is" rule below and come back as
            # MEDICINE_INFO, so the caller got a formulary answer to a shop
            # question. These run before that rule and after DELIVERY and
            # PRESCRIPTION, which own their own policies.
            r"\b(return|refund|exchange|privacy|payment|cancellation)s? polic",
            r"\bpolicy\b",
            r"\bcan i (return|exchange|refund)\b",
            r"\b(do you|can i) (accept|take|pay)\b.*\b(insurance|cards?|cash|upi|credit|debit)\b",
            r"\bhow do i (pay|contact|reach) you\b",
            r"\b(parking|wheelchair|pharmacist on duty)\b",
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
    # Set when a number was heard but refused - negative, zero, or over the
    # per-call pack limit. The order flow re-asks instead of guessing.
    quantity_issue: QuantityIssue | None = None
    strength: str | None = None
    form: str | None = None
    needs_pharmacist: bool = False
    # True when the utterance was longer than MAX_QUERY_CHARS and was cut down
    # before analysis.
    truncated: bool = False
    # Week 3 - details needed to turn an enquiry into an order.
    person_name: str | None = None
    phone: str | None = None
    fulfilment: str | None = None
    address: str | None = None
    confirmation: bool | None = None
    order_reference: str | None = None

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


# Every dispensing unit the pharmacy sells in. Missing units silently read as a
# quantity of one, which is how "ten inhalers" became a single-pack order.
_QUANTITY_UNITS = (
    r"(?:tablets?|capsules?|packs?|packets?|packages?|boxes|box|strips?|bottles?|sachets?"
    r"|tubes?|inhalers?|pens?|vials?|cartons?|units?|pieces?|gels?|creams?|syrups?)"
)


def _quantity_words_to_digits(text: str) -> str:
    """
    Spoken counts to digits, including the teens and "a dozen".

    Kept separate from the phone-number converter, which must only map single
    digits - "ten" is a quantity, never part of a dictated phone number.
    """
    for phrase, value in sorted(_WORD_NUMBERS.items(), key=lambda kv: -len(kv[0])):
        if " " in phrase:
            text = text.replace(phrase, str(value))
    return " ".join(str(_WORD_NUMBERS.get(token, token)) for token in text.split())


class QuantityIssue(str, Enum):
    """Why a spoken number was refused as a pack count."""

    NEGATIVE = "negative"
    ZERO = "zero"
    TOO_MANY = "too_many"


# A minus sign that belongs to the number rather than to a hyphenated name.
# Excluded: "co-amoxiclav" (letter before the dash) and "98765-43210" (digit
# before it), neither of which is a negative quantity.
_NEGATIVE_NUMBER = re.compile(r"(?:(?<![\w.])-\s*|\b(?:minus|negative)\s+)(\d{1,6})", re.I)


def _negative_numbers(text: str) -> set[int]:
    """
    Numbers the caller said with a minus in front.

    normalize() strips punctuation, so by the time the quantity patterns run
    the sign is already gone - which is exactly why "give me -5 tablets" came
    back as a cheerful 5. This works on a lighter normalisation that keeps the
    minus sign, and still converts spoken numbers, so "minus two packs" is
    caught as well as "-2".
    """
    lowered = re.sub(r"[^a-z0-9\-\s]+", " ", (text or "").lower())
    signed = _quantity_words_to_digits(lowered)
    return {int(m.group(1)) for m in _NEGATIVE_NUMBER.finditer(signed)}


def parse_quantity(text: str) -> tuple[int | None, QuantityIssue | None]:
    """
    Number of packs the caller asked for, plus why it was refused if it was.

    Spoken numbers are converted to digits first, so "two boxes" and "2 boxes"
    are handled by the same pattern. A number that is negative, zero, or above
    the per-call pack limit returns `(None, issue)` - the flow then re-asks
    rather than carrying a nonsense count towards an order.
    """
    lowered = _quantity_words_to_digits(normalize(text))
    match = re.search(rf"\b(\d{{1,6}})\s*(?:x\s*)?{_QUANTITY_UNITS}\b", lowered)
    if not match:
        # Bare "I need 2" / "reserve 3" style requests.
        match = re.search(
            r"\b(?:need|want|order|reserve|book|take|buy|give me|like)\s+(\d{1,6})\b", lowered
        )
    if not match:
        return None, None

    value = int(match.group(1))
    if value in _negative_numbers(text):
        return None, QuantityIssue.NEGATIVE
    if value == 0:
        return None, QuantityIssue.ZERO
    if value > settings.MAX_ORDER_QUANTITY:
        return None, QuantityIssue.TOO_MANY
    return value, None


def extract_quantity(text: str) -> int | None:
    """Valid pack count, or None when the caller gave none or gave a bad one."""
    quantity, _ = parse_quantity(text)
    return quantity


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


# --------------------------------------------------------------------------
# Customer detail extraction (Week 3 - order taking)
# --------------------------------------------------------------------------

_NAME_PATTERNS = (
    r"\bmy name is\s+([a-z][a-z '\-]{1,40})",
    r"\bthis is\s+([a-z][a-z '\-]{1,40})",
    r"\bit.s\s+([a-z][a-z '\-]{1,40})\s+(?:here|speaking|calling)",
    r"\bi am\s+([a-z][a-z '\-]{1,40})\s+(?:here|speaking|calling)",
    r"\bunder the name\s+([a-z][a-z '\-]{1,40})",
    r"\bname.s\s+([a-z][a-z '\-]{1,40})",
)

# Trailing filler that tends to be swept up with a spoken name.
_NAME_TRAILERS = (
    "calling", "speaking", "here", "and", "please", "thanks", "thank you",
    "i would like", "i want", "i need",
)

_DIGIT_WORDS = {
    "zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}


def extract_person_name(text: str) -> str | None:
    """Pull a caller's name out of a self-introduction."""
    lowered = normalize(text)
    for pattern in _NAME_PATTERNS:
        match = re.search(pattern, lowered)
        if match:
            return _clean_name(match.group(1))
    return None


def name_from_bare_reply(text: str) -> str | None:
    """
    Treat a short utterance as a name.

    Only used when the order flow has just asked "who is this for?", where a
    bare "Rathan Vengala" is the expected answer and there is no pattern to
    anchor on.
    """
    explicit = extract_person_name(text)
    if explicit:
        return explicit
    cleaned = _clean_name(normalize(text))
    if not cleaned:
        return None
    words = cleaned.split()
    if 1 <= len(words) <= 4 and all(w.isalpha() for w in words):
        return cleaned
    return None


def _clean_name(raw: str) -> str:
    name = re.sub(r"[^a-z '\-]+", " ", (raw or "").lower())
    name = re.sub(r"\s+", " ", name).strip()
    # "It's Rathan" normalises to "it s rathan"; drop the contraction remnant
    # before stopword filtering, or the orphan "s" survives as a first name.
    name = re.sub(r"^(?:it|that|this|name|here|he|she)\s+s\s+", "", name)
    name = re.sub(r"^(?:its|thats|names)\s+", "", name)
    for trailer in _NAME_TRAILERS:
        if name.endswith(" " + trailer):
            name = name[: -len(trailer) - 1].strip()
        if name.startswith(trailer + " "):
            name = name[len(trailer) + 1 :].strip()
    # A name is never one of our filler words.
    words = [w for w in name.split() if w not in _NGRAM_STOPWORDS]
    return " ".join(w.capitalize() for w in words[:4])


def _words_to_digits(text: str) -> str:
    return " ".join(_DIGIT_WORDS.get(token, token) for token in text.split())


def extract_phone(text: str, min_digits: int = 10) -> str | None:
    """
    Find a phone number, whether dictated as digits or as words.

    Defaults to 10-15 digits so a country code is tolerated while the short
    numbers in strengths and quantities are ignored.
    """
    lowered = _words_to_digits(normalize(text))
    for candidate in re.findall(r"\d[\d ]*\d", lowered):
        digits = re.sub(r"\D", "", candidate)
        if min_digits <= len(digits) <= 15:
            return digits
    digits = re.sub(r"\D", "", lowered)
    if min_digits <= len(digits) <= 15:
        return digits
    return None


def phone_from_bare_reply(text: str) -> str | None:
    """
    Read a phone number from the answer to "what's your number?".

    The question makes a bare run of digits unambiguous, so this accepts
    shorter numbers than the general extractor - landlines, extensions, and
    the digits Whisper drops from a fast-dictated mobile number.
    """
    return extract_phone(text, min_digits=6)


def extract_fulfilment(text: str) -> str | None:
    """"pickup" or "delivery", or None when the caller said neither or both."""
    lowered = normalize(text)
    wants_delivery = bool(
        re.search(r"\b(deliver|delivery|send it|send them|ship it|to my (home|house|address))\b", lowered)
    )
    wants_pickup = bool(
        re.search(
            r"\b(pick ?up|collect|collection|come (in|by|over|and get)|in store|at the counter|i.ll come)\b",
            lowered,
        )
    )
    if wants_delivery and not wants_pickup:
        return "delivery"
    if wants_pickup and not wants_delivery:
        return "pickup"
    return None


def extract_confirmation(text: str) -> bool | None:
    """True for yes, False for no, None when the caller said neither."""
    lowered = normalize(text)
    if re.search(
        r"^(yes|yeah|yep|yup|sure|ok|okay|correct|confirm|confirmed|please do|go ahead|absolutely|that.s right|sounds good)\b",
        lowered,
    ):
        return True
    if re.search(
        r"^(no|nope|nah|not now|not really|cancel|don.t|do not|never ?mind|forget it)\b",
        lowered,
    ):
        return False
    if re.search(r"\b(cancel (the|my) order|never ?mind)\b", lowered):
        return False
    if re.search(r"\b(that.s (right|correct)|please confirm|go ahead)\b", lowered):
        return True
    return None


def extract_order_reference(text: str) -> str | None:
    """
    Find an order number a caller reads back.

    Handles "ORD-1001", "order 1001" and the spoken form "one zero zero one",
    which is how the agent reads it out in the first place.
    """
    lowered = _words_to_digits(normalize(text))
    # "one zero zero one" becomes "1 0 0 1"; join adjacent digits back up so a
    # dictated order number looks the same as a typed one.
    collapsed = re.sub(r"(?<=\d)\s+(?=\d)", "", lowered)

    match = re.search(r"\bord\s*(\d{4})\b", collapsed)
    if match:
        return f"ORD-{match.group(1)}"
    match = re.search(r"\border(?:\s+(?:number|no|id))?\s+(\d{4})\b", collapsed)
    if match:
        return f"ORD-{match.group(1)}"
    # A bare four-digit number, but not something that is clearly a strength.
    match = re.search(r"(?<!\d)(\d{4})(?!\d)", collapsed)
    if match and not re.search(rf"{match.group(1)}\s*(?:mg|mcg|ml|iu)", collapsed):
        return f"ORD-{match.group(1)}"
    return None


def extract_address(text: str) -> str | None:
    """Take an address either from an explicit lead-in or as the whole reply."""
    match = re.search(r"\b(?:address is|deliver to|send it to|i live at|it.s at)\s+(.{5,120})", text, re.I)
    if match:
        return match.group(1).strip(" .")
    cleaned = text.strip(" .")
    # A bare reply to "what's the address?" - needs a house number or a comma
    # to look like an address rather than conversation.
    if len(cleaned) >= 8 and (re.search(r"\d", cleaned) or "," in cleaned):
        return cleaned
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


def clamp_text(text: str, limit: int | None = None) -> tuple[str, bool]:
    """
    Cut an utterance down to the length a phone turn can plausibly be.

    Returns `(text, was_truncated)`. The search below is O(tokens x formulary
    x name length): 40,000 characters of input took over four minutes on the
    development machine while holding the worker thread, which is a
    denial-of-service path on a public webhook. A real spoken turn is a
    sentence or two, so anything past the limit is discarded rather than
    processed.
    """
    limit = settings.MAX_QUERY_CHARS if limit is None else limit
    text = text or ""
    if len(text) <= limit:
        return text, False
    logger.warning("Utterance of %d chars truncated to %d.", len(text), limit)
    return text[:limit], True


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
    text, _ = clamp_text(text)
    tokens = normalize(text).split()
    if not tokens:
        return []
    # Second bound, on tokens rather than characters: a caller who says 60
    # words has said everything a pack of medicine needs, and this keeps the
    # cost of one turn flat no matter what arrives.
    if len(tokens) > settings.MAX_QUERY_TOKENS:
        logger.warning(
            "Utterance of %d tokens capped at %d for medicine extraction.",
            len(tokens),
            settings.MAX_QUERY_TOKENS,
        )
        tokens = tokens[: settings.MAX_QUERY_TOKENS]

    # Repeating a word repeats its n-gram; score each distinct phrase once and
    # reuse the result for every position it appears at.
    scored: dict[str, ExtractedMedicine | None] = {}
    candidates: list[tuple[int, int, ExtractedMedicine]] = []
    for start, end, gram in _ngrams(tokens):
        if gram not in scored:
            hits = kb.search(gram, limit=1, threshold=threshold, include_uses=False)
            scored[gram] = (
                ExtractedMedicine(gram, hits[0].medicine, hits[0].score, hits[0].match_type)
                if hits
                else None
            )
        found = scored[gram]
        if found is not None:
            candidates.append(
                (start, end, ExtractedMedicine(gram, found.medicine, found.score, found.match_type))
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
    """
    Run the full customer query -> knowledge base search pipeline.

    Oversized input is truncated rather than rejected, because this is the
    live phone path and a caller must always get an answer. The HTTP endpoint
    in app/main.py refuses it outright instead.
    """
    kb = kb or get_knowledge_base()
    text, truncated = clamp_text(text)

    intent = detect_intent(text)
    medicines = extract_medicines(text, kb=kb)
    quantity, quantity_issue = parse_quantity(text)

    result = QueryResult(
        text=text,
        intent=intent,
        medicines=medicines,
        truncated=truncated,
        quantity=quantity,
        quantity_issue=quantity_issue,
        strength=extract_strength(text),
        form=extract_form(text),
        needs_pharmacist=intent in (Intent.CLINICAL, Intent.PHARMACIST),
        person_name=extract_person_name(text),
        phone=extract_phone(text),
        fulfilment=extract_fulfilment(text),
        confirmation=extract_confirmation(text),
        order_reference=(
            extract_order_reference(text) if intent == Intent.ORDER_STATUS else None
        ),
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

# What the agent should say when a spoken pack count could not be accepted.
QUANTITY_ISSUE_NOTE = {
    QuantityIssue.NEGATIVE: (
        "The caller appeared to say a negative quantity. Do not guess a number. "
        "Ask how many packs they would like."
    ),
    QuantityIssue.ZERO: (
        "The caller appeared to ask for zero packs. Do not create an order. "
        "Ask how many packs they would like."
    ),
    QuantityIssue.TOO_MANY: (
        f"The caller asked for more than the {settings.MAX_ORDER_QUANTITY}-pack limit for a "
        "phone order. Say a pharmacist has to handle an order that large, and ask whether a "
        "smaller quantity would help in the meantime."
    ),
}

# The same messages as plain spoken sentences, for the no-LLM fallback path.
QUANTITY_ISSUE_REPLY = {
    QuantityIssue.NEGATIVE: "Sorry, I didn't catch that number. How many packs would you like?",
    QuantityIssue.ZERO: "Sorry, I didn't catch that number. How many packs would you like?",
    QuantityIssue.TOO_MANY: (
        f"That's more than the {settings.MAX_ORDER_QUANTITY} packs I can take over the phone. "
        "A pharmacist will need to arrange an order that size. Would a smaller quantity help for now?"
    ),
}


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
    if result.quantity_issue:
        lines.append(f"NOTE: {QUANTITY_ISSUE_NOTE[result.quantity_issue]}")
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
