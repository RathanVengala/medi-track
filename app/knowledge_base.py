"""
Pharmacy knowledge base (Week 2).

Loads the medicine formulary and store profile from `data/`, and exposes
fuzzy search + availability lookups that tolerate the kind of mangled
medicine names speech-to-text produces on a phone call
("amoxy sillin", "ventalin", "paracetemol").
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from app.config import settings

logger = logging.getLogger("meditrack.kb")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MEDICINES_FILE = DATA_DIR / "medicines.json"
PHARMACY_FILE = DATA_DIR / "pharmacy_info.json"

# A hit below this score is treated as "no match" rather than a wrong guess.
DEFAULT_THRESHOLD = settings.KB_MATCH_THRESHOLD

# Phonetic keys drop vowels, so short ones ("opn", "kvrt") score misleadingly
# high against each other. Only trust a phonetic comparison on longer keys, and
# discount it relative to a match on the literal spelling.
MIN_PHONETIC_LEN = 5
PHONETIC_FUZZY_WEIGHT = 0.92
# A genuine mis-transcription still shares most of its letters with the real
# name ("sitirizeen"/"cetirizine"). Without this floor, two unrelated names can
# collide purely on their consonant skeletons ("strsn" vs "storvas" -> "strvs").
MIN_LITERAL_FOR_PHONETIC = 0.5

STOCK_IN = "in_stock"
STOCK_LOW = "low_stock"
STOCK_OUT = "out_of_stock"


class KnowledgeBaseError(Exception):
    pass


# --------------------------------------------------------------------------
# Text normalisation
# --------------------------------------------------------------------------

# Ordered rewrites that fold the spellings STT commonly confuses onto a single
# canonical form. Applied to an already lowercased, letters-only string.
_PHONETIC_RULES: tuple[tuple[str, str], ...] = (
    ("ph", "f"),
    ("ch", "k"),
    ("ck", "k"),
    ("qu", "kw"),
    ("q", "k"),
    ("c", "k"),
    ("x", "ks"),
    ("z", "s"),
    ("y", "i"),
    ("w", "v"),
)


def normalize(text: str) -> str:
    """Lowercase, strip accents/punctuation, and collapse whitespace."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def phonetic_key(text: str) -> str:
    """
    Reduce a word to a rough sound-alike key.

    "amoxycillin", "amoxicillin" and "amoxisillin" all collapse to the same
    key, which is what makes the search survive a bad transcript.
    """
    key = re.sub(r"[^a-z]+", "", normalize(text))
    for src, dst in _PHONETIC_RULES:
        key = key.replace(src, dst)
    # Drop non-leading vowels, so "paracetamol" and "paracetemol" agree. The
    # first letter is kept because it carries most of the disambiguating weight.
    if key:
        key = key[0] + re.sub(r"[aeiou]+", "", key[1:])
    # Collapse doubled letters ("ll" -> "l") last, after the rewrites above.
    return re.sub(r"(.)\1+", r"\1", key)


def _ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------


@dataclass
class Medicine:
    id: str
    brand_name: str
    generic_name: str
    aliases: list[str]
    category: str
    form: str
    strength: str
    pack_size: str
    price: float
    stock_qty: int
    reorder_level: int
    prescription_required: bool
    common_uses: str
    storage: str
    manufacturer: str
    shelf_location: str
    substitutes: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict) -> "Medicine":
        return cls(
            id=raw["id"],
            brand_name=raw["brand_name"],
            generic_name=raw["generic_name"],
            aliases=list(raw.get("aliases", [])),
            category=raw.get("category", ""),
            form=raw.get("form", ""),
            strength=raw.get("strength", ""),
            pack_size=raw.get("pack_size", ""),
            price=float(raw.get("price", 0.0)),
            stock_qty=int(raw.get("stock_qty", 0)),
            reorder_level=int(raw.get("reorder_level", 0)),
            prescription_required=bool(raw.get("prescription_required", False)),
            common_uses=raw.get("common_uses", ""),
            storage=raw.get("storage", ""),
            manufacturer=raw.get("manufacturer", ""),
            shelf_location=raw.get("shelf_location", ""),
            substitutes=list(raw.get("substitutes", [])),
        )

    # -- derived views ----------------------------------------------------

    @property
    def names(self) -> list[str]:
        """Every string a caller might use to refer to this medicine."""
        return [self.brand_name, self.generic_name, *self.aliases]

    @property
    def display_name(self) -> str:
        if self.generic_name.lower() == self.brand_name.lower():
            return f"{self.brand_name} {self.strength}".strip()
        return f"{self.brand_name} ({self.generic_name}) {self.strength}".strip()

    @property
    def stock_status(self) -> str:
        if self.stock_qty <= 0:
            return STOCK_OUT
        if self.stock_qty <= self.reorder_level:
            return STOCK_LOW
        return STOCK_IN

    @property
    def in_stock(self) -> bool:
        return self.stock_qty > 0

    def stock_sentence(self) -> str:
        """A short, speakable stock statement."""
        if self.stock_status == STOCK_OUT:
            return f"{self.brand_name} is currently out of stock."
        if self.stock_status == STOCK_LOW:
            return f"{self.brand_name} is running low, only {self.stock_qty} packs left."
        return f"{self.brand_name} is in stock, we have {self.stock_qty} packs."

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "brand_name": self.brand_name,
            "generic_name": self.generic_name,
            "category": self.category,
            "form": self.form,
            "strength": self.strength,
            "pack_size": self.pack_size,
            "price": self.price,
            "stock_qty": self.stock_qty,
            "stock_status": self.stock_status,
            "prescription_required": self.prescription_required,
            "common_uses": self.common_uses,
            "shelf_location": self.shelf_location,
        }


@dataclass
class SearchHit:
    medicine: Medicine
    score: float
    matched_on: str
    match_type: str  # exact | phonetic | fuzzy | use | category


@dataclass
class AvailabilityResult:
    query: str
    medicine: Medicine | None
    status: str  # in_stock | low_stock | out_of_stock | not_found
    message: str
    alternatives: list[Medicine] = field(default_factory=list)


# --------------------------------------------------------------------------
# Knowledge base
# --------------------------------------------------------------------------


class PharmacyKnowledgeBase:
    """In-memory index over the medicine formulary and store profile."""

    def __init__(
        self,
        medicines_file: Path | str = MEDICINES_FILE,
        pharmacy_file: Path | str = PHARMACY_FILE,
    ) -> None:
        self.medicines_file = Path(medicines_file)
        self.pharmacy_file = Path(pharmacy_file)
        self.medicines: list[Medicine] = []
        self.pharmacy: dict = {}
        self._by_id: dict[str, Medicine] = {}
        # name index: normalized name -> (medicine, original name)
        self._name_index: dict[str, tuple[Medicine, str]] = {}
        self._phonetic_index: dict[str, list[tuple[Medicine, str]]] = {}
        self.load()

    # -- loading ----------------------------------------------------------

    def load(self) -> None:
        self.medicines = [
            Medicine.from_dict(row) for row in self._read_json(self.medicines_file)["medicines"]
        ]
        self.pharmacy = self._read_json(self.pharmacy_file)

        self._by_id = {m.id: m for m in self.medicines}
        self._name_index.clear()
        self._phonetic_index.clear()

        for med in self.medicines:
            for name in med.names:
                norm = normalize(name)
                if norm:
                    self._name_index.setdefault(norm, (med, name))
                self._phonetic_index.setdefault(phonetic_key(name), []).append((med, name))

        logger.info(
            "Knowledge base loaded: %d medicines, %d searchable names.",
            len(self.medicines),
            len(self._name_index),
        )

    @staticmethod
    def _read_json(path: Path) -> dict:
        if not path.exists():
            raise KnowledgeBaseError(f"Knowledge base file not found: {path}")
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as exc:
            raise KnowledgeBaseError(f"Invalid JSON in {path}: {exc}") from exc

    # -- lookups ----------------------------------------------------------

    def get(self, medicine_id: str) -> Medicine | None:
        return self._by_id.get(medicine_id)

    def all_categories(self) -> list[str]:
        return sorted({m.category for m in self.medicines if m.category})

    # -- name search ------------------------------------------------------

    def _score_name(self, query_norm: str, query_phon: str, name: str) -> tuple[float, str]:
        """Score a query against one of a medicine's names."""
        name_norm = normalize(name)
        name_phon = phonetic_key(name)

        if query_norm == name_norm:
            return 1.0, "exact"
        # Caller said "amoxicillin five hundred" -> the name is inside the query.
        if len(name_norm) >= 4 and name_norm in query_norm:
            return 0.9, "exact"
        if len(query_norm) >= 4 and query_norm in name_norm:
            return 0.86, "exact"

        literal = _ratio(query_norm, name_norm)
        phonetic_usable = min(len(query_phon), len(name_phon)) >= MIN_PHONETIC_LEN
        if phonetic_usable:
            if query_phon == name_phon:
                return 0.93, "phonetic"
            if name_phon.startswith(query_phon) or query_phon.startswith(name_phon):
                return 0.84, "phonetic"

        sounded = (
            _ratio(query_phon, name_phon) * PHONETIC_FUZZY_WEIGHT
            if phonetic_usable and literal >= MIN_LITERAL_FOR_PHONETIC
            else 0.0
        )
        if sounded > literal:
            return sounded, "phonetic"
        return literal, "fuzzy"

    def search(
        self,
        query: str,
        limit: int = 5,
        threshold: float = DEFAULT_THRESHOLD,
        include_uses: bool = True,
    ) -> list[SearchHit]:
        """
        Find medicines matching `query`.

        Tries medicine names first (brand, generic, aliases). If nothing
        matches by name and `include_uses` is set, falls back to matching
        against category and common-use text so "something for a headache"
        still returns useful results.
        """
        query_norm = normalize(query)
        if not query_norm:
            return []
        query_phon = phonetic_key(query)

        hits: list[SearchHit] = []
        for med in self.medicines:
            best_score, best_name, best_type = 0.0, "", "fuzzy"
            for name in med.names:
                score, match_type = self._score_name(query_norm, query_phon, name)
                if score > best_score:
                    best_score, best_name, best_type = score, name, match_type
            if best_score >= threshold:
                hits.append(SearchHit(med, round(best_score, 3), best_name, best_type))

        if not hits and include_uses:
            hits = self.search_by_use(query, limit=limit)

        hits.sort(key=lambda h: (-h.score, h.medicine.brand_name))
        return hits[:limit]

    def search_by_use(self, query: str, limit: int = 5) -> list[SearchHit]:
        """Match a symptom or condition against category + common-use text."""
        query_norm = normalize(query)
        tokens = {t for t in query_norm.split() if len(t) > 3 and t not in _USE_STOPWORDS}
        if not tokens:
            return []

        hits: list[SearchHit] = []
        for med in self.medicines:
            haystack = normalize(f"{med.category} {med.common_uses}")
            haystack_tokens = set(haystack.split())
            overlap = {t for t in tokens if t in haystack_tokens or t[:-1] in haystack_tokens}
            if not overlap:
                continue
            score = 0.6 + 0.1 * min(len(overlap), 3)
            hits.append(
                SearchHit(med, round(min(score, 0.88), 3), med.common_uses or med.category, "use")
            )

        hits.sort(key=lambda h: (-h.score, -h.medicine.stock_qty))
        return hits[:limit]

    def find(self, name: str, threshold: float = DEFAULT_THRESHOLD) -> Medicine | None:
        """Best single name match, or None."""
        hits = self.search(name, limit=1, threshold=threshold, include_uses=False)
        return hits[0].medicine if hits else None

    # -- availability -----------------------------------------------------

    def check_availability(self, name: str) -> AvailabilityResult:
        """Resolve a spoken medicine name to a concrete stock answer."""
        hits = self.search(name, limit=3, include_uses=False)
        if not hits:
            suggestions = self.search_by_use(name, limit=3)
            if suggestions:
                names = ", ".join(h.medicine.brand_name for h in suggestions)
                return AvailabilityResult(
                    query=name,
                    medicine=None,
                    status="not_found",
                    message=(
                        f"No medicine named '{name}' in the formulary, but these are "
                        f"stocked for that kind of problem: {names}."
                    ),
                    alternatives=[h.medicine for h in suggestions],
                )
            return AvailabilityResult(
                query=name,
                medicine=None,
                status="not_found",
                message=f"'{name}' was not found in the pharmacy formulary.",
            )

        med = hits[0].medicine
        alternatives = self.alternatives_for(med)
        message = med.stock_sentence()
        if med.stock_status == STOCK_OUT and alternatives:
            in_stock_alts = [a.brand_name for a in alternatives if a.in_stock]
            if in_stock_alts:
                message += " Available alternatives: " + ", ".join(in_stock_alts) + "."

        return AvailabilityResult(
            query=name,
            medicine=med,
            status=med.stock_status,
            message=message,
            alternatives=alternatives,
        )

    def alternatives_for(self, medicine: Medicine) -> list[Medicine]:
        """Listed substitutes, plus anything sharing the same generic name."""
        found: dict[str, Medicine] = {}
        for sub_id in medicine.substitutes:
            sub = self._by_id.get(sub_id)
            if sub:
                found[sub.id] = sub
        generic = normalize(medicine.generic_name)
        for other in self.medicines:
            if other.id != medicine.id and normalize(other.generic_name) == generic:
                found[other.id] = other
        return list(found.values())

    def low_stock(self) -> list[Medicine]:
        return [m for m in self.medicines if m.stock_status in (STOCK_LOW, STOCK_OUT)]

    # -- store profile ----------------------------------------------------

    @property
    def currency_symbol(self) -> str:
        return self.pharmacy.get("currency_symbol", "$")

    def format_price(self, amount: float) -> str:
        return f"{self.currency_symbol}{amount:,.2f}"

    def price_text(self, medicine: Medicine) -> str:
        return f"{self.format_price(medicine.price)} for {medicine.pack_size}"

    def hours_today(self, weekday: str) -> str | None:
        return self.pharmacy.get("hours", {}).get(weekday.lower())

    def store_summary(self) -> str:
        """Compact store profile used to ground the LLM's system prompt."""
        p = self.pharmacy
        hours = p.get("hours", {})
        weekdays = hours.get("monday", "")
        lines = [
            f"Pharmacy: {p.get('name')} - {p.get('address')} - phone {p.get('phone')}",
            f"Hours: Mon-Fri {weekdays}; Saturday {hours.get('saturday')}; Sunday {hours.get('sunday')}",
            "Services: " + ", ".join(p.get("services", [])),
        ]
        delivery = p.get("delivery", {})
        if delivery.get("available"):
            lines.append(
                f"Delivery: within {delivery.get('radius_miles')} miles, "
                f"{self.format_price(delivery.get('fee', 0))} fee, free above "
                f"{self.format_price(delivery.get('free_above', 0))}, "
                f"same-day cutoff {delivery.get('same_day_cutoff')}"
            )
        for key, value in p.get("policies", {}).items():
            lines.append(f"Policy ({key}): {value}")
        lines.append(f"Pharmacist on duty: {p.get('pharmacist_on_duty')}")
        return "\n".join(lines)


_USE_STOPWORDS = {
    "have",
    "need",
    "want",
    "some",
    "something",
    "anything",
    "there",
    "give",
    "take",
    "tell",
    "with",
    "that",
    "this",
    "from",
    "your",
    "please",
    "about",
    "medicine",
    "medication",
    "tablet",
    "tablets",
    "does",
    "stock",
}


# --------------------------------------------------------------------------
# Module-level singleton
# --------------------------------------------------------------------------

_kb: PharmacyKnowledgeBase | None = None


def get_knowledge_base() -> PharmacyKnowledgeBase:
    """Load the knowledge base once and reuse it for the process lifetime."""
    global _kb
    if _kb is None:
        _kb = PharmacyKnowledgeBase()
    return _kb


def search_medicines(query: str, limit: int = 5) -> list[SearchHit]:
    return get_knowledge_base().search(query, limit=limit)


def check_availability(name: str) -> AvailabilityResult:
    return get_knowledge_base().check_availability(name)
