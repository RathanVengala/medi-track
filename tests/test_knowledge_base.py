"""
Tests for the Week 2 knowledge base and query handler.

No API keys and no network needed. Run either way:

    python -m pytest tests/test_knowledge_base.py -q
    python -m tests.test_knowledge_base
"""

from __future__ import annotations

from app.knowledge_base import STOCK_LOW, STOCK_OUT, get_knowledge_base, phonetic_key
from app.stt import _whisper_prompt, is_probable_noise
from app.query_handler import (
    Intent,
    build_context,
    detect_intent,
    extract_medicines,
    extract_quantity,
    extract_strength,
    fallback_answer,
    handle_query,
)

kb = get_knowledge_base()


# --- knowledge base ---------------------------------------------------------


def test_knowledge_base_loads():
    assert len(kb.medicines) >= 25
    assert kb.pharmacy["name"]
    assert kb.pharmacy["hours"]["sunday"]


def test_exact_name_search():
    hits = kb.search("amoxicillin")
    assert hits and hits[0].medicine.brand_name == "Amoxil"
    assert hits[0].score >= 0.9


def test_brand_and_alias_search():
    assert kb.find("panadol").generic_name == "Paracetamol"
    assert kb.find("tylenol").brand_name == "Panadol"
    assert kb.find("albuterol").brand_name == "Ventolin"


def test_misheard_names_still_match():
    """STT mangles drug names; the phonetic index has to absorb that."""
    misheard = {
        "amoxycillin": "Amoxil",
        "amoxisillin": "Amoxil",
        "amoxy sillin": "Amoxil",
        "paracetemol": "Panadol",
        "ventalin": "Ventolin",
        "sitirizine": "Zyrtec",
        # Vowel-free phonetic keys made this collide with Lipitor's "storvas".
        "sitirizeen": "Zyrtec",
        "ibuprofen": "Brufen",
        "omeprazol": "Prilosec",
        "azithromiacin": "Zithromax",
        "sifroflaxacin": "Cipro",
        "insulin glargeen": "Lantus",
    }
    for spoken, expected in misheard.items():
        med = kb.find(spoken)
        assert med is not None, f"{spoken!r} matched nothing"
        assert med.brand_name == expected, f"{spoken!r} -> {med.brand_name}, expected {expected}"


def test_phonetic_key_folds_spelling_variants():
    assert phonetic_key("amoxycillin") == phonetic_key("amoxicillin")
    assert phonetic_key("Paracetamol") == phonetic_key("paracetemol")


def test_nonsense_is_rejected():
    assert kb.find("qwerty zxcvbn") is None
    assert kb.check_availability("blorbotrex").status == "not_found"


def test_availability_states():
    assert kb.check_availability("panadol").status == "in_stock"
    assert kb.check_availability("ventolin").status == STOCK_LOW
    assert kb.check_availability("augmentin").status == STOCK_OUT


def test_out_of_stock_offers_alternatives():
    result = kb.check_availability("augmentin")
    assert result.alternatives
    assert any(a.in_stock for a in result.alternatives)
    assert "alternative" in result.message.lower()


def test_symptom_search_finds_relevant_medicines():
    hits = kb.search_by_use("something for allergy and sneezing")
    assert hits
    assert any("Allergy" in h.medicine.category for h in hits)


# --- query handler ----------------------------------------------------------


def test_intent_detection():
    cases = {
        "Do you have amoxicillin in stock?": Intent.AVAILABILITY,
        "How much does panadol cost?": Intent.PRICE,
        "How many paracetamol should I take?": Intent.CLINICAL,
        "What time do you close today?": Intent.STORE_INFO,
        "Can you deliver it to my house?": Intent.DELIVERY,
        "I need a refill of my metformin": Intent.REFILL,
        "Hello there": Intent.GREETING,
        "That's all, goodbye": Intent.GOODBYE,
        # Info phrasings must not be read as an order...
        "Ok, I would like to know about Panadol": Intent.MEDICINE_INFO,
        "Tell me about Ventolin": Intent.MEDICINE_INFO,
        "What is amoxicillin used for": Intent.MEDICINE_INFO,
        # ...but the same opener with a quantity is a purchase request.
        "I would like two boxes of panadol": Intent.AVAILABILITY,
        # The loose "what is" patterns must not steal store questions.
        "What is your address": Intent.STORE_INFO,
    }
    for text, expected in cases.items():
        assert detect_intent(text) == expected, f"{text!r} -> {detect_intent(text)}"


def test_entity_extraction():
    result = handle_query("I need two boxes of amoxicillin 500 mg capsules")
    assert result.primary.brand_name == "Amoxil"
    assert result.quantity == 2
    assert result.strength == "500 mg"
    assert result.form == "capsule"


def test_extract_quantity_variants():
    assert extract_quantity("give me 3 packs of panadol") == 3
    assert extract_quantity("two bottles of cough syrup") == 2
    assert extract_quantity("do you have panadol") is None


def test_extract_strength_variants():
    assert extract_strength("ventolin 100 mcg") == "100 mcg"
    assert extract_strength("no numbers here") is None


def test_multiple_medicines_in_one_utterance():
    result = handle_query("Do you have panadol and cetirizine?")
    brands = {m.medicine.brand_name for m in result.medicines}
    assert brands == {"Panadol", "Zyrtec"}


def test_common_words_are_not_matched_as_medicines():
    """The n-gram scan must not turn filler words into drug names."""
    for text in (
        "Hello, how are you today?",
        "Are you open right now?",
        "Can I collect it tomorrow morning?",  # 'morning' vs Motrin
        "What time do you close on Sunday?",  # 'close' vs Losec
        "Do you sell that tablet here?",  # 'tablet' vs 'cold tablet'
    ):
        assert extract_medicines(text) == [], f"{text} -> {extract_medicines(text)}"


def test_multiword_aliases_still_resolve():
    """Blocking generic words must not break the aliases that contain them."""
    expected = {
        "cold tablet": "Dolo Cold",
        "cough syrup": "Benadryl Cough Formula",
        "insulin pen": "Lantus",
        "vitamin d": "D-Rise",
        "pain gel": "Voltaren Gel",
        "inhaler": "Ventolin",
    }
    for spoken, brand in expected.items():
        hits = extract_medicines(spoken)
        assert hits and hits[0].medicine.brand_name == brand, spoken


def test_greeting_prefix_does_not_mask_the_real_request():
    """'Hi, do you have X?' is an availability question, not a greeting."""
    result = handle_query("Hi, do you have amoxicillin in stock?")
    assert result.intent == Intent.AVAILABILITY
    assert result.primary.brand_name == "Amoxil"
    assert detect_intent("Good morning, how much is panadol?") == Intent.PRICE
    assert detect_intent("Hello there") == Intent.GREETING


def test_clinical_query_routes_to_pharmacist():
    result = handle_query("How much ibuprofen can I take with my blood pressure tablets?")
    assert result.needs_pharmacist
    assert "pharmacist" in fallback_answer(result).lower()


def test_context_block_is_grounded():
    context = build_context(handle_query("Is amoxicillin available?"))
    assert "Amoxil" in context
    assert "Prescription required: yes" in context
    assert "Price:" in context


def test_context_reports_no_match_instead_of_guessing():
    context = build_context(handle_query("Do you have blorbotrex?"))
    assert "Amoxil" not in context
    assert "No medicine matched" in context or "not found" in context.lower()


def test_fallback_answers_are_speakable():
    for text in (
        "Do you have amoxicillin?",
        "How much is ventolin?",
        "What time do you close on Sunday?",
        "Do you deliver?",
    ):
        answer = fallback_answer(handle_query(text))
        assert answer and len(answer) < 320
        assert "*" not in answer and "\n" not in answer


# --- speech-to-text noise filter --------------------------------------------


def test_whisper_hallucinations_are_rejected():
    """Whisper invents these from silence; they must never reach the agent."""
    hallucinations = [
        "ஆகும் குட்பை",  # Tamil
        "गुडबाई",  # Hindi
        "请不吝点赞 订阅 转发",  # Chinese subtitle spam
        "Thanks for watching!",
        "you",
        "[Music]",
        "",
        "   ",
    ]
    for text in hallucinations:
        assert is_probable_noise(text), f"not filtered: {text!r}"


def test_real_speech_survives_the_noise_filter():
    for text in (
        "Do you have amoxicillin in stock?",
        "goodbye",
        "bye",
        "no",
        "yes please",
        "Thank you",
        "OK",
    ):
        assert not is_probable_noise(text), f"wrongly filtered: {text!r}"


def test_whisper_prompt_stays_under_the_token_limit():
    prompt = _whisper_prompt()
    assert len(prompt) <= 720, len(prompt)  # ~180 tokens, limit is 224
    assert "Amoxil" in prompt


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
