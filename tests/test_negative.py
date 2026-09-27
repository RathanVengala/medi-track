"""
Negative, boundary, security, concurrency and recovery tests (Week 4).

The other suites test the happy path: the right medicine, a sane quantity, a
cooperative caller, a readable data file. They all passed while every failure
below was live, which is the real lesson from the Week 3 review - a green suite
that only ever asks "does the good case work?" measures nothing about what
happens when it doesn't.

Every test here pins one specific defect found in that review, named in its
docstring, so a regression is recognisable rather than just red.

    python -m tests.test_negative     # standalone
    python -m tests.run_all           # with everything else

No API key or network required.
"""

from __future__ import annotations

import json
import tempfile
import threading
import time
from pathlib import Path

from app.config import settings
from app.escalation import EscalationQueue, EscalationReason
from app.knowledge_base import KnowledgeBaseError, PharmacyKnowledgeBase, get_knowledge_base
from app.order_flow import OrderDraft, Slot, build_order_context, update_draft
from app.orders import (
    Customer,
    InvalidQuantity,
    OrderError,
    OrderItem,
    OrderStatus,
    OrderStore,
    validate_quantity,
)
from app.query_handler import (
    Intent,
    QuantityIssue,
    detect_intent,
    handle_query,
    parse_quantity,
)
from app.storage import load_json
from app.stt import is_probable_noise, transcribe_audio

kb = get_knowledge_base()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _tmp() -> Path:
    return Path(tempfile.mkdtemp())


def _fresh_store() -> OrderStore:
    return OrderStore(_tmp() / "orders.json")


def _any_medicine():
    """A well-stocked, over-the-counter medicine, for arithmetic tests."""
    return next(m for m in kb.medicines if m.stock_qty >= 20 and not m.prescription_required)


def _converse(turns: list[str], store: OrderStore, queue: EscalationQueue):
    """Feed turns through the order flow exactly as the live path does."""
    draft = OrderDraft(call_id="NEGATIVE")
    flow = None
    for turn in turns:
        flow = update_draft(draft, handle_query(turn), store=store, queue=queue)
        build_order_context(flow)  # sets draft.asked_for, as app/llm.py does
    return draft, flow


class _Setting:
    """Temporarily override a value on the settings singleton."""

    def __init__(self, **overrides) -> None:
        self.overrides = overrides
        self.previous: dict = {}

    def __enter__(self):
        for name, value in self.overrides.items():
            self.previous[name] = getattr(settings, name)
            setattr(settings, name, value)
        return settings

    def __exit__(self, *exc) -> None:
        for name, value in self.previous.items():
            setattr(settings, name, value)


# --------------------------------------------------------------------------
# F-01 / F-02 / F-04 - quantity bounds at the persistence boundary
# --------------------------------------------------------------------------


def test_zero_quantity_order_is_refused():
    """F-01: an order for zero packs was accepted and persisted with a total of 0.00."""
    store = _fresh_store()
    try:
        order = store.create_order(
            Customer("Test Caller", "9876543210"),
            [OrderItem.from_medicine(_any_medicine(), 0)],
        )
    except InvalidQuantity:
        assert store.list_orders() == [], "nothing should have been written"
        return
    raise AssertionError(f"zero-pack order was accepted as {order.id} (total {order.total})")


def test_negative_quantity_order_is_refused():
    """F-02: minus one pack was accepted and persisted with a NEGATIVE total."""
    store = _fresh_store()
    try:
        order = store.create_order(
            Customer("Test Caller", "9876543210"),
            [OrderItem.from_medicine(_any_medicine(), -1)],
        )
    except InvalidQuantity:
        assert store.list_orders() == [], "nothing should have been written"
        return
    raise AssertionError(f"negative order was accepted as {order.id} (total {order.total})")


def test_quantity_above_the_phone_limit_is_refused():
    """F-04: 999 packs was bounded only by whatever happened to be in stock."""
    store = _fresh_store()
    over = settings.MAX_ORDER_QUANTITY + 1
    try:
        store.create_order(
            Customer("Test Caller", "9876543210"),
            [OrderItem.from_medicine(_any_medicine(), over)],
        )
    except InvalidQuantity:
        return
    raise AssertionError(f"{over} packs was accepted, above the {settings.MAX_ORDER_QUANTITY} limit")


def test_non_integer_quantities_are_refused():
    """A float or a bool must never be read as a pack count."""
    for bad in (1.5, "2", None, True):
        try:
            validate_quantity(bad)
        except InvalidQuantity:
            continue
        raise AssertionError(f"{bad!r} was accepted as a quantity")


def test_valid_quantities_are_still_accepted():
    """Guard against over-correcting: the ordinary range must keep working."""
    for good in (1, 2, settings.MAX_ORDER_QUANTITY):
        assert validate_quantity(good) == good

    store = _fresh_store()
    order = store.create_order(
        Customer("Test Caller", "9876543210"),
        [OrderItem.from_medicine(_any_medicine(), 2)],
    )
    assert order.total > 0, "a valid order must still price correctly"


# --------------------------------------------------------------------------
# F-03 / F-04 - quantity bounds at the transcript boundary
# --------------------------------------------------------------------------


def test_spoken_negative_quantity_is_not_silently_made_positive():
    """F-03: "give me -5 tablets" returned 5; normalisation had eaten the minus."""
    quantity, issue = parse_quantity("give me -5 tablets")
    assert quantity is None, f"expected no quantity, got {quantity}"
    assert issue is QuantityIssue.NEGATIVE, f"expected a negative-quantity issue, got {issue}"

    for phrase in ("I want minus two packs", "order negative 3 boxes"):
        quantity, issue = parse_quantity(phrase)
        assert quantity is None, f"{phrase!r} -> {quantity}"
        assert issue is QuantityIssue.NEGATIVE


def test_hyphenated_names_and_phone_numbers_are_not_read_as_negatives():
    """The minus-sign guard must not fire on ordinary hyphens."""
    assert parse_quantity("two packs of co-amoxiclav") == (2, None)
    quantity, issue = parse_quantity("my number is 98765-43210")
    assert issue is None, f"a hyphenated phone number was read as a negative quantity ({issue})"


def test_spoken_quantity_above_the_limit_is_refused():
    """F-04: "999 packs please" returned 999 and relied on stock to stop it."""
    quantity, issue = parse_quantity("999 packs please")
    assert quantity is None, f"expected no quantity, got {quantity}"
    assert issue is QuantityIssue.TOO_MANY

    quantity, issue = parse_quantity("I need 0 packs")
    assert quantity is None and issue is QuantityIssue.ZERO


def test_ordinary_spoken_quantities_still_parse():
    """Regression guard for the cases that already worked."""
    assert parse_quantity("I need two boxes of amoxicillin") == (2, None)
    assert parse_quantity("give me 3 packs of panadol") == (3, None)
    assert parse_quantity("do you have panadol") == (None, None)


def test_order_flow_reasks_rather_than_defaulting_to_one_pack():
    """
    A refused quantity must not quietly become the default of one pack.

    "I'd like to order 999 packs of Panadol" previously either created a
    999-pack order or, once the count was rejected, would have fallen back to
    `result.quantity or 1` and ordered a single pack the caller never asked for.
    """
    tmp = _tmp()
    store, queue = OrderStore(tmp / "o.json"), EscalationQueue(tmp / "e.json")
    medicine = _any_medicine()

    draft, flow = _converse([f"I would like to order 999 packs of {medicine.brand_name}"], store, queue)

    assert draft.items, "the medicine should still be recognised"
    assert flow.quantity_issue is QuantityIssue.TOO_MANY
    assert draft.next_slot() is Slot.QUANTITY, (
        f"expected the agent to re-ask the quantity, it asked for {draft.next_slot().value}"
    )
    assert draft.order_id is None, "no order may be created from a refused quantity"

    # ...and a sensible answer on the next turn clears it.
    flow = update_draft(draft, handle_query("two packs"), store=store, queue=queue)
    build_order_context(flow)
    assert draft.items[0].quantity == 2
    assert draft.next_slot() is not Slot.QUANTITY


# --------------------------------------------------------------------------
# F-05 / F-06 / F-07 - intent routing
# --------------------------------------------------------------------------


def test_drug_interaction_questions_route_to_the_pharmacist():
    """F-05: "Can I mix amoxicillin with paracetamol?" was UNKNOWN, bypassing the clinical gate."""
    questions = [
        "Can I mix amoxicillin with paracetamol?",
        "Can I mix these two medicines?",
        "Is it safe to take Panadol and Brufen together?",
        "Can I combine Zyrtec with Claritin?",
        "Will amoxicillin react with my blood pressure tablets?",
        "I am allergic to penicillin, can I take this?",
    ]
    failures = [q for q in questions if detect_intent(q) is not Intent.CLINICAL]
    assert not failures, f"not routed as clinical: {failures}"

    # Routing is only half of it - the safety flag is what stops the AI answering.
    for question in questions:
        assert handle_query(question).needs_pharmacist, f"{question!r} did not reach a pharmacist"


def test_store_policy_questions_are_not_answered_from_the_formulary():
    """F-06: "What is your return policy?" was classified MEDICINE_INFO."""
    for question in (
        "What is your return policy?",
        "What's your refund policy",
        "Can I return an item",
        "Do you accept insurance cards",
    ):
        intent = detect_intent(question)
        assert intent is Intent.STORE_INFO, f"{question!r} -> {intent.value}"


def test_requests_for_a_person_reach_the_escalation_route():
    """F-07: "I need human help" was classified AVAILABILITY by the "i need" purchase rule."""
    for question in (
        "I need human help",
        "I want to talk to a real person",
        "Get me a pharmacist",
        "Can I speak to customer service",
        "I don't want a bot, I want a human",
    ):
        intent = detect_intent(question)
        assert intent in (Intent.PHARMACIST, Intent.CLINICAL), f"{question!r} -> {intent.value}"
        assert handle_query(question).needs_pharmacist, f"{question!r} did not escalate"


def test_ordinary_requests_are_not_dragged_into_escalation():
    """Guard against over-correcting the routing above."""
    assert detect_intent("Do you have amoxicillin in stock?") is Intent.AVAILABILITY
    assert detect_intent("I need two boxes of panadol") is Intent.AVAILABILITY
    assert detect_intent("What time do you close on Sunday") is Intent.STORE_INFO
    assert detect_intent("I would like to order two packs of panadol") is Intent.ORDER


# --------------------------------------------------------------------------
# F-08 - input length limits
# --------------------------------------------------------------------------


def test_oversized_utterance_is_bounded_and_fast():
    """
    F-08: 40,000 characters held the worker thread for minutes.

    Measured at 276 seconds on the development machine - a denial-of-service
    path on a public webhook, reachable by anyone who could post to it.
    """
    started = time.perf_counter()
    result = handle_query("paracetamol " * 4_000)
    elapsed = time.perf_counter() - started

    assert elapsed < 2.0, f"a 48,000-character utterance took {elapsed:.1f}s"
    assert result.truncated, "the utterance should be marked as truncated"
    assert len(result.text) <= settings.MAX_QUERY_CHARS


def test_a_long_utterance_still_finds_the_medicine():
    """Truncation must not break a merely chatty caller."""
    chatty = (
        "Hello there, sorry to bother you on a busy afternoon, I was wondering "
        "whether you might happen to have any amoxicillin in stock today"
    )
    assert len(chatty) < settings.MAX_QUERY_CHARS
    result = handle_query(chatty)
    assert result.matched, "a long but normal sentence must still resolve the medicine"


# --------------------------------------------------------------------------
# F-09 - startup with a missing or broken formulary
# --------------------------------------------------------------------------


def test_missing_formulary_degrades_instead_of_crashing():
    """F-09: the app crashed at import time; the module could not even be loaded."""
    missing = _tmp()
    broken = PharmacyKnowledgeBase(missing / "medicines.json", missing / "pharmacy_info.json")

    assert not broken.is_healthy
    assert broken.load_error and "medicines.json" in broken.load_error, (
        f"the error must name the missing file, got: {broken.load_error}"
    )
    # Still answers, rather than raising, so the API can return a clean 503.
    assert broken.search("amoxicillin") == []
    assert broken.find("amoxicillin") is None
    assert broken.check_availability("amoxicillin").status == "not_found"


def test_corrupt_formulary_degrades_instead_of_crashing():
    """A hand-edited medicines.json must not take the process down either."""
    tmp = _tmp()
    (tmp / "medicines.json").write_text("{not json at all", encoding="utf-8")
    (tmp / "pharmacy_info.json").write_text("{}", encoding="utf-8")

    broken = PharmacyKnowledgeBase(tmp / "medicines.json", tmp / "pharmacy_info.json")
    assert not broken.is_healthy
    assert "Invalid JSON" in (broken.load_error or "")


def test_strict_mode_still_raises_for_callers_that_want_it():
    """Degrading is the default, not the only option."""
    missing = _tmp()
    try:
        PharmacyKnowledgeBase(missing / "a.json", missing / "b.json", strict=True)
    except KnowledgeBaseError:
        return
    raise AssertionError("strict=True should raise on a missing formulary")


def test_the_system_prompt_is_not_built_at_import_time():
    """
    F-09 root cause: `SYSTEM_PROMPT = build_system_prompt()` ran on import, so
    importing app.llm read data/medicines.json - and a missing file aborted the
    process before any error handling could report it.
    """
    import app.llm

    assert "SYSTEM_PROMPT" not in vars(app.llm), (
        "SYSTEM_PROMPT is being built at import time again"
    )
    # The name must still resolve, for the Week 1 code that imports it.
    assert isinstance(app.llm.SYSTEM_PROMPT, str) and app.llm.SYSTEM_PROMPT


# --------------------------------------------------------------------------
# F-10 / F-11 - authentication and webhook verification
# --------------------------------------------------------------------------


def _client():
    from fastapi.testclient import TestClient

    from app.main import app as fastapi_app

    return TestClient(fastapi_app)


PROTECTED_ENDPOINTS = (
    ("GET", "/orders"),
    ("GET", "/orders/ORD-1001"),
    ("GET", "/escalations"),
    ("GET", "/kb/medicines"),
    ("GET", "/kb/search?q=panadol"),
    ("GET", "/kb/availability?name=panadol"),
    ("GET", "/inventory/check?name=panadol"),
)


def test_staff_endpoints_reject_unauthenticated_callers():
    """F-10: all 13 endpoints were open; the order book was readable by anyone."""
    with _Setting(ADMIN_API_KEY="test-secret-key", DISABLE_AUTH=False):
        client = _client()
        open_endpoints = []
        for method, path in PROTECTED_ENDPOINTS:
            response = client.request(method, path)
            if response.status_code != 401:
                open_endpoints.append(f"{method} {path} -> {response.status_code}")
        assert not open_endpoints, f"served without credentials: {open_endpoints}"

        response = client.post("/query/analyze", data={"text": "do you have panadol"})
        assert response.status_code == 401


def test_staff_endpoints_serve_a_correct_key():
    """The lock has to open for the pharmacist, not only close on strangers."""
    with _Setting(ADMIN_API_KEY="test-secret-key", DISABLE_AUTH=False):
        client = _client()
        response = client.get("/orders", headers={"X-API-Key": "test-secret-key"})
        assert response.status_code == 200, response.text

        response = client.get("/orders", headers={"X-API-Key": "wrong-key"})
        assert response.status_code == 401


def test_endpoints_fail_closed_when_no_key_is_configured():
    """An unset secret must refuse to serve, never silently open the API."""
    with _Setting(ADMIN_API_KEY="", DISABLE_AUTH=False):
        response = _client().get("/orders")
        assert response.status_code == 503, response.text
        assert "ADMIN_API_KEY" in response.text


def test_health_stays_open_and_reports_configuration_problems():
    """A monitor should not need a secret, and should see what is wrong."""
    with _Setting(ADMIN_API_KEY="", DISABLE_AUTH=False):
        response = _client().get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] in ("ok", "degraded")
        assert any("ADMIN_API_KEY" in w for w in body["warnings"])


def test_query_endpoint_rejects_an_oversized_body():
    """F-08 at the HTTP boundary: reject, rather than truncate, for an API client."""
    with _Setting(ADMIN_API_KEY="test-secret-key", DISABLE_AUTH=False):
        response = _client().post(
            "/query/analyze",
            data={"text": "a" * (settings.MAX_QUERY_CHARS + 1)},
            headers={"X-API-Key": "test-secret-key"},
        )
        assert response.status_code == 413, response.text


def test_voice_webhooks_reject_unsigned_requests():
    """F-11: anyone who found the public URL could inject a fake call."""
    with _Setting(
        VALIDATE_TWILIO_SIGNATURE=True,
        TWILIO_AUTH_TOKEN="test-twilio-token",
        PUBLIC_BASE_URL="https://example.test",
    ):
        client = _client()
        for path in ("/voice/incoming", "/voice/process"):
            response = client.post(path, data={"CallSid": "CA-forged"})
            assert response.status_code == 403, f"{path} -> {response.status_code}"

            response = client.post(
                path,
                data={"CallSid": "CA-forged"},
                headers={"X-Twilio-Signature": "obviously-not-valid"},
            )
            assert response.status_code == 403, f"{path} (bad signature) -> {response.status_code}"


def test_voice_webhooks_accept_a_correctly_signed_request():
    """Verification must not break the real Twilio call path."""
    from twilio.request_validator import RequestValidator

    token = "test-twilio-token"
    base = "https://example.test"
    with _Setting(
        VALIDATE_TWILIO_SIGNATURE=True, TWILIO_AUTH_TOKEN=token, PUBLIC_BASE_URL=base
    ):
        params = {"CallSid": "CA-genuine-call"}
        signature = RequestValidator(token).compute_signature(f"{base}/voice/incoming", params)

        response = _client().post(
            "/voice/incoming", data=params, headers={"X-Twilio-Signature": signature}
        )
        assert response.status_code == 200, response.text
        assert "<Response>" in response.text


def test_voice_webhooks_fail_closed_without_a_twilio_token():
    """Signature checking on with no token must refuse, not wave requests through."""
    with _Setting(VALIDATE_TWILIO_SIGNATURE=True, TWILIO_AUTH_TOKEN=""):
        response = _client().post("/voice/incoming", data={"CallSid": "CA-1"})
        assert response.status_code == 503, response.text


# --------------------------------------------------------------------------
# F-12 / P-21 - call session lifecycle
# --------------------------------------------------------------------------


def test_sessions_are_evicted_once_they_go_idle():
    """F-12: a Twilio callback that never arrives left the session there forever."""
    from app.conversation import active_session_count, get_session, reset_sessions

    reset_sessions()
    try:
        with _Setting(SESSION_TTL_SECONDS=0):
            get_session("CA-abandoned")
            time.sleep(0.02)  # older than a zero-second TTL
            assert active_session_count() == 0, "an idle session was not evicted"
    finally:
        reset_sessions()


def test_session_count_is_capped():
    """The TTL alone cannot stop a burst inside one TTL window."""
    from app.conversation import active_session_count, get_session, reset_sessions

    reset_sessions()
    try:
        with _Setting(MAX_ACTIVE_SESSIONS=5, SESSION_TTL_SECONDS=3600):
            for index in range(25):
                get_session(f"CA-{index}")
            assert active_session_count() <= 5, "the session cap was not enforced"
    finally:
        reset_sessions()


def test_a_live_session_survives_across_turns():
    """P-21: the conversation has to continue across turns, not restart each time."""
    from app.conversation import end_session, get_session, reset_sessions

    reset_sessions()
    try:
        with _Setting(SESSION_TTL_SECONDS=3600, MAX_ACTIVE_SESSIONS=100):
            first = get_session("CA-live")
            first.add_user_turn("do you have panadol")
            second = get_session("CA-live")
            assert second is first, "each turn started a new session"
            assert len(second.history) == 1, "the history was lost between turns"

            end_session("CA-live")
            assert get_session("CA-live") is not first, "the session outlived the call"
    finally:
        reset_sessions()


# --------------------------------------------------------------------------
# F-13 - silence on the line
# --------------------------------------------------------------------------


def test_transcription_of_silence_returns_empty_rather_than_raising():
    """
    F-13: the fallback recogniser raised an uncaught OSError/TimeoutError on
    silence and the session died mid-call.
    """
    for audio in (b"", b"not a wav file at all", b"RIFF\x00\x00\x00\x00WAVEfmt "):
        text = transcribe_audio(audio, filename="silence.wav")
        assert text == "", f"expected an empty transcript, got {text!r}"


def test_an_empty_transcript_is_treated_as_noise():
    """The empty string has to route to a re-prompt, which is what keeps the call alive."""
    assert is_probable_noise("")
    assert is_probable_noise("   ")


# --------------------------------------------------------------------------
# F-14 - concurrent writers
# --------------------------------------------------------------------------


def test_a_second_worker_cannot_oversell_the_last_packs():
    """
    F-14: threading.Lock is process-local, so two uvicorn workers each held
    their own lock, read the same file and both sold the last pack.

    Two OrderStore instances on one file are exactly that situation: each has
    its own lock and its own stale in-memory copy.
    """
    tmp = _tmp()
    path = tmp / "orders.json"
    worker_a, worker_b = OrderStore(path), OrderStore(path)

    medicine = _any_medicine()
    available = worker_a.check_stock(medicine, 1).available

    # This test is about locking, not about the pack limit, so the limit is
    # lifted far enough for one order to take the whole shelf.
    with _Setting(MAX_ORDER_QUANTITY=available + 1):
        worker_a.create_order(
            Customer("First Caller", "9000000001"),
            [OrderItem.from_medicine(medicine, available)],
        )

        # worker_b's in-memory copy predates that order. It must notice anyway.
        try:
            worker_b.create_order(
                Customer("Second Caller", "9000000002"),
                [OrderItem.from_medicine(medicine, 1)],
            )
        except OrderError:
            pass
        else:
            raise AssertionError("the second worker oversold stock the first had reserved")

    # The first worker's order must also have survived the second one's save.
    on_disk = load_json(path).get("orders", [])
    assert len(on_disk) == 1, f"expected the first order to survive, found {len(on_disk)}"
    assert on_disk[0]["customer"]["name"] == "First Caller"


def test_concurrent_writers_do_not_lose_orders():
    """Parallel writes must all land; a lost update is a lost customer."""
    tmp = _tmp()
    path = tmp / "orders.json"
    medicine = _any_medicine()
    workers = 6

    start = threading.Barrier(workers)
    errors: list[Exception] = []

    def place(index: int) -> None:
        store = OrderStore(path)  # a separate "worker", with its own lock
        try:
            start.wait(timeout=10)
            store.create_order(
                Customer(f"Caller {index}", f"90000000{index:02d}"),
                [OrderItem.from_medicine(medicine, 1)],
            )
        except Exception as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=place, args=(i,)) for i in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors, f"concurrent writes failed: {errors}"
    on_disk = load_json(path).get("orders", [])
    assert len(on_disk) == workers, f"expected {workers} orders on disk, found {len(on_disk)}"
    assert len({o["id"] for o in on_disk}) == workers, "order ids collided between workers"


# --------------------------------------------------------------------------
# F-15 - recovery from a damaged store
# --------------------------------------------------------------------------


def test_a_corrupt_orders_file_is_recovered_from_the_backup():
    """F-15: a corrupt orders.json crashed on load, with no backup and no recovery path."""
    tmp = _tmp()
    path = tmp / "orders.json"

    store = OrderStore(path)
    order = store.create_order(
        Customer("Backup Caller", "9876543210"),
        [OrderItem.from_medicine(_any_medicine(), 1)],
    )
    # A second write, so the .bak holds a state that still contains the order.
    store.update_status(order.id, OrderStatus.READY_FOR_PICKUP)

    path.write_text('{"orders": [ truncated mid-writ', encoding="utf-8")

    recovered = OrderStore(path)  # must not raise
    assert recovered.get(order.id) is not None, "the order was not recovered from the backup"
    assert list(tmp.glob("orders.json.corrupt-*")), "the damaged file was not kept for inspection"


def test_a_corrupt_orders_file_without_a_backup_starts_empty():
    """With nothing to recover from, start clean and keep the evidence - do not crash."""
    tmp = _tmp()
    path = tmp / "orders.json"
    path.write_text("]]] not json [[[", encoding="utf-8")

    store = OrderStore(path)  # must not raise
    assert store.list_orders() == []
    assert list(tmp.glob("orders.json.corrupt-*")), "the damaged file was not quarantined"

    # ...and the store is usable again immediately.
    order = store.create_order(
        Customer("Recovered Caller", "9876543210"),
        [OrderItem.from_medicine(_any_medicine(), 1)],
    )
    assert store.get(order.id) is not None


def test_a_corrupt_escalations_file_does_not_take_the_queue_down():
    """The pharmacist queue needs the same resilience as the order book."""
    tmp = _tmp()
    path = tmp / "escalations.json"
    path.write_text("{ broken", encoding="utf-8")

    queue = EscalationQueue(path)  # must not raise
    item = queue.raise_escalation(EscalationReason.CLINICAL_QUESTION, "can I mix these?")
    assert queue.get(item.id) is not None
    assert queue.open_count == 1


def test_one_bad_record_does_not_discard_the_whole_file():
    """A single malformed row must cost one order, not the entire order book."""
    tmp = _tmp()
    path = tmp / "orders.json"

    store = OrderStore(path)
    good = store.create_order(
        Customer("Good Record", "9876543210"),
        [OrderItem.from_medicine(_any_medicine(), 1)],
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["orders"].append({"id": "ORD-9999", "missing": "everything else"})
    path.write_text(json.dumps(payload), encoding="utf-8")

    reloaded = OrderStore(path)
    assert reloaded.get(good.id) is not None, "a good record was lost with the bad one"
    assert reloaded.get("ORD-9999") is None


def test_an_interrupted_write_leaves_a_readable_file():
    """P-14 regression: the atomic write must survive the temp file being left behind."""
    tmp = _tmp()
    path = tmp / "orders.json"
    store = OrderStore(path)
    store.create_order(
        Customer("Atomic Caller", "9876543210"),
        [OrderItem.from_medicine(_any_medicine(), 1)],
    )

    # A crashed write leaves a .tmp; the real file must still be complete.
    (tmp / "orders.json.tmp").write_text("half a fi", encoding="utf-8")
    assert len(OrderStore(path).list_orders()) == 1


# --------------------------------------------------------------------------
# F-16 - false medicine matches on ordinary words
# --------------------------------------------------------------------------


def test_ordinary_english_words_do_not_match_a_medicine():
    """F-16: "daily" phonetically matched Centrum, through the alias "daily vitamin"."""
    matched = {
        word: handle_query(word).medicines[0].medicine.brand_name
        for word in ("daily", "cold", "pain", "cough", "formula", "tablets")
        if handle_query(word).medicines
    }
    assert not matched, f"ordinary words matched medicines: {matched}"


def test_real_medicine_names_still_match():
    """The guard above must not cost any of the 35 mis-transcription cases."""
    for spoken, expected in (
        ("amoxycillin", "Amoxil"),
        ("paracetemol", "Panadol"),
        ("cough syrup", "Benadryl Cough Formula"),
        ("vitamin d", "D-Rise"),
        ("daily vitamin", "Centrum"),
        ("cold tablet", "Dolo Cold"),
        ("pain gel", "Voltaren Gel"),
        ("inhaler", "Ventolin"),
    ):
        found = kb.find(spoken)
        assert found is not None and found.brand_name == expected, (
            f"{spoken!r} -> {found.brand_name if found else 'no match'} (expected {expected})"
        )


# --------------------------------------------------------------------------
# Standalone runner
# --------------------------------------------------------------------------


def main() -> None:
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    failures: list[str] = []
    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failures.append(f"{test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 - report, don't abort the run
            failures.append(f"{test.__name__}: {type(exc).__name__}: {exc}")

    print(f"{len(tests) - len(failures)}/{len(tests)} negative tests passed.")
    for failure in failures:
        print(f"  FAIL {failure}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
