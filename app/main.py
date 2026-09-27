from __future__ import annotations

import logging

import httpx
from fastapi import Depends, FastAPI, Form, HTTPException, Request, Response, status
from twilio.twiml.voice_response import VoiceResponse

from app.config import settings
from app.conversation import active_session_count, end_session, get_session
from app.escalation import EscalationStatus, get_escalation_queue
from app.knowledge_base import PharmacyKnowledgeBase, get_knowledge_base
from app.llm import generate_response
from app.orders import (
    InvalidQuantity,
    OrderError,
    OrderStatus,
    get_order_store,
    validate_quantity,
)
from app.query_handler import build_context, handle_query
from app.security import StaffAuth, TwilioAuth, startup_warnings
from app.stt import is_probable_noise, transcribe_audio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("meditrack.main")

app = FastAPI(
    title="MediTrack AI - Voice Agent (Week 4: Hardening, Security & Recovery)",
    description=(
        "Staff endpoints require an X-API-Key header. Voice webhooks require a "
        "valid X-Twilio-Signature. /health is open."
    ),
)

GOODBYE_PHRASES = ("bye", "goodbye", "that's all", "thats all", "hang up", "no thank you")

RETRY_PROMPT = "Sorry, I didn't catch that. Could you say that again?"


@app.on_event("startup")
def announce_configuration() -> None:
    """
    Surface an unsafe or incomplete configuration at boot, not at 2am.

    The knowledge base is loaded here too, so a missing formulary is reported
    once as a clear log line instead of failing the first caller's request.
    """
    for warning in startup_warnings():
        logger.warning("CONFIG: %s", warning)

    kb = get_knowledge_base()
    if kb.is_healthy:
        logger.info("Formulary loaded: %d medicines.", len(kb.medicines))
    else:
        logger.error("Starting DEGRADED: %s", kb.load_error)


def require_knowledge_base() -> PharmacyKnowledgeBase:
    """
    Dependency for the routes that cannot work without the formulary.

    The app still starts when data/medicines.json is missing - it just says so,
    clearly, on the routes that need it, instead of crashing at import.
    """
    kb = get_knowledge_base()
    if not kb.is_healthy:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=kb.load_error or "The pharmacy formulary is empty.",
        )
    return kb


def _check_length(text: str, field: str = "text") -> str:
    """Refuse an oversized request body before any work is done on it."""
    if len(text or "") > settings.MAX_QUERY_CHARS:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"'{field}' is {len(text)} characters; the limit is "
                f"{settings.MAX_QUERY_CHARS}. A spoken turn is a sentence or two."
            ),
        )
    return text


@app.get("/health")
def health() -> dict:
    """
    Liveness check. Open by design - a monitor should not need a secret.

    Reports what is wrong rather than only whether the process is up, so a
    degraded start (missing formulary, unset API key) is visible.
    """
    kb = get_knowledge_base()
    warnings = startup_warnings()
    if not kb.is_healthy:
        warnings.append(kb.load_error or "The pharmacy formulary is empty.")
    return {
        "status": "ok" if kb.is_healthy else "degraded",
        "service": "meditrack-ai-week4",
        "medicines_loaded": len(kb.medicines),
        "orders": len(get_order_store().list_orders()),
        "open_escalations": get_escalation_queue().open_count,
        "active_call_sessions": active_session_count(),
        "warnings": warnings,
    }


# --- Knowledge base endpoints (useful for the Week 2 demo / debugging) -------


@app.get("/kb/medicines", dependencies=[StaffAuth])
def kb_medicines(
    category: str | None = None, kb: PharmacyKnowledgeBase = Depends(require_knowledge_base)
) -> dict:
    """List the formulary, optionally filtered by category."""
    meds = kb.medicines
    if category:
        meds = [m for m in meds if category.lower() in m.category.lower()]
    return {"count": len(meds), "medicines": [m.to_dict() for m in meds]}


@app.get("/kb/search", dependencies=[StaffAuth])
def kb_search(
    q: str, limit: int = 5, kb: PharmacyKnowledgeBase = Depends(require_knowledge_base)
) -> dict:
    """Fuzzy medicine search — the same call the voice pipeline makes."""
    _check_length(q, "q")
    limit = max(1, min(limit, 50))  # an unbounded limit is a free full dump
    hits = kb.search(q, limit=limit)
    return {
        "query": q,
        "results": [
            {
                "score": h.score,
                "matched_on": h.matched_on,
                "match_type": h.match_type,
                **h.medicine.to_dict(),
            }
            for h in hits
        ],
    }


@app.get("/kb/availability", dependencies=[StaffAuth])
def kb_availability(
    name: str, kb: PharmacyKnowledgeBase = Depends(require_knowledge_base)
) -> dict:
    """Availability answer for a spoken medicine name."""
    _check_length(name, "name")
    result = kb.check_availability(name)
    return {
        "query": result.query,
        "status": result.status,
        "message": result.message,
        "medicine": result.medicine.to_dict() if result.medicine else None,
        "alternatives": [a.to_dict() for a in result.alternatives],
    }


@app.get("/inventory/check", dependencies=[StaffAuth])
def inventory_check(
    name: str, quantity: int = 1, kb: PharmacyKnowledgeBase = Depends(require_knowledge_base)
) -> dict:
    """Can we supply N packs of this right now? (on hand minus reserved)"""
    _check_length(name, "name")
    try:
        quantity = validate_quantity(quantity)
    except InvalidQuantity as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    medicine = kb.find(name)
    if medicine is None:
        return {"query": name, "found": False, "message": f"'{name}' is not in the formulary."}
    return {"query": name, "found": True, **get_order_store().check_stock(medicine, quantity).to_dict()}


# --- Orders (Week 3) --------------------------------------------------------


@app.get("/orders", dependencies=[StaffAuth])
def list_orders(status: str | None = None, phone: str | None = None) -> dict:
    orders = get_order_store().list_orders(status=status, phone=phone)
    return {"count": len(orders), "orders": [o.to_dict() for o in orders]}


@app.get("/orders/{order_id}", dependencies=[StaffAuth])
def get_order(order_id: str) -> dict:
    order = get_order_store().get(order_id)
    if order is None:
        raise HTTPException(status_code=404, detail=f"Order not found: {order_id}")
    return order.to_dict()


@app.post("/orders/{order_id}/status", dependencies=[StaffAuth])
def set_order_status(order_id: str, status: str = Form(...), note: str = Form("")) -> dict:
    """Pharmacist-side transition, e.g. awaiting_prescription -> ready_for_pickup."""
    try:
        new_status = OrderStatus(status)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status. Valid values: {[s.value for s in OrderStatus]}",
        )
    try:
        return get_order_store().update_status(order_id, new_status, note).to_dict()
    except OrderError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


# --- Pharmacist escalation queue (Week 3) -----------------------------------


@app.get("/escalations", dependencies=[StaffAuth])
def list_escalations(status: str | None = None, reason: str | None = None) -> dict:
    queue = get_escalation_queue()
    items = queue.list_escalations(status=status, reason=reason)
    return {"count": len(items), "open": queue.open_count, "escalations": [e.to_dict() for e in items]}


@app.post("/escalations/{escalation_id}/status", dependencies=[StaffAuth])
def set_escalation_status(
    escalation_id: str, status: str = Form(...), resolution: str = Form("")
) -> dict:
    try:
        new_status = EscalationStatus(status)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status. Valid values: {[s.value for s in EscalationStatus]}",
        )
    try:
        return get_escalation_queue().update_status(escalation_id, new_status, resolution).to_dict()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/query/analyze", dependencies=[StaffAuth])
def analyze_query(
    text: str = Form(...), kb: PharmacyKnowledgeBase = Depends(require_knowledge_base)
) -> dict:
    """
    Run a customer utterance through intent detection + knowledge base search
    without calling the LLM. Shows exactly what the AI reply is grounded on.

    Oversized input is rejected here rather than truncated: an HTTP client can
    be told it sent too much, where a phone caller cannot.
    """
    _check_length(text)
    result = handle_query(text, kb=kb)
    return {
        "text": result.text,
        "intent": result.intent.value,
        "quantity": result.quantity,
        "quantity_issue": result.quantity_issue.value if result.quantity_issue else None,
        "strength": result.strength,
        "form": result.form,
        "needs_pharmacist": result.needs_pharmacist,
        "medicines": [
            {
                "heard_as": m.spoken_text,
                "score": m.score,
                "match_type": m.match_type,
                **m.medicine.to_dict(),
            }
            for m in result.medicines
        ],
        "context": build_context(result),
    }


@app.post("/voice/incoming", dependencies=[TwilioAuth])
async def voice_incoming(CallSid: str = Form(...)) -> Response:
    """
    First webhook Twilio hits when a call comes in.
    Greets the caller, then records their response.
    """
    get_session(CallSid)  # initializes a fresh conversation session

    vr = VoiceResponse()
    vr.say(
        f"Thanks for calling {settings.PHARMACY_NAME}. "
        "I'm your AI assistant. How can I help you today?",
        voice="Polly.Joanna",
    )
    _record_next_turn(vr)
    return Response(content=str(vr), media_type="application/xml")


@app.post("/voice/process", dependencies=[TwilioAuth])
async def voice_process(
    CallSid: str = Form(...),
    RecordingUrl: str | None = Form(None),
) -> Response:
    """
    Webhook Twilio hits once a <Record> finishes.
    Downloads the recording, runs STT, generates an AI reply, speaks it,
    and either continues the conversation or ends the call.

    Every step that can fail - the download, the transcription, the model call
    - is contained here. A turn that goes wrong costs the caller one "say that
    again", never the call: an exception escaping this handler is a 500 to
    Twilio, which plays an error tone and hangs up on a real customer.
    """
    vr = VoiceResponse()
    session = get_session(CallSid)

    if not RecordingUrl:
        return _reprompt(vr)

    try:
        audio_bytes = await _download_twilio_recording(RecordingUrl)
    except (httpx.HTTPError, OSError) as exc:
        logger.warning("Could not fetch the recording for %s: %s", CallSid, exc)
        return _reprompt(vr)

    # transcribe_audio never raises; "" means nothing was understood, which
    # is_probable_noise treats as noise and the caller is asked to repeat.
    user_text = transcribe_audio(audio_bytes, filename=f"{CallSid}.wav")
    logger.info("Caller (%s) said: %s", CallSid, user_text)

    # Silence and line noise make Whisper invent text; re-prompt instead of
    # answering something the caller never said.
    if is_probable_noise(user_text):
        logger.info("Discarded probable noise transcript for %s: %r", CallSid, user_text)
        return _reprompt(vr)

    if any(phrase in user_text.lower() for phrase in GOODBYE_PHRASES):
        vr.say(f"Thanks for calling {settings.PHARMACY_NAME}. Have a great day!", voice="Polly.Joanna")
        vr.hangup()
        end_session(CallSid)
        return Response(content=str(vr), media_type="application/xml")

    try:
        ai_reply = generate_response(session, user_text)
    except Exception as exc:  # noqa: BLE001 - a live call must never drop
        logger.exception("Reply generation failed for %s: %s", CallSid, exc)
        return _reprompt(
            vr, "Sorry, I had trouble with that. Could you say it another way?"
        )

    vr.say(ai_reply, voice="Polly.Joanna")
    _record_next_turn(vr)
    return Response(content=str(vr), media_type="application/xml")


def _reprompt(vr: VoiceResponse, message: str = RETRY_PROMPT) -> Response:
    """Ask the caller to repeat and keep listening - the call stays open."""
    vr.say(message, voice="Polly.Joanna")
    _record_next_turn(vr)
    return Response(content=str(vr), media_type="application/xml")


def _record_next_turn(vr: VoiceResponse) -> None:
    """Attach a <Record> verb that captures the customer's next utterance."""
    vr.record(
        action=f"{settings.PUBLIC_BASE_URL}/voice/process",
        method="POST",
        max_length=settings.RECORD_MAX_LENGTH_SECONDS,
        timeout=settings.RECORD_TIMEOUT_SECONDS,
        play_beep=False,
        trim="trim-silence",
    )


async def _download_twilio_recording(recording_url: str) -> bytes:
    """
    Twilio recording URLs need HTTP Basic auth (account SID / auth token)
    and a format suffix to get raw audio rather than an HTML player page.
    """
    url = f"{recording_url}.wav"
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            url,
            auth=(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN),
            timeout=settings.TWILIO_DOWNLOAD_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        return resp.content
