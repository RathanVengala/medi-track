from __future__ import annotations

import logging

import httpx
from fastapi import FastAPI, Form, Request, Response
from twilio.twiml.voice_response import VoiceResponse

from app.config import settings
from app.conversation import end_session, get_session
from app.knowledge_base import get_knowledge_base
from app.llm import generate_response
from app.query_handler import build_context, handle_query
from app.stt import is_probable_noise, transcribe_audio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("meditrack.main")

app = FastAPI(title="MediTrack AI - Voice Agent (Week 2: Pharmacy Knowledge Base)")

GOODBYE_PHRASES = ("bye", "goodbye", "that's all", "thats all", "hang up", "no thank you")


@app.get("/health")
def health() -> dict:
    """Simple liveness check."""
    kb = get_knowledge_base()
    return {
        "status": "ok",
        "service": "meditrack-ai-week2",
        "medicines_loaded": len(kb.medicines),
    }


# --- Knowledge base endpoints (useful for the Week 2 demo / debugging) -------


@app.get("/kb/medicines")
def kb_medicines(category: str | None = None) -> dict:
    """List the formulary, optionally filtered by category."""
    kb = get_knowledge_base()
    meds = kb.medicines
    if category:
        meds = [m for m in meds if category.lower() in m.category.lower()]
    return {"count": len(meds), "medicines": [m.to_dict() for m in meds]}


@app.get("/kb/search")
def kb_search(q: str, limit: int = 5) -> dict:
    """Fuzzy medicine search — the same call the voice pipeline makes."""
    hits = get_knowledge_base().search(q, limit=limit)
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


@app.get("/kb/availability")
def kb_availability(name: str) -> dict:
    """Availability answer for a spoken medicine name."""
    result = get_knowledge_base().check_availability(name)
    return {
        "query": result.query,
        "status": result.status,
        "message": result.message,
        "medicine": result.medicine.to_dict() if result.medicine else None,
        "alternatives": [a.to_dict() for a in result.alternatives],
    }


@app.post("/query/analyze")
def analyze_query(text: str = Form(...)) -> dict:
    """
    Run a customer utterance through intent detection + knowledge base search
    without calling the LLM. Shows exactly what the AI reply is grounded on.
    """
    result = handle_query(text)
    return {
        "text": result.text,
        "intent": result.intent.value,
        "quantity": result.quantity,
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


@app.post("/voice/incoming")
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


@app.post("/voice/process")
async def voice_process(
    CallSid: str = Form(...),
    RecordingUrl: str | None = Form(None),
) -> Response:
    """
    Webhook Twilio hits once a <Record> finishes.
    Downloads the recording, runs STT, generates an AI reply, speaks it,
    and either continues the conversation or ends the call.
    """
    vr = VoiceResponse()
    session = get_session(CallSid)

    if not RecordingUrl:
        vr.say("Sorry, I didn't catch that. Could you say that again?", voice="Polly.Joanna")
        _record_next_turn(vr)
        return Response(content=str(vr), media_type="application/xml")

    audio_bytes = await _download_twilio_recording(RecordingUrl)
    user_text = transcribe_audio(audio_bytes, filename=f"{CallSid}.wav")
    logger.info("Caller (%s) said: %s", CallSid, user_text)

    # Silence and line noise make Whisper invent text; re-prompt instead of
    # answering something the caller never said.
    if is_probable_noise(user_text):
        logger.info("Discarded probable noise transcript for %s: %r", CallSid, user_text)
        vr.say("Sorry, I didn't catch that. Could you say that again?", voice="Polly.Joanna")
        _record_next_turn(vr)
        return Response(content=str(vr), media_type="application/xml")

    if any(phrase in user_text.lower() for phrase in GOODBYE_PHRASES):
        vr.say(f"Thanks for calling {settings.PHARMACY_NAME}. Have a great day!", voice="Polly.Joanna")
        vr.hangup()
        end_session(CallSid)
        return Response(content=str(vr), media_type="application/xml")

    ai_reply = generate_response(session, user_text)
    vr.say(ai_reply, voice="Polly.Joanna")
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
            timeout=15.0,
        )
        resp.raise_for_status()
        return resp.content
