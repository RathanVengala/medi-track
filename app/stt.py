from __future__ import annotations

import io
import logging
import re
from functools import lru_cache

from app.config import settings

logger = logging.getLogger("meditrack.stt")


class TranscriptionError(Exception):
    pass


# Whisper fills silence with memorised training data - YouTube outros, subtitle
# credits, sound tags. These come back as confident transcripts, so they have to
# be filtered rather than trusted. Compared after stripping punctuation.
_WHISPER_ARTIFACTS = {
    "you",
    "thanks for watching",
    "thank you for watching",
    "thanks for watching and dont forget to subscribe",
    "please subscribe",
    "subscribe",
    "subtitles by the amaraorg community",
    "amaraorg",
    "music",
    "applause",
    "silence",
    "background noise",
    "beep",
}


def is_probable_noise(text: str) -> bool:
    """
    True when a transcript is almost certainly hallucinated from silence.

    Two signals: a transcript containing no ASCII letters or digits at all (we
    pin Whisper to English, so another script means it invented the audio), and
    the known filler phrases above.

    Digits count. A dictated phone number - "9 8 7 6 5 4 3 2 1 0" - is a real
    utterance with no letters in it, and counting letters alone silently threw
    away every phone number a caller gave us.
    """
    stripped = (text or "").strip()
    if not stripped:
        return True
    ascii_alnum = sum(1 for ch in stripped if ch.isascii() and ch.isalnum())
    if ascii_alnum < 2:
        return True
    collapsed = re.sub(r"[^a-z ]+", "", stripped.lower()).strip()
    collapsed = re.sub(r"\s+", " ", collapsed)
    return collapsed in _WHISPER_ARTIFACTS


@lru_cache(maxsize=1)
def _whisper_prompt() -> str:
    """
    Bias Whisper toward pharmacy vocabulary.

    Whisper's optional prompt steers spelling of unusual words, which is
    exactly what drug names are. Capped well under the 224-token limit.
    """
    from app.knowledge_base import get_knowledge_base

    names: list[str] = []
    for med in get_knowledge_base().medicines:
        names.append(med.brand_name)
        # Only the first word of a generic, to keep combination products short.
        names.append(med.generic_name.split(",")[0].split(" and ")[0].strip())

    seen: set[str] = set()
    unique = [n for n in names if n and not (n.lower() in seen or seen.add(n.lower()))]

    prompt = "A customer phone call to a pharmacy about medicine availability, price and refills. Medicines: "
    for name in unique:
        candidate = f"{prompt}{name}, "
        if len(candidate) > 700:  # ~180 tokens, safely under the limit
            break
        prompt = candidate
    return prompt.rstrip(", ") + "."


def transcribe_audio(audio_bytes: bytes, filename: str = "audio.wav") -> str:
    """
    Convert raw audio bytes into text. Returns "" when nothing was understood.

    This function never raises. Silence on the line is the single most common
    thing that happens during a call, and it is exactly what used to kill the
    session: the fallback recogniser reached out to Google, and the resulting
    OSError / TimeoutError / socket error propagated all the way out of the
    webhook. The caller sat in silence, and the conversation was over.

    An empty string means "I did not catch that" - `is_probable_noise("")` is
    True, so the voice handler re-prompts and the call continues.
    """
    if settings.OPENAI_API_KEY:
        try:
            return _transcribe_with_whisper(audio_bytes, filename)
        except Exception as exc:  # noqa: BLE001 - we want a graceful fallback
            logger.warning("Whisper transcription failed (%s); falling back.", exc)

    try:
        return _transcribe_with_speech_recognition(audio_bytes)
    except TranscriptionError as exc:
        logger.info("No transcript from the fallback recogniser: %s", exc)
    except Exception as exc:  # noqa: BLE001 - a live call must never drop
        # Network errors, a malformed WAV, a missing optional dependency: all
        # of these mean "no transcript", not "end the call".
        logger.warning("Fallback transcription failed (%s: %s).", type(exc).__name__, exc)
    return ""


def _transcribe_with_whisper(audio_bytes: bytes, filename: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = filename  # the SDK inspects this for the file extension

    kwargs: dict = {
        "model": settings.WHISPER_MODEL,
        "file": audio_file,
        "prompt": _whisper_prompt(),
        # Greedy decoding; the default lets Whisper resample and drift into
        # invented text when the audio is quiet.
        "temperature": 0.0,
    }
    if settings.STT_LANGUAGE:
        kwargs["language"] = settings.STT_LANGUAGE

    result = client.audio.transcriptions.create(**kwargs)
    text = (result.text or "").strip()
    if not text:
        raise TranscriptionError("Whisper returned an empty transcript.")
    logger.info("Whisper transcript: %s", text)
    return text


def _transcribe_with_speech_recognition(audio_bytes: bytes) -> str:
    import speech_recognition as sr

    recognizer = sr.Recognizer()
    # Bound the outbound request. Without this the recogniser inherits the
    # global socket timeout - which is None - so a silent or unreachable
    # endpoint blocks the worker thread indefinitely.
    recognizer.operation_timeout = settings.STT_TIMEOUT_SECONDS

    try:
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio_data = recognizer.record(source)
    except (ValueError, OSError, EOFError) as exc:
        # Twilio occasionally hands back a truncated or zero-length recording.
        raise TranscriptionError(f"Unreadable audio: {exc}") from exc

    try:
        text = recognizer.recognize_google(audio_data)
    except sr.UnknownValueError as exc:
        raise TranscriptionError("Could not understand audio.") from exc
    except sr.RequestError as exc:
        raise TranscriptionError(f"Speech recognition service error: {exc}") from exc
    except (OSError, TimeoutError) as exc:
        # Includes socket.timeout and urllib's URLError, which SpeechRecognition
        # does not always wrap in RequestError.
        raise TranscriptionError(f"Speech recognition unreachable: {exc}") from exc

    logger.info("SpeechRecognition transcript: %s", text)
    return (text or "").strip()
