import os
from dotenv import load_dotenv

# Loads variables from a .env file in the project root, if present.
load_dotenv()


def _flag(name: str, default: bool) -> bool:
    """Read a boolean env var. Anything but the listed true words is false."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class Settings:
    # --- LLM / STT provider keys -------------------------------------------------
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

    # Which LLM to use for conversation generation: "openai" or "anthropic"
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai").lower()

    # Model names (override in .env if needed).
    # claude-opus-5 is the current Claude default; set ANTHROPIC_LLM_MODEL to
    # claude-haiku-4-5 if you want lower call latency on the phone pipeline.
    OPENAI_LLM_MODEL: str = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
    ANTHROPIC_LLM_MODEL: str = os.getenv("ANTHROPIC_LLM_MODEL", "claude-opus-5")
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "whisper-1")

    # ISO-639-1 code pinned on every Whisper request. Without it Whisper
    # auto-detects per chunk and will happily transcribe an English caller into
    # Tamil, Hindi or Chinese script. Set to "" to re-enable auto-detection.
    STT_LANGUAGE: str = os.getenv("STT_LANGUAGE", "en")

    # --- Twilio (only needed for the live phone pipeline in app/main.py) --------
    TWILIO_ACCOUNT_SID: str = os.getenv("TWILIO_ACCOUNT_SID", "")
    TWILIO_AUTH_TOKEN: str = os.getenv("TWILIO_AUTH_TOKEN", "")
    TWILIO_PHONE_NUMBER: str = os.getenv("TWILIO_PHONE_NUMBER", "")

    # --- App behavior --------------------------------------------------------
    # Base public URL (e.g. an ngrok URL) Twilio will call back into.
    PUBLIC_BASE_URL: str = os.getenv("PUBLIC_BASE_URL", "http://localhost:8000")

    # How many seconds of silence before Twilio's <Record> stops listening.
    RECORD_TIMEOUT_SECONDS: int = int(os.getenv("RECORD_TIMEOUT_SECONDS", "3"))
    RECORD_MAX_LENGTH_SECONDS: int = int(os.getenv("RECORD_MAX_LENGTH_SECONDS", "20"))

    PHARMACY_NAME: str = os.getenv("PHARMACY_NAME", "GreenLeaf Pharmacy")

    # --- Knowledge base (Week 2) ---------------------------------------------
    # Minimum fuzzy-match score before a spoken word is accepted as a medicine
    # name. Lower it if STT mangles names badly; raise it to reduce wrong guesses.
    KB_MATCH_THRESHOLD: float = float(os.getenv("KB_MATCH_THRESHOLD", "0.62"))

    # --- Input limits (Week 4) -----------------------------------------------
    # A phone turn is a sentence or two. Anything longer is either a broken
    # transcript or an attempt to make the n-gram search do unbounded work, so
    # it is rejected at the boundary rather than processed.
    MAX_QUERY_CHARS: int = int(os.getenv("MAX_QUERY_CHARS", "500"))
    # Hard cap on n-grams scanned per utterance, so cost stays bounded even for
    # an input that is under the character limit.
    MAX_QUERY_TOKENS: int = int(os.getenv("MAX_QUERY_TOKENS", "60"))

    # Largest number of packs a caller can order in one line. Above this the
    # quantity is treated as a mis-transcription and re-asked, rather than
    # relying on the stock check to refuse it.
    MAX_ORDER_QUANTITY: int = int(os.getenv("MAX_ORDER_QUANTITY", "50"))

    # --- Security (Week 4) ---------------------------------------------------
    # Shared secret for the pharmacist-facing endpoints (/orders, /escalations,
    # /kb/*, /query/analyze). Sent as `X-API-Key`. Protected routes refuse to
    # serve until this is set - the API is never open by accident.
    ADMIN_API_KEY: str = os.getenv("ADMIN_API_KEY", "")
    # Escape hatch for a purely local demo. Leave false anywhere reachable.
    DISABLE_AUTH: bool = _flag("DISABLE_AUTH", False)

    # Verify the X-Twilio-Signature header on the voice webhooks, so a stranger
    # who finds the public URL cannot inject a fake call.
    VALIDATE_TWILIO_SIGNATURE: bool = _flag("VALIDATE_TWILIO_SIGNATURE", True)

    # --- Call session lifecycle ----------------------------------------------
    # Twilio does not always send a final callback - a dropped call just stops.
    # Sessions older than this are evicted so memory cannot grow without bound.
    SESSION_TTL_SECONDS: int = int(os.getenv("SESSION_TTL_SECONDS", "1800"))
    MAX_ACTIVE_SESSIONS: int = int(os.getenv("MAX_ACTIVE_SESSIONS", "500"))

    # --- Outbound call timeouts ----------------------------------------------
    # Every network call out of the process is bounded, so one slow provider
    # cannot hold a phone line (or a worker thread) open indefinitely.
    LLM_TIMEOUT_SECONDS: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "20"))
    STT_TIMEOUT_SECONDS: float = float(os.getenv("STT_TIMEOUT_SECONDS", "30"))
    TWILIO_DOWNLOAD_TIMEOUT_SECONDS: float = float(
        os.getenv("TWILIO_DOWNLOAD_TIMEOUT_SECONDS", "15")
    )

    # --- Persistence ---------------------------------------------------------
    # Seconds to wait for the cross-process write lock on the JSON stores.
    STORE_LOCK_TIMEOUT_SECONDS: float = float(os.getenv("STORE_LOCK_TIMEOUT_SECONDS", "10"))


settings = Settings()
