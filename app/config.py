import os
from dotenv import load_dotenv

# Loads variables from a .env file in the project root, if present.
load_dotenv()


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


settings = Settings()
