"""
Local demo/test harness for the MediTrack AI pipeline — no Twilio needed.

Week 2 adds knowledge-base modes so you can show the full flow:

    Customer Query -> Knowledge Base Search -> AI Response
"""

from __future__ import annotations

import argparse
import sys

from app.conversation import get_session
from app.knowledge_base import get_knowledge_base
from app.llm import generate_response
from app.query_handler import build_context, handle_query
from app.stt import TranscriptionError, is_probable_noise, transcribe_audio

GOODBYE_PHRASES = ("bye", "goodbye", "that's all", "thats all", "hang up")

DEMO_QUERIES = [
    "Hi, do you have amoxicillin in stock?",
    "How much is a pack of ventolin inhaler?",
    "I need two boxes of augmentin six twenty five",
    "Do you have anything for a headache?",
    "How many paracetamol tablets should I take for a fever?",
    "What time do you close on Sunday?",
]


def _rule(title: str = "") -> None:
    print("\n" + "=" * 72)
    if title:
        print(title)
        print("=" * 72)


def show_pipeline(session, text: str, verbose: bool = True) -> None:
    """Print all three demo stages for one customer utterance."""
    print(f"\nCustomer Query   : {text}")

    result = handle_query(text)
    meds = ", ".join(
        f"{m.medicine.brand_name} (heard '{m.spoken_text}', score {m.score})"
        for m in result.medicines
    )
    print(f"  [1] Intent     : {result.intent.value}")
    print(f"  [2] KB Search  : {meds or 'no medicine matched'}")
    for avail in result.availability:
        print(f"      Stock      : {avail.message}")
    if result.suggestions:
        print(
            "      Suggested  : "
            + ", ".join(h.medicine.brand_name for h in result.suggestions)
        )
    if result.quantity:
        print(f"      Quantity   : {result.quantity}")
    if result.needs_pharmacist:
        print("      Routing    : clinical question -> pharmacist")

    if verbose:
        print("  --- grounding context sent to the LLM ---")
        for line in build_context(result).splitlines():
            print(f"      {line}")

    reply = generate_response(session, text)
    print(f"  [3] AI Response: {reply}")


def run_demo(verbose: bool) -> None:
    kb = get_knowledge_base()
    _rule(f"MediTrack AI - Week 2 demo ({len(kb.medicines)} medicines in the knowledge base)")
    session = get_session("local-demo")
    for query in DEMO_QUERIES:
        show_pipeline(session, query, verbose=verbose)
    print()


def run_kb_search(query: str) -> None:
    kb = get_knowledge_base()
    _rule(f"Knowledge base search: {query!r}")
    hits = kb.search(query, limit=5)
    if not hits:
        print("No matches.")
        return
    for hit in hits:
        med = hit.medicine
        print(
            f"  {hit.score:>5.3f}  {med.display_name:<45} "
            f"{med.stock_status:<13} {kb.price_text(med)}  [matched '{hit.matched_on}' via {hit.match_type}]"
        )
    print()
    print("Availability answer:", kb.check_availability(query).message)


def run_text_turn(session, text: str) -> None:
    print(f"You said        : {text}")
    reply = generate_response(session, text)
    print(f"MediTrack AI    : {reply}\n")


def run_file_mode(path: str) -> None:
    session = get_session("local-file-test")
    with open(path, "rb") as f:
        audio_bytes = f.read()
    text = transcribe_audio(audio_bytes, filename=path)
    show_pipeline(session, text, verbose=False)


def run_mic_mode() -> None:
    import speech_recognition as sr

    session = get_session("local-mic-test")
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    print("Starting mic conversation. Say 'goodbye' to end.\n")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)

    while True:
        with mic as source:
            print("Listening... (speak now)")
            try:
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)
            except sr.WaitTimeoutError:
                print("  (no speech detected, still listening)")
                continue

        wav_bytes = audio.get_wav_data()
        try:
            text = transcribe_audio(wav_bytes, filename="mic.wav")
        except TranscriptionError as exc:
            print(f"  (could not transcribe: {exc})")
            continue

        # Whisper invents text from silence and room noise; answering those
        # hallucinations makes the agent look like it is talking to itself.
        if is_probable_noise(text):
            print(f"  (ignored background noise, heard: {text!r})")
            continue

        if any(p in text.lower() for p in GOODBYE_PHRASES):
            print(f"You said        : {text}")
            print("MediTrack AI    : Thanks for calling! Have a great day.")
            break

        show_pipeline(session, text, verbose=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local test of the MediTrack AI voice pipeline.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mic", action="store_true", help="Use your microphone for a live conversation.")
    group.add_argument("--file", type=str, help="Path to a WAV/MP3 file to transcribe and respond to.")
    group.add_argument("--text", type=str, help="Skip STT; send raw text straight to the LLM turn.")
    group.add_argument("--kb", type=str, help="Search the pharmacy knowledge base only (no LLM).")
    group.add_argument(
        "--demo",
        action="store_true",
        help="Run the scripted Week 2 demo: query -> knowledge base search -> AI response.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="In --demo mode, hide the raw grounding context block.",
    )
    args = parser.parse_args()

    try:
        if args.mic:
            run_mic_mode()
        elif args.file:
            run_file_mode(args.file)
        elif args.kb:
            run_kb_search(args.kb)
        elif args.demo:
            run_demo(verbose=not args.quiet)
        elif args.text:
            session = get_session("local-text-test")
            show_pipeline(session, args.text, verbose=not args.quiet)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
