

from __future__ import annotations

import argparse
import sys

from app.conversation import get_session
from app.escalation import get_escalation_queue
from app.knowledge_base import get_knowledge_base
from app.llm import generate_response
from app.orders import get_order_store
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


ORDER_DEMO_TURNS = [
    "Hi, do you have Panadol in stock?",
    "Great, I would like to order two packs please",
    "My name is Rathan Vengala",
    "My number is 9876543210",
    "I'll collect it from the counter",
    "Yes, please place the order",
]


def run_order_demo() -> None:
    """Request -> Availability Check -> Order Creation -> Confirmation."""
    kb = get_knowledge_base()
    store = get_order_store()
    queue = get_escalation_queue()

    _rule("MediTrack AI - Week 3 demo: placing an order over the phone")
    session = get_session("local-order-demo")

    panadol = kb.find("panadol")
    print(f"Stock before call : {store.check_stock(panadol, 1).message}")
    print(f"Orders on file    : {len(store.list_orders())}")
    print(f"Open escalations  : {queue.open_count}")

    for turn in ORDER_DEMO_TURNS:
        print(f"\nCustomer   : {turn}")
        reply = generate_response(session, turn)
        draft = session.draft
        if draft:
            slot = draft.next_slot().value
            collected = [
                f"{k}={v}"
                for k, v in (
                    ("items", ", ".join(f"{i.quantity}x{i.brand_name}" for i in draft.items)),
                    ("name", draft.name),
                    ("phone", draft.phone),
                    ("fulfilment", draft.fulfilment),
                )
                if v
            ]
            print(f"  [order]  : {' | '.join(collected)}")
            print(f"  [awaiting]: {slot}")
        print(f"MediTrack AI: {reply}")

    _rule("Result")
    print(f"Stock after call  : {store.check_stock(panadol, 1).message}")
    for order in store.list_orders()[:3]:
        print(f"Order record      : {order.summary_line()}")
    print(f"Open escalations  : {queue.open_count}")


def run_escalation_demo() -> None:
    """Show each way a call gets handed to the pharmacist."""
    queue = get_escalation_queue()
    _rule("MediTrack AI - Week 3: pharmacist escalation paths")
    cases = [
        ("Clinical question", "How many paracetamol can I take in a day?"),
        ("Prescription-only order", "I want to order one pack of Amoxil"),
        ("Not enough stock", "Please reserve ten Ventolin inhalers"),
        ("Customer asks for a human", "Can I speak to the pharmacist please?"),
    ]
    for label, turn in cases:
        session = get_session(f"local-esc-{label}")
        print(f"\n[{label}]")
        print(f"Customer   : {turn}")
        print(f"MediTrack AI: {generate_response(session, turn)}")

    _rule("Pharmacist queue")
    for item in queue.list_escalations(status="open"):
        print(" ", item.summary_line())


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
    group.add_argument(
        "--order-demo",
        action="store_true",
        help="Run the Week 3 demo: request -> availability -> order creation -> confirmation.",
    )
    group.add_argument(
        "--escalation-demo",
        action="store_true",
        help="Run the Week 3 pharmacist escalation demo.",
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
        elif args.order_demo:
            run_order_demo()
        elif args.escalation_demo:
            run_escalation_demo()
        elif args.text:
            session = get_session("local-text-test")
            show_pipeline(session, args.text, verbose=not args.quiet)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
