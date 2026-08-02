"""
LLM response generation, grounded on the pharmacy knowledge base (Week 2).

Every turn now runs:

    customer text -> query handler -> knowledge base search
                  -> grounding context -> LLM -> spoken reply

If no LLM provider is configured (or the call fails), the reply is composed
directly from the knowledge base instead, so the pipeline still answers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.config import settings
from app.knowledge_base import get_knowledge_base
from app.query_handler import QueryResult, build_context, fallback_answer, handle_query

logger = logging.getLogger("meditrack.llm")

BASE_PROMPT = """You are the MediTrack AI voice assistant answering the phone for {pharmacy_name}.

You are speaking with a customer over a phone call, so:
- Keep every reply short (1-3 sentences) and conversational, like a real phone call - not a written document.
- Never use bullet points, markdown, or numbered lists; you are being converted to speech.
- Say prices and quantities the way a person would. Prices are in Indian rupees,
  so "Rs.361.25" is spoken as "three hundred sixty one rupees twenty five paise".
  Never read the price out as digits or as dollars.
- Be warm, clear, and professional, the way a helpful pharmacy staff member would sound.

Each customer turn is accompanied by a PHARMACY KNOWLEDGE BASE RESULT block containing
the results of a live search of our formulary and stock. Ground your answer in it:
- State availability, price, pack size and prescription status ONLY from that block.
- If the block says a medicine is out of stock, say so and offer the listed alternatives.
- If the block reports no match, do not guess a medicine. Ask the caller to repeat or
  spell the name, or ask what the medicine is for.
- Never invent a medicine, a price, or a stock number that is not in the block.

Safety rules:
- Do not give dosage advice, drug interaction advice, or any diagnosis. For those,
  offer to connect the caller with the pharmacist on duty.
- Prescription-only medicines always need a valid prescription; say so when relevant.
- Reply with the spoken sentences only. Do not include internal or system XML tags.

You cannot yet place orders in the system. If the customer wants to order or reserve
something, confirm the details back to them and tell them a pharmacist will follow up.

STORE PROFILE (background facts you may use):
{store_profile}
"""


def build_system_prompt() -> str:
    kb = get_knowledge_base()
    return BASE_PROMPT.format(
        pharmacy_name=kb.pharmacy.get("name", settings.PHARMACY_NAME),
        store_profile=kb.store_summary(),
    )


# Kept as a module-level constant for backwards compatibility with Week 1 code
# and tests that imported SYSTEM_PROMPT directly.
SYSTEM_PROMPT = build_system_prompt()


@dataclass
class ConversationSession:
    """Holds the message history for a single caller (in-memory, per call)."""

    history: list[dict] = field(default_factory=list)
    last_query: QueryResult | None = None

    def add_user_turn(self, text: str) -> None:
        self.history.append({"role": "user", "content": text})

    def add_assistant_turn(self, text: str) -> None:
        self.history.append({"role": "assistant", "content": text})


def generate_response(session: ConversationSession, user_text: str) -> str:
    """
    Given the running conversation and the customer's latest transcribed
    utterance, search the knowledge base and produce the next spoken reply.
    """
    query = handle_query(user_text)
    session.last_query = query
    context = build_context(query)
    logger.debug("Knowledge base context:\n%s", context)

    session.add_user_turn(user_text)
    messages = _messages_with_context(session, context)

    try:
        if settings.LLM_PROVIDER == "anthropic" and settings.ANTHROPIC_API_KEY:
            reply = _generate_with_anthropic(messages)
        elif settings.LLM_PROVIDER == "openai" and settings.OPENAI_API_KEY:
            reply = _generate_with_openai(messages)
        else:
            logger.warning(
                "No API key configured for provider '%s'; answering from the knowledge base only.",
                settings.LLM_PROVIDER,
            )
            reply = fallback_answer(query)
    except Exception as exc:  # noqa: BLE001 - a live call must never drop
        logger.error("LLM call failed (%s); falling back to knowledge base reply.", exc)
        reply = fallback_answer(query)

    session.add_assistant_turn(reply)
    logger.info("LLM reply: %s", reply)
    return reply


def _messages_with_context(session: ConversationSession, context: str) -> list[dict]:
    """
    Copy the history and append the knowledge base block to the latest user
    turn. The stored history keeps the caller's plain words, so the context of
    one turn never leaks into the next.
    """
    messages = [dict(turn) for turn in session.history]
    if messages and messages[-1]["role"] == "user":
        messages[-1]["content"] = f"{messages[-1]['content']}\n\n{context}"
    return messages


def _generate_with_openai(messages: list[dict]) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    completion = client.chat.completions.create(
        model=settings.OPENAI_LLM_MODEL,
        messages=[{"role": "system", "content": build_system_prompt()}, *messages],
        max_tokens=200,
        temperature=0.3,
    )
    return (completion.choices[0].message.content or "").strip()


def _generate_with_anthropic(messages: list[dict]) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

    # max_tokens caps thinking + reply text together, and Claude Opus 5 thinks by
    # default, so leave headroom. The reply itself stays short because the system
    # prompt constrains it to 1-3 spoken sentences.
    response = client.messages.create(
        model=settings.ANTHROPIC_LLM_MODEL,
        system=build_system_prompt(),
        messages=messages,
        max_tokens=1000,
    )
    return "".join(block.text for block in response.content if block.type == "text").strip()
