from __future__ import annotations

from app.llm import ConversationSession

_sessions: dict[str, ConversationSession] = {}


def get_session(call_id: str) -> ConversationSession:
    if call_id not in _sessions:
        _sessions[call_id] = ConversationSession()
    return _sessions[call_id]


def end_session(call_id: str) -> None:
    _sessions.pop(call_id, None)
