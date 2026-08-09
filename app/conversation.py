from __future__ import annotations

from app.llm import ConversationSession

_sessions: dict[str, ConversationSession] = {}


def get_session(call_id: str) -> ConversationSession:
    if call_id not in _sessions:
        # The call id is carried into every order and escalation raised during
        # this call, so a pharmacist can trace a record back to the recording.
        _sessions[call_id] = ConversationSession(call_id=call_id)
    return _sessions[call_id]


def end_session(call_id: str) -> None:
    _sessions.pop(call_id, None)
