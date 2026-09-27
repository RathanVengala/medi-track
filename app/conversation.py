"""
Per-call conversation sessions.

A session holds the running history and the order draft for one phone call, so
the caller can be asked for their name on one turn and their number on the
next. The whole point is that it survives between webhooks - which is also why
it needs an expiry.

Twilio does not reliably tell us a call ended. A caller who hangs up
mid-sentence, a dropped connection, a webhook that never arrives: in every one
of those cases `end_session` is never called, and the entry used to sit in the
dict for the lifetime of the process. Under real traffic that is an unbounded
memory leak, so sessions are now evicted on two rules - idle for longer than
SESSION_TTL_SECONDS, or oldest-first once MAX_ACTIVE_SESSIONS is reached.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from app.config import settings
from app.llm import ConversationSession

logger = logging.getLogger("meditrack.conversation")


@dataclass
class _Entry:
    session: ConversationSession
    last_seen: float


# Webhooks for different calls arrive on different worker threads, so every
# read-modify-write of the registry is guarded.
_lock = threading.Lock()
_sessions: dict[str, _Entry] = {}


def _evict_expired(now: float) -> int:
    """Drop sessions idle for longer than the TTL. Caller holds the lock."""
    ttl = settings.SESSION_TTL_SECONDS
    stale = [call_id for call_id, entry in _sessions.items() if now - entry.last_seen > ttl]
    for call_id in stale:
        del _sessions[call_id]
    if stale:
        logger.info("Evicted %d idle call session(s) after %ds.", len(stale), ttl)
    return len(stale)


def _evict_oldest(now: float) -> None:
    """
    Enforce the hard cap, oldest first. Caller holds the lock.

    The TTL alone is not enough: a burst of calls inside one TTL window can
    still exhaust memory, so there is a ceiling on the count as well.
    """
    limit = settings.MAX_ACTIVE_SESSIONS
    while len(_sessions) >= limit:
        oldest = min(_sessions, key=lambda call_id: _sessions[call_id].last_seen)
        del _sessions[oldest]
        logger.warning(
            "Session limit of %d reached; evicted the least recently used call %s.",
            limit,
            oldest,
        )


def get_session(call_id: str) -> ConversationSession:
    """
    The session for this call, creating it on the first webhook of the call.

    Every access sweeps expired sessions, so eviction needs no background
    thread - the traffic that creates sessions is the traffic that cleans them
    up.
    """
    now = time.monotonic()
    with _lock:
        _evict_expired(now)
        entry = _sessions.get(call_id)
        if entry is None:
            _evict_oldest(now)
            # The call id is carried into every order and escalation raised
            # during this call, so a pharmacist can trace a record back to the
            # recording.
            entry = _Entry(session=ConversationSession(call_id=call_id), last_seen=now)
            _sessions[call_id] = entry
            logger.info("Started session for call %s (%d active).", call_id, len(_sessions))
        else:
            entry.last_seen = now
        return entry.session


def end_session(call_id: str) -> None:
    """Drop a session as soon as we know the call is over."""
    with _lock:
        if _sessions.pop(call_id, None) is not None:
            logger.info("Ended session for call %s (%d active).", call_id, len(_sessions))


def active_session_count() -> int:
    """Live sessions, after sweeping expired ones. Used by /health."""
    with _lock:
        _evict_expired(time.monotonic())
        return len(_sessions)


def reset_sessions() -> None:
    """Drop every session. Used by the tests."""
    with _lock:
        _sessions.clear()
