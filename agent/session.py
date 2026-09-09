#!/usr/bin/env python3
"""Conversation session management with history context."""

from typing import List, Any


class Message:
    """Immutable message representation with role and content."""

    def __init__(self, role: str, content: str):
        self._role = role
        self._content = content

    @property
    def role(self) -> str:
        return self._role

    @property
    def content(self) -> str:
        return self._content

    def __repr__(self) -> str:
        return f'Message(role={self._role!r}, content={self._content!r} ...)' 


class MessageHistory:
    """Manages conversation history with sliding window truncation.

    Strategy: SlidingWindow - keeps oldest message plus newest N-1 messages
    when limit is exceeded, preserving system prompt and tail context.
    """

    def __init__(self, limit: int = 32):
        self._messages: List[Message] = []
        self._limit = limit

    def add_user(self, content: str) -> None:
        self._messages.append(Message('user', content))

    def add_assistant(self, content: str) -> None:
        self._messages.append(Message('assistant', content))

    def apply_search_trigger(self, assistant_reply: str, search_results: str) -> str:
        """Apply intermediate assistant reply + search results to history.

        Pattern: Observer - notifies history of search trigger.
        """
        self._messages.append(Message('assistant', assistant_reply))
        self._messages.append(Message('user', search_results))
        return assistant_reply

    def append_final(self, content: str) -> None:
        """Append final assistant response to history."""
        self._messages.append(Message('assistant', content))

    def trim(self) -> None:
        """Trim history to limit using sliding window.

        Strategy: Truncation preserves system prompt (index 0)
        and keeps tail to maintain context.
        """
        if len(self._messages) > self._limit:
            self._messages = [self._messages[0]] + self._messages[-(self._limit - 1):]

    def messages(self) -> List[Message]:
        return self._messages[:]
