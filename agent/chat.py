#!/usr/bin/env python3
"""Chat interface and Ollama implementation using Abstract Factory."""

from abc import ABC, abstractmethod
from typing import List, Any, Union
import html
import json
import re
import uuid
import urllib.error
import urllib.parse
import urllib.request
from datetime import timedelta


class ChatRequest:
    """DTO for chat request."""

    Model: str
    Messages: List[Any]
    Stream: bool
    SessionId: str

    def __init__(self, model: str, messages: List[Any], stream: bool, session_id: str):
        self.Model = model
        self.Messages = messages
        self.Stream = stream
        self.SessionId = session_id


class ChatResponse:
    """DTO for chat response."""

    Message: Any

    def __init__(self, message: Any):
        self.Message = message

    def get_content(self) -> str:
        return self.Message['content']


class ChatSession(ABC):
    """Abstract interface for chat sessions."""

    @abstractmethod
    def chat(self, history: Any, response) -> str:
        """Send chat request and return response."""
        pass


class OllamaChatSession(ChatSession):
    """Ollama chat implementation using Ollama API."""

    Model: str = 'gemma4:latest'
    # The OpenAI-compatible endpoint, not /api/chat: it is the only one that
    # accepts an audio prompt, which the model transcribes on its own.
    Url: str = 'http://localhost:11434/v1/chat/completions'
    SessionId: str = uuid.uuid4().hex

    def chat(self, history, response) -> str:
        payload = json.dumps({
            'model': self.Model,
            'messages': self._serialize_messages(history),
            'stream': self.Stream,
            'session_id': self.SessionId,
        }).encode('utf-8')

        req = urllib.request.Request(
            self.Url,
            data=payload,
            headers={'Content-Type': 'application/json'},
            method='POST',
        )

        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode('utf-8'))

        return data['choices'][0]['message']['content']

    def _serialize_messages(self, history) -> List[dict]:
        """Convert history objects to dict format."""
        return [{'role': m.role, 'content': m.content} for m in history]


class ChatFactory:
    """Abstract Factory for creating chat sessions."""

    @staticmethod
    def create(url: str, model: str, timeout: int = 120) -> ChatSession:
        """Create a chat session for the provider."""
        pass


class DefaultChatFactory(ChatFactory):
    """Simple factory for Ollama implementation."""

    @staticmethod
    def create(url: str, model: str, timeout: int = 120) -> ChatSession:
        """Create Ollama chat session via Simple Factory pattern."""
        session = OllamaChatSession()
        session._Url = url
        session._Model = model
        session._Stream = False
        session._Timeout = timeout
        return session
