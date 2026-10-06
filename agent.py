#!/usr/bin/env python3
"""Interactive CLI AI agent with conversation history, web search and file reads."""

import base64
import functools
import html
import io
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
import wave
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

try:  # readline is POSIX-only, may be missing on Windows or embedded builds
    import readline
except ImportError:
    readline = None  # type: ignore[assignment]

try:  # termios/tty are POSIX-only and needed to read a bare Enter keypress
    import termios
    import tty
except ImportError:
    termios = None  # type: ignore[assignment]
    tty = None  # type: ignore[assignment]

try:  # select is POSIX-only, used to watch for a keypress without blocking
    import select
except ImportError:
    select = None  # type: ignore[assignment]

# Configuration: model, API endpoint, history limit, search service
MODEL: str = "gemma4:latest"
# The OpenAI-compatible endpoint, not /api/chat: it is the only one that
# accepts an audio prompt, which the model transcribes on its own.
OLLAMA_URL: str = "http://localhost:11434/v1/chat/completions"
OLLAMA_TIMEOUT: int = 300  # audio prompts make the model think before answering
HISTORY_LIMIT: int = 32  # cap history to stay within model context
HISTORY_FILE: str = str(
    Path.home() / ".local" / "share" / "ai-python-agent" / "history"
)
# The request sent to the model on every turn, so the prompt that produced the
# last answer can be read afterwards, system prompt and history included.
LAST_PROMPT_FILE: str = str(
    Path.home() / ".local" / "share" / "ai-python-agent" / "last_prompt.json"
)
DDG_URL: str = "https://html.duckduckgo.com/html/"
# A tool tag counts only on a line of its own, the way the system prompt asks
# for it: a model that quotes the syntax in an answer is not asking for a tool.
SEARCH_RE: re.Pattern[str] = re.compile(
    r"^[ \t]*\[SEARCH:\s*(.+?)\][ \t]*$", re.MULTILINE
)

READ_RE: re.Pattern[str] = re.compile(r"^[ \t]*\[READ:\s*(.+?)\][ \t]*$", re.MULTILINE)
# An image is not read as text: it is attached to the request, so the model can
# look at it.
IMAGE_RE: re.Pattern[str] = re.compile(
    r"^[ \t]*\[IMAGE:\s*(.+?)\][ \t]*$", re.MULTILINE
)
# Image formats the model can look at, by suffix; anything else is refused
# rather than sent as something the model would only fail to decode.
IMAGE_MIME_TYPES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}
IMAGE_MAX_BYTES: int = 5_000_000  # a bigger image is not attached
# How many images stay in the history with their bytes: an image is hundreds of
# kilobytes of base64 in every later request, so only the newest ones are kept
# and the model can ask for an older one again.
IMAGE_HISTORY_KEEP: int = 1
# Said along with the binary files a prompt names, when one of them is an image
# the model could look at.
IMAGE_HINT: str = "An image among them can be attached with [IMAGE: path] to be seen."
# Said in place of an image that was dropped from the history to keep it small.
IMAGE_DROPPED: str = "The image is not attached any more, ask for it again if needed."
# Stands in for the base64 of an image or a recording in the saved prompt.
MEDIA_LOGGED: str = "base64 not saved"
# Where the model may read files: None confines it to the directory the agent
# was started in, a list of directories widens that, an empty list opens the
# reads up to the whole filesystem.
READ_ROOTS: list[str] | None = None
READ_MAX_BYTES: int = 200_000  # a bigger file is sent truncated
# How much of a file is looked at to tell text from binary: a binary file is
# never read any further, the model only gets its path.
READ_PROBE_BYTES: int = 8_192
# A file name written in a prompt, e.g. "blood.jpg", "~/pics/a.png" or
# "data/1.bin": the agent notes the binary ones, since their contents are not
# sent and the model would not know what the prompt was about.
PROMPT_PATH_RE: re.Pattern[str] = re.compile(r"[~\w./\\-]*[\w-]\.[A-Za-z0-9]{1,8}\b")
BINARY_PATH_NOTE: str = (
    "Files named in this prompt that are binary, given by path and size only: {paths}."
)
MAX_TOOL_ROUNDS: int = 4  # tool results fed back per user prompt
# Added to a tool result, since a small model tends to ask for another tool
# instead of answering from what it just got.
ANSWER_FROM_RESULT: str = "Answer from this result, do not write a tool tag."

SYSTEM_PROMPT: str = """You are a helpful assistant with access to the internet
and to the files of the directory the agent was started in.
1st check your internal knowledge for general questions.
2nd when you need real-time info: output exactly [SEARCH: query] on its own line.
3rd when you need the content of a text file: output exactly [READ: path] on its own line.
4th when you need to see an image: output exactly [IMAGE: path] on its own line.
Search and file results will be provided in the next message as numbered title+snippet
pairs, or as the file contents prefixed with line numbers, or as the image itself.
Once a result is provided, answer from it — do not write another tag in the same turn.
Paths are relative to the current directory; `~` is expanded.
Extract ALL facts from all snippets — dates, numbers, names, capabilities, features.
For service/website descriptions: summarize what they offer.
Weather example: if snippet says 'hourly forecast with precipitation/wind/UV',
report that info is available and describe the content.
Never say 'I cannot find' when results are provided.
Only search or read when needed — otherwise use general knowledge.
Accept and print voice prompt in Russian.
Provide respones in Russian"""

MAX_SEARCH_RESULTS: int = 5  # max results to show per search
OPEN_METEO_GEO: str = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_WX: str = "https://api.open-meteo.com/v1/forecast"

WEATHER_RE: re.Pattern[str] = re.compile(
    r"\b(weather|temperature|forecast|rain|snow|wind|humid|cloud|sunny|storm)\b",
    re.IGNORECASE,
)

# Typing one of these sends the prompt of that language in place of the
# command, telling the model that the prompts to follow are written in it.
TRANSLATE_COMMAND: str = "/e"
TRANSLATE_PROMPT: str = "translate next prompts from English to Russian"
TRANSLATE_COMMAND_RU: str = "/r"
TRANSLATE_PROMPT_RU: str = "translate next prompts from Russian to English"

HELP_COMMAND: str = "/?"  # show help for slash commands
HELP_TEXT: str = """Commands:
  /? - show this help
  /q - quit (or empty prompt)
  /s - toggle voice output (on by default)
  /v - record voice prompt
  /e - next prompts English -> answer Russian
  /r - next prompts Russian -> answer English
  /? - show this help
"""

VOICE_COMMAND: str = "/v"  # type the prompt instead of typing it
VOICE_SAMPLE_RATE: int = 16000  # sample rate of the WAV sent to the model
VOICE_BLOCK_MS: int = 100  # microphone read block size
# A pause this long ends the prompt on its own, so it can be dictated
# without touching the keyboard.
VOICE_SILENCE_MS: int = 2000
# What counts as a pause is decided against the room, not against a fixed
# level: the quietest of the last VOICE_NOISE_BLOCKS blocks is the noise
# the microphone hears, and only what is VOICE_NOISE_FACTOR times louder
# than that is speech. A fan, a hiss or a hot mic gain therefore cannot
# drown out the pause, and a quiet microphone is not deafened by the
# threshold either. VOICE_SPEECH_RMS is the last resort for a dead-silent
# input, where there is no noise to compare against.
VOICE_NOISE_BLOCKS: int = 10
VOICE_NOISE_FACTOR: float = 2.5
VOICE_SPEECH_RMS: float = 0.003
# The bar is raised at once when the room gets louder, but lowered only
# slowly, so one unvoiced syllable does not make the noise look like speech.
VOICE_NOISE_FALL: float = 0.1
VOICE_POLL_S: float = 0.1  # how often Enter and the pause are looked at
VOICE_HINT: str = "Voice input needs sounddevice: pip install sounddevice"
# A recording is turned up to this peak before it is sent, because speech
# spoken quietly from across the desk reaches the model's audio tower as a
# whisper it can only guess at, and a guess is what comes back as the
# transcript. The gain is capped, or a recording of nothing but room noise
# would be blown up into noise worth transcribing.
VOICE_PEAK: float = 0.95
VOICE_MAX_GAIN: float = 100.0
# The model transcribes the recording itself, so it is only asked to write
# down what it hears, in the language it hears it in, and thinking is off so
# the reply is short and quick.
TRANSCRIBE_PROMPT: str = (
    "Transcribe the audio verbatim, in the language it is spoken in."
    " Output the transcription only."
)

SPEAK_COMMAND: str = "/s"  # toggle spoken replies
SPEAK_DEFAULT: bool = True  # spoken replies are on unless muted with /s
# One Piper voice per language, picked automatically from the reply's script.
# Any name from `python -m piper.download_voices` works.
PIPER_VOICE_EN: str = "en_US-lessac-medium"
PIPER_VOICE_RU: str = "ru_RU-denis-medium"
PIPER_USE_CUDA: bool = False  # "cuda" needs onnxruntime-gpu, keep False on CPU
TTS_DIR: str = str(Path.home() / ".local" / "share" / "ai-python-agent" / "voices")
TTS_MAX_CHARS: int = 1200  # speak at most this much of a long reply, ~1 minute
TTS_HINT: str = (
    "Voice output needs piper-tts and sounddevice: pip install piper-tts sounddevice"
)

CODE_BLOCK_RE: re.Pattern[str] = re.compile(r"```.*?(?:```|\Z)", re.DOTALL)
MARKDOWN_LINK_RE: re.Pattern[str] = re.compile(r"\[([^\]]*)\]\([^)]*\)")
URL_RE: re.Pattern[str] = re.compile(r"(?:https?://|www\.)\S+")
MARKUP_RE: re.Pattern[str] = re.compile(r"[*_#>~|\[\]]")
CYRILLIC_RE: re.Pattern[str] = re.compile(r"[Ѐ-ӿ]")
LATIN_RE: re.Pattern[str] = re.compile(r"[A-Za-z]")

# Voice and the word spoken in place of a URL, per detected language.
TTS_VOICES: dict[str, str] = {"en": PIPER_VOICE_EN, "ru": PIPER_VOICE_RU}
TTS_LINK_WORDS: dict[str, str] = {"en": "link", "ru": "ссылка"}


def _get_weather(location: str) -> str:
    """Fetch and format current weather for a location using Open-Meteo APIs.

    Uses two steps:
    1. Geocoding API: find lat/lon/name for the given location string.
    2. Forecast API: get current weather (temperature, wind, condition).

    Returns a formatted string with weather info, or empty string on any error.
    """
    try:
        geo_url = (
            OPEN_METEO_GEO
            + "?"
            + urllib.parse.urlencode({"name": location, "count": 1})
        )
        with urllib.request.urlopen(geo_url, timeout=10) as resp:
            geo = json.loads(resp.read().decode("utf-8"))
        results = geo.get("results", [])
        if not results:
            return ""
        lat, lon = results[0]["latitude"], results[0]["longitude"]
        name = results[0].get("name", location)

        wx_url = (
            OPEN_METEO_WX
            + "?"
            + urllib.parse.urlencode(
                {
                    "latitude": lat,
                    "longitude": lon,
                    "current_weather": "true",
                }
            )
        )
        with urllib.request.urlopen(wx_url, timeout=10) as resp:
            wx = json.loads(resp.read().decode("utf-8"))

        cw = wx.get("current_weather", {})
        temp = cw.get("temperature")
        wind = cw.get("windspeed")
        desc = cw.get("weathercode")
        WMO = {
            0: "Clear sky",
            1: "Mainly clear",
            2: "Partly cloudy",
            3: "Overcast",
            45: "Fog",
            48: "Rime fog",
            51: "Light drizzle",
            53: "Moderate drizzle",
            55: "Dense drizzle",
            61: "Slight rain",
            63: "Moderate rain",
            65: "Heavy rain",
            71: "Slight snow",
            73: "Moderate snow",
            75: "Heavy snow",
            80: "Slight rain showers",
            81: "Moderate rain showers",
            82: "Violent rain showers",
            95: "Thunderstorm",
            96: "Thunderstorm with hail",
            99: "Severe thunderstorm with hail",
        }
        condition = WMO.get(desc, f"Code {desc}") if desc is not None else "Unknown"

        lines = [f"Current weather in {name}:"]
        if temp is not None:
            lines.append(f"  Temperature: {temp}°C")
        lines.append(f"  Condition: {condition}")
        if wind is not None:
            lines.append(f"  Wind speed: {wind} km/h")
        return "\n".join(lines)
    except (urllib.error.URLError, OSError, KeyError, json.JSONDecodeError):
        return ""


def search_web(query: str) -> str:
    """Search the web for the given query using DuckDuckGo, optionally fetching weather.

    Returns a formatted string with up to MAX_SEARCH_RESULTS (5) entries:
    each entry has a numbered title and a snippet.

    If the query mentions weather (via WEATHER_RE), extracts the location string
    from the query (stripping common prepositions and country names), then calls
    _get_weather() and prepends the weather info to the results.

    Returns:
        Formatted search results string, or error message if the search fails.
    """
    data = urllib.parse.urlencode({"q": query}).encode("utf-8")
    req = urllib.request.Request(
        DDG_URL,
        data=data,
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
            "Content-Type": "application/x-www-form-urlencoded",
            "Referer": "https://html.duckduckgo.com/",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            page = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError) as e:
        return f"Search error: {e}"

    titles = re.findall(r'class="result__a"[^>]*>(.*?)</a>', page)
    snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', page)

    results: list[str] = []
    for i in range(min(len(titles), len(snippets), MAX_SEARCH_RESULTS)):
        title = html.unescape(re.sub(r"<[^>]+>", "", titles[i])).strip()
        snippet = html.unescape(re.sub(r"<[^>]+>", "", snippets[i])).strip()
        results.append(f"{i + 1}. {title}\n   {snippet}")

    if not results:
        return "No search results found."

    output = "Search results:\n" + "\n".join(results)

    if WEATHER_RE.search(query):
        location = re.sub(WEATHER_RE, "", query).strip()
        location = re.sub(
            r"\b(in|at|for|today|now|tomorrow|yesterday|this|current|russia|usa|uk|canada|germany|france|china|japan|india|australia|brazil)\b",
            "",
            location,
            flags=re.IGNORECASE,
        )
        location = re.sub(r"[,]", " ", location)
        location = re.sub(r"\s+", " ", location).strip(" ,.-")
        if location:
            wx = _get_weather(location)
            if wx:
                output = wx + "\n\n" + output

    return output


def resolve_read_path(path: str) -> tuple[Path | None, str]:
    """Resolve a path the model or the user named, keeping it inside the roots.

    A relative path is taken from the first READ_ROOTS directory, `~` is
    expanded, and symlinks are followed, but the result must still sit inside
    one of the READ_ROOTS directories, so neither can walk out with `../`.

    Args:
        path: the file to reach, as it was written.

    Returns:
        The resolved file, or None and the reason it is not readable.
    """
    roots = [Path.cwd()] if READ_ROOTS is None else READ_ROOTS
    allowed = [Path(root).expanduser().resolve() for root in roots]
    base = allowed[0] if allowed else Path.cwd()
    try:
        target = Path(path).expanduser()
        target = (target if target.is_absolute() else base / target).resolve()
    except (OSError, ValueError) as e:
        return None, str(e)
    if not any(target == root or root in target.parents for root in allowed):
        return None, f"outside {', '.join(str(r) for r in allowed)}"
    if target.is_dir():
        return None, "it is a directory"
    return target, ""


def is_binary(target: Path) -> bool:
    """Tell a binary file from a text one by the first bytes of its content.

    Only READ_PROBE_BYTES are looked at, and a file that cannot be opened is
    not binary, so the caller reports it as unreadable instead.

    Args:
        target: an already resolved path.

    Returns:
        True if the file looks binary, False if it looks like text or is
        missing.
    """
    try:
        with target.open("rb") as handle:
            return b"\0" in handle.read(READ_PROBE_BYTES)
    except OSError:
        return False


def binary_paths_note(prompt: str) -> str:
    """Add the paths of the binary files a prompt names, with their sizes.

    The contents of a binary file never reach the model, so a prompt such as
    "what is in blood.jpg?" would otherwise leave it guessing which file is
    meant. The path and the size are appended to the message instead, plus a
    hint when one of them is an image the model can ask to see. A name that is
    not a file, or a text file, is left out.

    Args:
        prompt: the prompt as typed or transcribed.

    Returns:
        A note to append to the message, or an empty string when the prompt
        names no binary file.
    """
    noted: list[str] = []
    seeable: bool = False
    for name in dict.fromkeys(PROMPT_PATH_RE.findall(prompt)):
        target, _ = resolve_read_path(name)
        if target is None or not is_binary(target):
            continue
        try:
            size = target.stat().st_size
        except OSError:
            continue
        noted.append(f"{name} ({size} bytes)")
        seeable = seeable or target.suffix.lower() in IMAGE_MIME_TYPES
    if not noted:
        return ""
    note = BINARY_PATH_NOTE.format(paths=", ".join(noted))
    return f"\n{note} {IMAGE_HINT}" if seeable else f"\n{note}"


def read_file(path: str) -> str:
    """Read a local file on behalf of the model and return its contents.

    The path is resolved inside the READ_ROOTS directories (see
    resolve_read_path). Directories and unreadable files are reported as a
    short sentence the model can report back, a binary file is not read at all
    and only its path and size are sent, and a file bigger than READ_MAX_BYTES
    is sent truncated rather than dropped.

    Args:
        path: the file to read, as the model asked for it.

    Returns:
        The file contents prefixed with line numbers, or a message explaining
        why it could not be read.
    """
    target, reason = resolve_read_path(path)
    if target is None:
        return f"Cannot read {path}: {reason}"
    if is_binary(target):
        return (
            f"{path} is a binary file of {target.stat().st_size} bytes; it was not "
            f"read, only its path {target} is known"
        )
    try:
        with target.open("rb") as handle:
            data = handle.read(READ_MAX_BYTES + 1)
    except OSError as e:
        return f"Cannot read {path}: {e.strerror or e}"
    truncated = len(data) > READ_MAX_BYTES
    text = data[:READ_MAX_BYTES].decode("utf-8", errors="replace")
    lines = text.splitlines()
    numbered = "\n".join(f"{i}: {line}" for i, line in enumerate(lines, 1))
    note = "\n... truncated, file is bigger than READ_MAX_BYTES" if truncated else ""
    return f"Contents of {path} ({len(lines)} lines):\n{numbered}{note}"


def read_image(path: str) -> str | list[dict[str, Any]]:
    """Attach an image file so the model can look at it.

    The model takes images as base64 data URLs in an `image_url` part of a
    message, which is what this builds: the file is read from inside the
    READ_ROOTS directories (see resolve_read_path) and returned as the content
    parts of the next message. A path outside the roots, a file that is not one
    of the IMAGE_MIME_TYPES formats, or one bigger than IMAGE_MAX_BYTES is
    reported as a short sentence the model can pass on.

    Args:
        path: the image to attach, as the model asked for it.

    Returns:
        The content parts of the next message, with the image attached, or a
        message explaining why it could not be attached.
    """
    target, reason = resolve_read_path(path)
    if target is None:
        return f"Cannot attach {path}: {reason}"
    mime = IMAGE_MIME_TYPES.get(target.suffix.lower())
    if mime is None:
        known = ", ".join(sorted(IMAGE_MIME_TYPES))
        return f"Cannot attach {path}: not an image format ({known} are supported)"
    try:
        size = target.stat().st_size
        if size > IMAGE_MAX_BYTES:
            return (
                f"Cannot attach {path}: it is {size} bytes, at most "
                f"{IMAGE_MAX_BYTES} are attached — tell the user to make it smaller"
            )
        encoded = base64.b64encode(target.read_bytes()).decode("ascii")
    except OSError as e:
        return f"Cannot attach {path}: {e.strerror or e}"
    return [
        {"type": "text", "text": f"Image: {path} ({mime}, {size} bytes)"},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
    ]


def image_note(parts: list[dict[str, Any]]) -> str:
    """Return the text of a message built by read_image, without the image.

    Args:
        parts: the content parts of the message.

    Returns:
        The text of the first text part, or an empty string.
    """
    texts = [part.get("text", "") for part in parts if part.get("type") == "text"]
    return texts[0] if texts else ""


def drop_stale_images(messages: list[dict[str, Any]]) -> None:
    """Keep the bytes of the newest IMAGE_HISTORY_KEEP images only.

    An image costs hundreds of kilobytes of base64 in every request made after
    it, so the data of the older ones is dropped and their text note says so:
    the model can ask for such a file again with [IMAGE: path]. A limit of 0
    drops every image as soon as the turn is answered.

    Args:
        messages: the conversation history, edited in place.

    Returns:
        None
    """
    if IMAGE_HISTORY_KEEP <= 0:
        return
    attached = [
        message
        for message in messages
        if isinstance(message["content"], list)
        and any(part.get("type") == "image_url" for part in message["content"])
    ]
    for message in attached[:-IMAGE_HISTORY_KEEP]:
        note = image_note(message["content"])
        message["content"] = [{"type": "text", "text": f"{note}\n{IMAGE_DROPPED}"}]


# Tools the model can ask for by writing a tag into its reply: the tag name, the
# pattern that recognizes it, and the function that carries it out. Each returns
# the content of the next message: text for the tools that answer with text,
# content parts for the ones that attach an image.
TOOLS: dict[
    str, tuple[re.Pattern[str], Callable[[str], str | list[dict[str, Any]]]]
] = {
    "SEARCH": (SEARCH_RE, search_web),
    "READ": (READ_RE, read_file),
    "IMAGE": (IMAGE_RE, read_image),
}


def chat(messages: list[dict[str, Any]], session_id: str) -> str:
    """Send a chat request to Ollama and return the model's response.

    The request is also written to LAST_PROMPT_FILE, so the prompt behind the
    answer can be read afterwards.

    Args:
        messages: list of message dicts with "role" ("system" | "user" | "assistant")
                  and "content" keys.
        session_id: unique session identifier for the conversation.

    Returns:
        The model's response content.

    Raises:
        URLError, KeyError, or json.JSONDecodeError if the request fails.
    """
    request = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "session_id": session_id,
    }
    save_last_prompt(request)
    payload = json.dumps(request).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    return data["choices"][0]["message"]["content"]


def without_media_data(content: Any) -> Any:
    """Copy a message content with the payload of its media parts taken out.

    An image or a recording is a long base64 string that would make the saved
    prompt unreadable and huge, so each media part keeps its type and a
    placeholder. The text parts are left as they are.

    Args:
        content: the content of a message, text or a list of parts.

    Returns:
        The same content with the media payloads replaced, the very same
        object when it is plain text.
    """
    if not isinstance(content, list):
        return content
    parts: list[dict[str, Any]] = []
    for part in content:
        if part.get("type") == "image_url":
            parts.append(
                {"type": "image_url", "image_url": {"url": f"data:...{MEDIA_LOGGED}"}}
            )
        elif part.get("type") == "input_audio":
            audio = {**part.get("input_audio", {})}
            audio["data"] = MEDIA_LOGGED
            parts.append({"type": "input_audio", "input_audio": audio})
        else:
            parts.append(dict(part))
    return parts


def save_last_prompt(request: dict[str, Any]) -> None:
    """Write the request sent to the model to LAST_PROMPT_FILE.

    The file is replaced on every request, so it always holds the prompt the
    last answer was made from: the system prompt, the whole history and the
    last user message, as JSON. It is written indented and unescaped, to stay
    readable, and the base64 of an image or a recording is replaced by a
    placeholder, which would otherwise fill the whole file. A file that cannot
    be written is ignored, since keeping the prompt must never break the
    dialog.

    Args:
        request: the request body sent to Ollama.

    Returns:
        None
    """
    logged = {
        **request,
        "messages": [
            {**message, "content": without_media_data(message["content"])}
            for message in request.get("messages", [])
        ],
    }
    try:
        path = Path(LAST_PROMPT_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(logged, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError:
        pass


def init_input_history() -> None:
    """Enable readline line editing and restore prompts from previous sessions.

    With readline available, Up/Down arrows walk through previously submitted
    prompts. The history is loaded from HISTORY_FILE, so prompts entered in
    earlier runs of the agent are available too, and is limited to
    HISTORY_LIMIT entries.
    """
    if readline is None:
        return
    readline.set_history_length(HISTORY_LIMIT)
    load_input_history()


def load_input_history() -> None:
    """Load prompts from previous sessions into the readline history.

    Reads HISTORY_FILE, keeping only the most recent HISTORY_LIMIT entries.
    A missing or unreadable file is not an error: the agent simply starts
    with an empty history.
    """
    if readline is None:
        return
    try:
        readline.read_history_file(HISTORY_FILE)
    except OSError:
        return
    # Older files may hold more entries than the cap: drop the oldest ones.
    while readline.get_current_history_length() > HISTORY_LIMIT:
        readline.remove_history_item(0)


def save_input_history() -> None:
    """Persist the prompt history for the next session.

    Writes HISTORY_FILE, creating its directory when needed. Failures are
    ignored: losing the history must never break the agent.
    """
    if readline is None:
        return
    try:
        Path(HISTORY_FILE).parent.mkdir(parents=True, exist_ok=True)
        readline.write_history_file(HISTORY_FILE)
    except OSError:
        pass


def read_prompt(prompt: str = "You: ") -> str:
    """Read one line from stdin, recording it in the command history.

    Args:
        prompt: text shown before the input cursor.

    Returns:
        The entered line, or an empty string on EOF (Ctrl+D) or Ctrl+C.
    """
    try:
        line = input(prompt)
    except (EOFError, KeyboardInterrupt):
        print()
        return ""

    remember_prompt(line)
    return line


def remember_prompt(line: str) -> None:
    """Add a submitted prompt to the readline history.

    Blank lines, the quit command and immediate repeats (Up + Enter) are
    skipped, so the history stays free of noise.

    Args:
        line: the prompt to remember, as typed or transcribed.

    Returns:
        None
    """
    if readline is None:
        return
    stripped = line.strip()
    if not stripped or stripped == "/q" or stripped == HELP_COMMAND or stripped == SPEAK_COMMAND:
        return
    length = readline.get_current_history_length()
    if not length or readline.get_history_item(length) != line:
        readline.add_history(line)


class PauseWatcher:
    """Notice the pause that ends a dictated prompt.

    Whether a block is speech is decided against the room instead of against
    a fixed level: the quietest of the last VOICE_NOISE_BLOCKS blocks is the
    noise the microphone hears, and a block has to be VOICE_NOISE_FACTOR
    times louder than that to count as speech. Room noise — a fan, a hiss, a
    microphone gain turned up — therefore stays silence, and a quiet
    microphone is not made deaf by the threshold. A loud block restarts the
    clock; while nothing but quieter blocks arrive the pause keeps growing,
    and once it reaches VOICE_SILENCE_MS the recording is over, so the
    prompt can be dictated without pressing Enter. The clock itself only
    starts at the first word, and silence before it ends nothing, or a
    quiet room would stop the recording before anything was said. The first
    VOICE_NOISE_BLOCKS blocks only go to measuring the room, since a word
    cannot be told from the room before the room is known.
    """

    def __init__(self) -> None:
        self.levels: deque[float] = deque(maxlen=VOICE_NOISE_BLOCKS)
        self.blocks: int = 0  # blocks seen
        self.spoken: bool = False  # a word was heard, so the pause means one
        self.voice_blocks: int = 0  # blocks up to and including the last word
        self.silent_ms: int = 0  # how long the pause has been going on
        self.threshold: float = VOICE_SPEECH_RMS  # level that counts as speech

    @property
    def noise(self) -> float:
        """What the room itself sounds like, as the recent blocks see it.

        The quietest of the last VOICE_NOISE_BLOCKS blocks, with the very
        quietest fifth ignored: a block that dropped to nothing would
        otherwise pass for a silent room and let the noise look like speech.
        """
        levels = sorted(self.levels)
        return levels[len(levels) // 5] if levels else 0.0

    def _follow(self) -> None:
        """Move the speech bar towards what the room turns out to be."""
        wanted = max(VOICE_SPEECH_RMS, self.noise * VOICE_NOISE_FACTOR)
        if wanted > self.threshold:
            self.threshold = wanted  # louder room: raise the bar at once
        else:
            self.threshold += (wanted - self.threshold) * VOICE_NOISE_FALL

    def add(self, level: float, samples: int) -> None:
        """Account one recorded block of `samples` values at RMS `level`.

        Args:
            level: RMS level of the block, 0.0 to 1.0.
            samples: number of samples in the block.

        Returns:
            None
        """
        self.blocks += 1
        self.levels.append(level)
        # The level is weighed against the bar the previous blocks left
        # behind, so a block cannot pass itself off as the noise it is, and
        # the opening blocks only measure the room: a microphone that comes
        # up with a bang in its first block would otherwise pass for a word
        # and set the level the whole recording is judged against too high
        # to ever hear another one.
        loud = self.blocks > VOICE_NOISE_BLOCKS and level > self.threshold
        if loud:
            self.spoken = True
            self.voice_blocks = self.blocks
            self.silent_ms = 0
        elif self.spoken:
            # The clock only runs once a word has been heard: the silence
            # before the first word belongs to no pause, so it is not counted.
            self.silent_ms += samples * 1000 // VOICE_SAMPLE_RATE
        self._follow()

    @property
    def paused(self) -> bool:
        """True once the speaker has paused long enough to end the prompt."""
        return self.spoken and self.silent_ms >= VOICE_SILENCE_MS


def wait_for_stop(is_done: Callable[[], bool]) -> bool:
    """Block until Enter is pressed or `is_done` turns true.

    The terminal is switched to cbreak mode, so a single keypress is
    delivered without waiting for a newline while Ctrl+C keeps working.
    Stdin is polled instead of read in one blocking go, so the pause of a
    voice prompt can end the wait on its own. Every keypress that comes in
    while the wait is on belongs to the recording: a prompt typed ahead of
    it is swallowed rather than answered after the recording. When terminal
    control is unavailable the call falls back to reading a whole line, and
    only Enter stops the wait.

    Args:
        is_done: called between keypresses; when it turns true the wait
                 ends even if nothing was typed.

    Returns:
        True if Enter (or the end of input) stopped the wait, False if
        `is_done` did.
    """
    if termios is None or tty is None or select is None:
        sys.stdin.readline()
        return True
    fd = sys.stdin.fileno()
    try:
        saved = termios.tcgetattr(fd)
    except (termios.error, OSError, ValueError):  # not a terminal (piped input)
        sys.stdin.readline()
        return True
    try:
        tty.setcbreak(fd)
        while True:
            if is_done():
                # A keypress that raced with the pause belongs to the
                # recording, not to the prompt that comes after it.
                while select.select([fd], [], [], 0)[0]:
                    if os.read(fd, 1) in (b"", b"\r", b"\n"):
                        break
                return False
            if select.select([fd], [], [], VOICE_POLL_S)[0]:
                # Straight from the terminal, never through sys.stdin: the
                # buffered reader takes every byte that is waiting at once,
                # and select then reports nothing while the rest of what was
                # typed sits unread in the buffer, so the wait would never end.
                char = os.read(fd, 1)
                if char in (b"", b"\r", b"\n"):  # Enter, or end of input
                    return True
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def record_voice() -> Any:
    """Record mono 16 kHz audio from the default microphone.

    Recording ends on Enter or on a pause of VOICE_SILENCE_MS, whichever
    comes first, and the silence of the closing pause is left out of the
    samples. A recording without a word in it is not returned at all.

    Returns:
        A numpy float32 array with the recorded samples, empty if nothing
        was captured.

    Raises:
        ImportError: if sounddevice is not installed.
        OSError: if the microphone cannot be opened.
    """
    import numpy as np
    import sounddevice as sd

    blocks: list[Any] = []
    watcher = PauseWatcher()

    def callback(indata: Any, frames: int, time_info: Any, status: Any) -> None:
        # A raw stream hands out a buffer over memory that PortAudio reuses
        # for the next block, so the samples have to be copied out.
        block = np.frombuffer(indata, dtype="float32").copy()
        blocks.append(block)
        watcher.add(float(np.sqrt(np.mean(block * block))), len(block))

    block_size = int(VOICE_SAMPLE_RATE * VOICE_BLOCK_MS / 1000)
    try:
        with sd.RawInputStream(
            samplerate=VOICE_SAMPLE_RATE,
            blocksize=block_size,
            dtype="float32",
            channels=1,
            callback=callback,
        ):
            print(
                "Recording... press Enter to stop, a"
                f" {VOICE_SILENCE_MS // 1000}s pause ends it, Ctrl+C to cancel",
                flush=True,
            )
            by_enter = wait_for_stop(lambda: watcher.paused)
    except sd.PortAudioError as e:
        raise OSError(f"cannot open microphone: {e}") from e

    if not blocks:
        return np.empty(0, dtype="float32")
    if not by_enter:
        print("Heard a pause, ending the prompt.", flush=True)
    if not watcher.spoken:
        # Nothing was said, so the pause could never have been noticed. The
        # room is not sent to the model either: it would only come back as a
        # word guessed out of the noise. The levels involved are printed
        # instead, to make a threshold easy to tune.
        print(
            f"Nothing said over the noise: {watcher.noise:.4f} is the noise,"
            f" {watcher.threshold:.4f} is what counts as speech.",
            flush=True,
        )
        return np.empty(0, dtype="float32")
    # Whatever came after the last word is silence, so it is not sent.
    return np.concatenate(blocks[: watcher.voice_blocks])


def encode_wav(audio: Any) -> bytes:
    """Wrap recorded samples into a WAV file held in memory.

    The microphone hands out float samples, while the model wants a WAV
    container, so the samples are scaled to signed 16-bit PCM and written
    into a 44-byte header. A quiet recording is first turned up to
    VOICE_PEAK, since the model transcribes a whisper badly.

    Args:
        audio: mono float32 samples at VOICE_SAMPLE_RATE, as returned by
               record_voice().

    Returns:
        The recording as a WAV file, ready to be sent to the model.

    Raises:
        ImportError: if numpy is not installed.
    """
    import numpy as np

    pcm = np.clip(audio, -1.0, 1.0)
    peak = float(np.max(np.abs(pcm))) if pcm.size else 0.0
    if 0.0 < peak < VOICE_PEAK:
        pcm = np.clip(pcm * min(VOICE_PEAK / peak, VOICE_MAX_GAIN), -1.0, 1.0)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(VOICE_SAMPLE_RATE)
        wav.writeframes((pcm * 32767).astype("<i2").tobytes())
    return buffer.getvalue()


def transcribe_voice(audio: Any) -> str:
    """Let the model transcribe the recording, by sending it the WAV itself.

    The audio is attached to a single request as an `input_audio` block, so
    no local speech recognition is involved. The reply of that request is
    the transcript, which then takes the place of the recording in the
    conversation history.

    Args:
        audio: mono float32 samples at VOICE_SAMPLE_RATE, as returned by
               record_voice().

    Returns:
        The recognized text, stripped of surrounding whitespace.

    Raises:
        URLError, KeyError, or json.JSONDecodeError if the request fails.
    """
    encoded = base64.b64encode(encode_wav(audio)).decode("ascii")
    payload = json.dumps(
        {
            "model": MODEL,
            "stream": False,
            "reasoning_effort": "none",
            "options": {"temperature": 0},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": TRANSCRIBE_PROMPT},
                        {
                            "type": "input_audio",
                            "input_audio": {"data": encoded, "format": "wav"},
                        },
                    ],
                }
            ],
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    return data["choices"][0]["message"]["content"].strip()


def voice_prompt() -> str:
    """Record a prompt by voice and return its transcription.

    Handles a missing microphone, missing packages, a model that cannot be
    reached and Ctrl+C, reporting them to the user instead of raising, so
    the dialog can continue.

    Returns:
        The transcribed prompt, or an empty string if nothing was captured.
    """
    try:
        audio = record_voice()
        seconds = len(audio) / VOICE_SAMPLE_RATE
        if not seconds:
            print("Nothing recorded.", flush=True)
            return ""
        print(f"Recorded {seconds:.1f}s, transcribing with the model...", flush=True)
        text = transcribe_voice(audio)
    except ImportError as e:
        print(f"Error: {VOICE_HINT} (missing: {e.name})")
        return ""
    except KeyboardInterrupt:
        print("\nRecording cancelled.")
        return ""
    except (OSError, RuntimeError, ValueError) as e:
        print(f"Error: voice input failed ({e})")
        return ""

    if not text:
        print("Nothing recognized, try again.", flush=True)
        return ""
    print(f"You (voice): {text}")
    remember_prompt(text)
    return text


def detect_language(text: str) -> str:
    """Guess the language of a reply from the script it is written in.

    Russian is spoken with a Russian voice, everything else with the English
    one. A reply that mixes both is named after the script it mostly uses, so
    a Russian answer with a couple of English words still gets a Russian voice.

    Args:
        text: the raw assistant reply.

    Returns:
        "ru" for mostly Cyrillic text, "en" otherwise.
    """
    if len(CYRILLIC_RE.findall(text)) > len(LATIN_RE.findall(text)):
        return "ru"
    return "en"


def speakable_text(text: str, link_word: str = "link") -> str:
    """Turn an assistant reply into plain text a TTS voice can read out.

    Code blocks are dropped and raw URLs are replaced by link_word, since a
    voice reading them aloud is noise; inline markup is stripped and the
    result is whitespace-collapsed and capped at TTS_MAX_CHARS, cutting on a
    word boundary.

    Args:
        text: the raw assistant reply.
        link_word: what to say instead of a URL, in the spoken language.

    Returns:
        The speakable part of the reply, possibly an empty string.
    """
    text = CODE_BLOCK_RE.sub(" ", text)
    text = MARKDOWN_LINK_RE.sub(r"\1", text)
    text = URL_RE.sub(f" {link_word} ", text)
    text = text.replace("`", " ")
    text = MARKUP_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > TTS_MAX_CHARS:
        text = text[:TTS_MAX_CHARS].rsplit(" ", 1)[0] + ", and more."
    return text


@functools.cache
def load_piper(voice: str) -> Any:
    """Load a Piper TTS voice, reusing it on later calls.

    The voice (an ONNX model plus its config) is downloaded to TTS_DIR on the
    first call and cached on disk, so following calls (and later sessions)
    work without internet access. Voices are cached one by one, so only the
    languages actually spoken are downloaded.

    Args:
        voice: Piper voice name, e.g. PIPER_VOICE_EN.

    Returns:
        A piper.PiperVoice instance.

    Raises:
        ImportError: if piper-tts is not installed.
        OSError: if the voice cannot be downloaded.
    """
    from piper import PiperVoice
    from piper.download_voices import download_voice

    voice_dir = Path(TTS_DIR)
    voice_dir.mkdir(parents=True, exist_ok=True)
    model_path = voice_dir / f"{voice}.onnx"
    config_path = voice_dir / f"{voice}.onnx.json"
    if not (model_path.exists() and config_path.exists()):
        print(f"Downloading voice {voice} (once, ~60 MB)...", flush=True)
        download_voice(voice, voice_dir)
    return PiperVoice.load(
        model_path,
        config_path=config_path,
        use_cuda=PIPER_USE_CUDA,
        download_dir=voice_dir,
    )


def speak(text: str) -> None:
    """Read the reply out loud with Piper, in the language it is written in.

    The voice matching the reply's script is used and loaded on first use.
    Playback blocks until it finishes, so the next prompt appears only when the
    voice is done. Ctrl+C stops the audio and leaves the dialog.

    Args:
        text: the assistant reply; only its speakable part is rendered.

    Returns:
        None

    Raises:
        ImportError: if piper-tts or sounddevice is not installed.
        OSError, RuntimeError, ValueError: if synthesis or playback fails.
    """
    language = detect_language(text)
    spoken = speakable_text(text, TTS_LINK_WORDS[language])
    if not spoken:
        return
    import sounddevice as sd

    chunks = list(load_piper(TTS_VOICES[language]).synthesize(spoken))
    if not chunks:
        return
    try:
        for chunk in chunks:
            sd.play(chunk.audio_float_array, chunk.sample_rate)
            sd.wait()
    finally:
        sd.stop()


def enable_voice_output() -> bool:
    """Turn spoken replies on and preload the default voice.

    Only the default language is loaded, so enabling is quick; a voice for
    another language is downloaded and loaded the first time it is needed.

    Returns:
        True if voice output is ready to use, False if it could not be enabled
        (the reason is printed).
    """
    try:
        import sounddevice  # noqa: F401  # verify playback is possible

        load_piper(PIPER_VOICE_EN)
    except ImportError as e:
        print(f"Error: {TTS_HINT} (missing: {e.name})")
        return False
    except (OSError, RuntimeError, ValueError) as e:
        print(f"Error: cannot enable voice output ({e})")
        return False
    print(
        f"Voice output on — {PIPER_VOICE_EN} / {PIPER_VOICE_RU}, "
        f"picked per reply. Type {SPEAK_COMMAND} again to mute."
    )
    return True


def ask_model(messages: list[dict[str, Any]], session_id: str) -> str | None:
    """Send the history to the model and report the usual failures.

    Args:
        messages: the conversation history, sent as is.
        session_id: unique session identifier for the conversation.

    Returns:
        The model's reply, or None if the model could not be reached or
        answered something unexpected (the reason is printed).
    """
    try:
        return chat(messages, session_id)
    except urllib.error.URLError as e:
        print(f"Error: cannot reach Ollama ({e.reason})")
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Error: unexpected response from Ollama ({e})")
    return None


def agent_turn(messages: list[dict[str, Any]], session_id: str) -> str | None:
    """Answer the last user message, running the tools the model asks for.

    A reply carrying a tool tag such as [SEARCH: query] or [READ: path] is not
    shown as the answer: the tool runs, its result is appended to the history
    and the model is asked again, up to MAX_TOOL_ROUNDS times. The reply that
    carries no tag is the answer, and the history is trimmed afterwards.

    Args:
        messages: the conversation history, extended in place.
        session_id: unique session identifier for the conversation.

    Returns:
        The answer to show to the user, or None if a request failed (the
        reason is printed and the dialog ends).
    """
    reply = ask_model(messages, session_id)
    if reply is None:
        return None
    messages.append({"role": "assistant", "content": reply})

    for _ in range(MAX_TOOL_ROUNDS):
        call = next(
            (
                (name, tool, match.group(1))
                for name, (pattern, tool) in TOOLS.items()
                if (match := pattern.search(reply))
            ),
            None,
        )
        if call is None:
            break
        name, tool, argument = call
        print(f"[{name.lower()}] {argument}", flush=True)
        # A tool answers with text, or with content parts when it attaches an
        # image. Either way a reminder to answer follows, since a small model
        # tends to write another tag instead of answering.
        result = tool(argument)
        if isinstance(result, str):
            result = f"{result}\n\n{ANSWER_FROM_RESULT}"
        else:
            result = [*result, {"type": "text", "text": ANSWER_FROM_RESULT}]
        messages.append({"role": "user", "content": result})
        reply = ask_model(messages, session_id)
        if reply is None:
            return None
        messages.append({"role": "assistant", "content": reply})

    # Enforce history limit by dropping oldest messages after index 1
    if len(messages) > HISTORY_LIMIT:
        del messages[1 : len(messages) - (HISTORY_LIMIT - 1)]
    drop_stale_images(messages)

    return reply


def main() -> None:
    """Main entry point for the interactive CLI agent.

    Runs an infinite loop prompting for user input, sending the full conversation
    history to Ollama, and displaying the assistant's response.

    The loop exits on:
      - empty prompt (just press Enter)
      - /q command
      - Ctrl+C

    Up/Down arrows recall previously entered commands, including prompts from
    earlier sessions, when readline is available. Typing /v instead of a prompt
    records it from the microphone, has the model transcribe the recording and
    sends the transcript as the prompt. Spoken replies are on from the start:
    every answer is read out loud with Piper in addition to being printed, and
    typing /s mutes and unmutes them. Typing /e sends TRANSLATE_PROMPT to the
    model, which then answers the English prompts to follow in Russian; /r sends
    TRANSLATE_PROMPT_RU, for Russian prompts answered in English.

    Args:
        None

    Returns:
        None
    """
    session_id = uuid.uuid4().hex
    init_input_history()
    # Initialize conversation with system prompt and at least one dummy message
    # to ensure context is available on first user input
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    print(f"AI agent — using model: {MODEL}")
    print(f"Type {VOICE_COMMAND} to speak your prompt, or {HELP_COMMAND} for help.")
    print("Press Enter with an empty prompt to exit.\n")
    # Spoken replies start on unless SPEAK_DEFAULT says otherwise; /s mutes
    # and unmutes them, so a missing voice package only mutes the dialog
    speak_replies = enable_voice_output() if SPEAK_DEFAULT else False

    try:
        while True:
            line = read_prompt()

            # Exit conditions: empty input or /q command
            if not line or line.strip() == "/q":
                break

            # Language command: send its prompt in place of the command, so the
            # model is told which language the prompts to follow are written in
            stripped = line.strip()
            if stripped == TRANSLATE_COMMAND:
                line = TRANSLATE_PROMPT
            elif stripped == TRANSLATE_COMMAND_RU:
                line = TRANSLATE_PROMPT_RU

            # Help: show available commands
            if stripped == HELP_COMMAND:
                print(HELP_TEXT)
                continue

            # Voice output toggle: speak every reply, or mute again
            if stripped == SPEAK_COMMAND:
                if speak_replies:
                    speak_replies = False
                    print("Voice output off.\n")
                else:
                    speak_replies = enable_voice_output()
                    print()
                continue

            # Voice prompt: replace the typed command with its transcription
            if line.strip() == VOICE_COMMAND:
                line = voice_prompt()
                if not line:
                    continue

            # Add user message to conversation history, telling the model which
            # of the files it names are binary, since they are not read.
            messages.append({"role": "user", "content": line + binary_paths_note(line)})

            print("Thinking...", flush=True)

            # Answer, running whatever tools the model asks for on the way
            reply = agent_turn(messages, session_id)
            if reply is None:
                break

            # Debug: print full message history (uncomment to inspect)
            # print(messages)

            print(f"Assistant: {reply}\n")

            # Read the reply out loud when voice output is on; a TTS problem is
            # reported and mutes the voice, but never breaks the dialog.
            if speak_replies:
                try:
                    speak(reply)
                except ImportError as e:
                    print(f"Error: {TTS_HINT} (missing: {e.name})")
                    speak_replies = False
                except KeyboardInterrupt:
                    break
                except (OSError, RuntimeError, ValueError) as e:
                    print(f"Error: voice output failed ({e})")
                    speak_replies = False
    finally:
        save_input_history()
        print("Bye!")


if __name__ == "__main__":
    main()
