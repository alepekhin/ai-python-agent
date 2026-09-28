#!/usr/bin/env python3
"""Interactive CLI AI agent with conversation history and web search via Ollama."""

import base64
import functools
import html
import io
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
import wave
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
DDG_URL: str = "https://html.duckduckgo.com/html/"
SEARCH_RE: re.Pattern[str] = re.compile(r"\[SEARCH:\s*(.+?)\]")

SYSTEM_PROMPT: str = """You are a helpful assistant with access to the internet.
1st check your internal knowledge for general questions.
2nd when you need real-time info: output exactly [SEARCH: query] on its own line.
Search results will be provided in the next message as numbered title+snippet pairs.
Extract ALL facts from all snippets — dates, numbers, names, capabilities, features.
For service/website descriptions: summarize what they offer.
Weather example: if snippet says 'hourly forecast with precipitation/wind/UV',
report that info is available and describe the content.
Never say 'I cannot find' when results are provided.
Only search when needed — otherwise use general knowledge."""

MAX_SEARCH_RESULTS: int = 5  # max results to show per search
OPEN_METEO_GEO: str = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_WX: str = "https://api.open-meteo.com/v1/forecast"

WEATHER_RE: re.Pattern[str] = re.compile(
    r"\b(weather|temperature|forecast|rain|snow|wind|humid|cloud|sunny|storm)\b",
    re.IGNORECASE,
)

VOICE_COMMAND: str = "/v"  # type the prompt instead of typing it
VOICE_SAMPLE_RATE: int = 16000  # sample rate of the WAV sent to the model
VOICE_BLOCK_MS: int = 100  # microphone read block size
VOICE_HINT: str = "Voice input needs sounddevice: pip install sounddevice"
# The model transcribes the recording itself, so it is only asked to write
# down what it hears, and thinking is off so the reply is short and quick.
TRANSCRIBE_PROMPT: str = "Transcribe the audio. Output the transcription only."

SPEAK_COMMAND: str = "/s"  # toggle spoken replies
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


def chat(messages: list[dict[str, str]], session_id: str) -> str:
    """Send a chat request to Ollama and return the model's response.

    Args:
        messages: list of message dicts with "role" ("system" | "user" | "assistant")
                  and "content" keys.
        session_id: unique session identifier for the conversation.

    Returns:
        The model's response content.

    Raises:
        URLError, KeyError, or json.JSONDecodeError if the request fails.
    """
    payload = json.dumps(
        {
            "model": MODEL,
            "messages": messages,
            "stream": False,
            "session_id": session_id,
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

    return data["choices"][0]["message"]["content"]


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
    if not line.strip() or line.strip() == "/q":
        return
    length = readline.get_current_history_length()
    if not length or readline.get_history_item(length) != line:
        readline.add_history(line)


def wait_for_enter() -> None:
    """Block until Enter is pressed, with terminal echo turned off.

    The terminal is switched to cbreak mode, so a single keypress is
    delivered without waiting for a newline while Ctrl+C keeps working.
    When terminal control is unavailable the call falls back to reading a
    whole line.

    Returns:
        None
    """
    if termios is None or tty is None:
        sys.stdin.readline()
        return
    fd = sys.stdin.fileno()
    try:
        saved = termios.tcgetattr(fd)
    except termios.error:  # not a terminal (e.g. piped input)
        sys.stdin.readline()
        return
    try:
        tty.setcbreak(fd)
        while True:
            char = sys.stdin.buffer.read(1)
            if char in (b"", b"\r", b"\n"):  # Enter, or end of input
                break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def record_voice() -> Any:
    """Record mono 16 kHz audio from the default microphone until Enter.

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

    def callback(indata: Any, frames: int, time_info: Any, status: Any) -> None:
        # A raw stream hands out a buffer over memory that PortAudio reuses
        # for the next block, so the samples have to be copied out.
        blocks.append(np.frombuffer(indata, dtype="float32").copy())

    block_size = int(VOICE_SAMPLE_RATE * VOICE_BLOCK_MS / 1000)
    try:
        with sd.RawInputStream(
            samplerate=VOICE_SAMPLE_RATE,
            blocksize=block_size,
            dtype="float32",
            channels=1,
            callback=callback,
        ):
            print("Recording... press Enter to stop, Ctrl+C to cancel", flush=True)
            wait_for_enter()
    except sd.PortAudioError as e:
        raise OSError(f"cannot open microphone: {e}") from e

    if not blocks:
        return np.empty(0, dtype="float32")
    return np.concatenate(blocks)


def encode_wav(audio: Any) -> bytes:
    """Wrap recorded samples into a WAV file held in memory.

    The microphone hands out float samples, while the model wants a WAV
    container, so the samples are scaled to signed 16-bit PCM and written
    into a 44-byte header.

    Args:
        audio: mono float32 samples at VOICE_SAMPLE_RATE, as returned by
               record_voice().

    Returns:
        The recording as a WAV file, ready to be sent to the model.

    Raises:
        ImportError: if numpy is not installed.
    """
    import numpy as np

    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(VOICE_SAMPLE_RATE)
        wav.writeframes(pcm.tobytes())
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
    sends the transcript as the prompt. Typing /s toggles spoken replies: the
    assistant's answer is then read out loud with Piper, in addition to being
    printed.

    Args:
        None

    Returns:
        None
    """
    session_id = uuid.uuid4().hex
    init_input_history()
    # Initialize conversation with system prompt and at least one dummy message
    # to ensure context is available on first user input
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    speak_replies = False
    print(f"AI agent — using model: {MODEL}")
    print(f"Type {VOICE_COMMAND} to speak your prompt.")
    print(f"Type {SPEAK_COMMAND} to hear the replies.")
    print("Press Enter with an empty prompt to exit.\n")

    try:
        while True:
            line = read_prompt()

            # Exit conditions: empty input or /q command
            if not line or line.strip() == "/q":
                break

            # Voice output toggle: speak every reply, or mute again
            if line.strip() == SPEAK_COMMAND:
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

            # Add user message to conversation history
            messages.append({"role": "user", "content": line})

            print("Thinking...", flush=True)

            # Get response from Ollama model with full history context
            try:
                reply = chat(messages, session_id)
            except urllib.error.URLError as e:
                print(f"Error: cannot reach Ollama ({e.reason})")
                break
            except (KeyError, json.JSONDecodeError) as e:
                print(f"Error: unexpected response from Ollama ({e})")
                break

            # Assistant's reply
            messages.append({"role": "assistant", "content": reply})

            # Check if the reply contains a [SEARCH: query] pattern
            match = SEARCH_RE.search(reply)
            if match:
                query = match.group(1)
                print(f"[search] {query}", flush=True)
                search_results = search_web(query)
                # Append both the original reply (with search trigger) and
                # the search results to history, then regenerate response
                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "user", "content": search_results})
                try:
                    reply = chat(messages, session_id)
                except urllib.error.URLError as e:
                    print(f"Error: cannot reach Ollama ({e.reason})")
                    break
                except (KeyError, json.JSONDecodeError) as e:
                    print(f"Error: unexpected response from Ollama ({e})")
                    break

            # Add final assistant reply to history and trim to HISTORY_LIMIT
            messages.append({"role": "assistant", "content": reply})

            # Debug: print full message history (uncomment to inspect)
            # print(messages)

            # Enforce history limit by dropping oldest messages after index 1
            if len(messages) > HISTORY_LIMIT:
                messages = [messages[0]] + messages[-(HISTORY_LIMIT - 1) :]

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
