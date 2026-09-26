#!/usr/bin/env python3
"""Interactive CLI AI agent with conversation history and web search via Ollama."""

import html
import json
import re
import urllib.error
import urllib.parse
import urllib.request
import uuid

try:  # readline is POSIX-only, may be missing on Windows or embedded builds
    import readline
except ImportError:
    readline = None  # type: ignore[assignment]

# Configuration: model, API endpoint, history limit, search service
MODEL: str = "carstenuhlig/omnicoder-2-9b:latest"
OLLAMA_URL: str = "http://localhost:11434/api/chat"
HISTORY_LIMIT: int = 32  # cap history to stay within model context
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

    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    return data["message"]["content"]


def init_input_history() -> None:
    """Enable readline line editing and cap the command history.

    With readline available, Up/Down arrows walk through previously submitted
    prompts. History is kept in memory for the current session only and is
    limited to HISTORY_LIMIT entries.
    """
    if readline is None:
        return
    readline.set_history_length(HISTORY_LIMIT)


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

    # Remember accepted commands, skipping blanks, the quit command and
    # immediate repeats (Up + Enter) so history stays free of duplicates.
    if readline is not None and line.strip() and line.strip() != "/q":
        length = readline.get_current_history_length()
        if not length or readline.get_history_item(length) != line:
            readline.add_history(line)
    return line


def main() -> None:
    """Main entry point for the interactive CLI agent.

    Runs an infinite loop prompting for user input, sending the full conversation
    history to Ollama, and displaying the assistant's response.

    The loop exits on:
      - empty prompt (just press Enter)
      - /q command
      - Ctrl+C

    Up/Down arrows recall previously entered commands when readline is available.

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
    print(f"AI agent — using model: {MODEL}")
    print("Press Enter with an empty prompt to exit.\n")

    try:
        while True:
            line = read_prompt()

            # Exit conditions: empty input or /q command
            if not line or line.strip() == "/q":
                break

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
    finally:
        print("Bye!")


if __name__ == "__main__":
    main()
