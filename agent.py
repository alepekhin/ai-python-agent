#!/usr/bin/env python3
import html
import json
import os
import re
import readline
import urllib.error
import urllib.parse
import urllib.request

MODEL: str = "carstenuhlig/omnicoder-2-9b:latest"
OLLAMA_URL: str = "http://localhost:11434/api/chat"
HISTORY_LIMIT: int = 32
HISTORY_FILE: str = os.path.expanduser("~/.agent_history")
DDG_URL: str = "https://html.duckduckgo.com/html/"
SEARCH_RE: re.Pattern[str] = re.compile(r"\[SEARCH:\s*(.+?)\]")
SYSTEM_PROMPT: str = (
    "You are a helpful assistant with access to the internet. "
    "When you need current or real-time information that you don't have, "
    "output exactly [SEARCH: your search query] on its own line. "
    "Search results will be provided to you in the next message. "
    "IMPORTANT: Each result has a title and a snippet with useful information. "
    "Extract ALL facts from the snippets — dates, numbers, names, descriptions, "
    "capabilities, features, or any concrete details mentioned. "
    "If snippets describe what services or websites offer, summarize that information. "
    "For example, if a snippet says 'hourly weather forecast with precipitation, "
    "wind, and UV index', report that this information is available and summarize "
    "what the results indicate. Never say 'I cannot find' if results are provided. "
    "Only search when truly necessary — for general knowledge you already "
    "know, answer directly without searching."
)
MAX_SEARCH_RESULTS: int = 5
OPEN_METEO_GEO: str = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_WX: str = "https://api.open-meteo.com/v1/forecast"
WEATHER_RE: re.Pattern[str] = re.compile(
    r"\b(weather|temperature|forecast|rain|snow|wind|humid|cloud|sunny|storm)\b",
    re.IGNORECASE,
)


def _get_weather(location: str) -> str:
    try:
        geo_url = OPEN_METEO_GEO + "?" + urllib.parse.urlencode({"name": location, "count": 1})
        with urllib.request.urlopen(geo_url, timeout=10) as resp:
            geo = json.loads(resp.read().decode("utf-8"))
        results = geo.get("results", [])
        if not results:
            return ""
        lat, lon = results[0]["latitude"], results[0]["longitude"]
        name = results[0].get("name", location)

        wx_url = OPEN_METEO_WX + "?" + urllib.parse.urlencode({
            "latitude": lat,
            "longitude": lon,
            "current_weather": "true",
        })
        with urllib.request.urlopen(wx_url, timeout=10) as resp:
            wx = json.loads(resp.read().decode("utf-8"))

        cw = wx.get("current_weather", {})
        temp = cw.get("temperature")
        wind = cw.get("windspeed")
        desc = cw.get("weathercode")
        WMO = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Fog", 48: "Rime fog", 51: "Light drizzle", 53: "Moderate drizzle",
            55: "Dense drizzle", 61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
            71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
            80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
            95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Severe thunderstorm with hail",
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
            "", location, flags=re.IGNORECASE,
        )
        location = re.sub(r"[,]", " ", location)
        location = re.sub(r"\s+", " ", location).strip(" ,.-")
        if location:
            wx = _get_weather(location)
            if wx:
                output = wx + "\n\n" + output

    return output


def chat(messages: list[dict[str, str]]) -> str:
    payload = json.dumps(
        {
            "model": MODEL,
            "messages": messages,
            "stream": False,
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


def main() -> None:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    if os.path.exists(HISTORY_FILE):
        readline.read_history_file(HISTORY_FILE)
    readline.set_history_length(1000)

    print(f"AI agent — using model: {MODEL}")
    print("Press Enter with an empty prompt to exit.\n")

    try:
        while True:
            try:
                line = input("You: ")
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not line or line.strip() == "/q":
                break

            if line.strip():
                readline.add_history(line)

            messages.append({"role": "user", "content": line})

            print("Thinking...", flush=True)

            try:
                reply = chat(messages)
            except urllib.error.URLError as e:
                print(f"Error: cannot reach Ollama ({e.reason})")
                break
            except (KeyError, json.JSONDecodeError) as e:
                print(f"Error: unexpected response from Ollama ({e})")
                break

            match = SEARCH_RE.search(reply)
            if match:
                query = match.group(1)
                ddg_url = DDG_URL + "?" + urllib.parse.urlencode({"q": query})
                print(f"[search] {query}", flush=True)
                print(f"[url]    {ddg_url}", flush=True)
                search_results = search_web(query)
                print(f"[results] {search_results}", flush=True)
                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "user", "content": search_results})
                try:
                    reply = chat(messages)
                except urllib.error.URLError as e:
                    print(f"Error: cannot reach Ollama ({e.reason})")
                    break
                except (KeyError, json.JSONDecodeError) as e:
                    print(f"Error: unexpected response from Ollama ({e})")
                    break

            messages.append({"role": "assistant", "content": reply})

            if len(messages) > HISTORY_LIMIT:
                messages = [messages[0]] + messages[-(HISTORY_LIMIT - 1) :]

            print(f"Assistant: {reply}\n")
    finally:
        readline.write_history_file(HISTORY_FILE)

    print("Bye!")


if __name__ == "__main__":
    main()
