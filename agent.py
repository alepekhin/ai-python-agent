#!/usr/bin/env python3
import json
import urllib.error
import urllib.request

MODEL: str = "carstenuhlig/omnicoder-2-9b:latest"
OLLAMA_URL: str = "http://localhost:11434/api/chat"
HISTORY_LIMIT: int = 32


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
    messages: list[dict[str, str]] = []

    print("AI agent — press Enter with an empty prompt to exit.\n")

    while True:
        try:
            line = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line or line.strip() == "/q":
            break

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

        messages.append({"role": "assistant", "content": reply})

        if len(messages) > HISTORY_LIMIT:
            messages = messages[-HISTORY_LIMIT:]

        print(f"Assistant: {reply}\n")

    print("Bye!")


if __name__ == "__main__":
    main()
