# ai-python-agent

A simple CLI AI agent with conversation history, built on Python and Ollama. It keeps the whole chat history in context so the model can answer follow-up questions, while trimming the history to stay within the context length.

## Requirements

- Python 3
- [Ollama](https://ollama.com) running locally
- Model `carstenuhlig/omnicoder-2-9b:latest` pulled (`ollama pull carstenuhlig/omnicoder-2-9b`)

## Usage

```sh
python3 agent.py
```

Type your prompt and press Enter. The agent sends the full conversation history to the model, so it remembers earlier turns. End the dialog by pressing Enter on an empty prompt (or typing `/q`, or pressing Ctrl+C).

Example session:

```
AI agent — press Enter with an empty prompt to exit.

You: hi
Thinking...
Assistant: Hello! How can I help you?
You:
Bye!
```

## How it works

- The full message history is sent to Ollama's `/api/chat` endpoint on each turn, so the model has the previous context.
- The history is capped at `HISTORY_LIMIT` (32) messages (`agent.py:9`), dropping the oldest turns so the prompt stays within the model's context length.
- The model URL and name are configurable via the `OLLAMA_URL` and `MODEL` constants at the top of `agent.py`.

## Tests

The interactive loop is exercised by a test that emulates a TTY (via `pty`) against a fake Ollama server, so no real model is needed:

```sh
python3 test_agent.py
```

## License

MIT