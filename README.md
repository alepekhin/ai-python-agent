# ai-python-agent

A simple CLI AI agent with conversation history, built on Python and Ollama. It keeps the whole chat history in context so the model can answer follow-up questions, while trimming the history to stay within the context length. Prompts can be typed or dictated.

## Requirements

- Python 3
- [Ollama](https://ollama.com) running locally
- Model `carstenuhlig/omnicoder-2-9b:latest` pulled (`ollama pull carstenuhlig/omnicoder-2-9b`)

Optional, for voice input only (the system Python is externally managed on Debian/Ubuntu, so use a venv):

```sh
python3 -m venv .venv
.venv/bin/python -m pip install faster-whisper sounddevice
sudo apt install libportaudio2   # PortAudio backend for sounddevice (macOS: brew install portaudio)
```

Run the agent with the venv interpreter (`.venv/bin/python agent.py`); `run_debug.sh` picks it up automatically. Without these packages the agent still works for typed prompts and `/v` just prints the install hint.

## Usage

```sh
.venv/bin/python agent.py
```

Type your prompt and press Enter. The agent sends the full conversation history to the model, so it remembers earlier turns. End the dialog by pressing Enter on an empty prompt (or typing `/q`, or pressing Ctrl+C).

Press Up/Down to walk through the commands you already entered and repeat one (standard readline editing, so Ctrl+A/E/K and other shortcuts work too). The command history is saved to `~/.local/share/ai-python-agent/history` when the agent exits, so prompts from previous sessions are also available with the Up arrow; the last `HISTORY_LIMIT` (32) entries are kept. Set `HISTORY_FILE` in `agent.py` to store it elsewhere.

Example session:

```
AI agent — using model: carstenuhlig/omnicoder-2-9b:latest
Type /v to speak your prompt.
Press Enter with an empty prompt to exit.

You: hi
Thinking...
Assistant: Hello! How can I help you?
You:
Bye!
```

## Voice input

Type `/v` instead of a prompt to dictate it: recording starts immediately, you press Enter again to stop, and the transcript is sent to the model as your prompt (echoed as `You (voice): ...`, and kept in the input history so Up recalls it). Ctrl+C cancels the recording. Silence is trimmed with faster-whisper's VAD filter, and a transcript that comes back empty is ignored, so the dialog just shows the prompt again.

The Whisper model is downloaded on the first `/v` and cached on disk, so later sessions need no internet. It runs on the CPU with `int8` quantization, which fits comfortably in 16 GB of RAM. The model, device, quantization, language and audio settings are constants at the top of `agent.py`:

| Constant | Default | Meaning |
| --- | --- | --- |
| `VOICE_COMMAND` | `/v` | command that starts voice input |
| `WHISPER_MODEL` | `base` | `tiny`, `base`, `small`, `medium`, `large-v3` |
| `WHISPER_DEVICE` | `cpu` | set to `cuda` to use a GPU (with `WHISPER_COMPUTE_TYPE = "float16"`) |
| `WHISPER_COMPUTE_TYPE` | `int8` | Whisper quantization |
| `WHISPER_LANGUAGE` | `None` | `None` detects the language, e.g. `"en"` to force English |

Without `faster-whisper` and `sounddevice` installed the agent works as before; `/v` just prints the install hint.

## How it works

- The full message history is sent to Ollama's `/api/chat` endpoint on each turn, so the model has the previous context.
- The history is capped at `HISTORY_LIMIT` (32) messages (`agent.py:9`), dropping the oldest turns so the prompt stays within the model's context length.
- The model URL and name are configurable via the `OLLAMA_URL` and `MODEL` constants at the top of `agent.py`.
- Prompts are persisted to `HISTORY_FILE` (`~/.local/share/ai-python-agent/history`): loaded on start, saved on exit, capped at `HISTORY_LIMIT` entries.
- Voice prompts are recorded with `sounddevice` (16 kHz mono, Enter stops the recording) and transcribed with a cached faster-whisper model; the transcript is just another user message in the same history.

## Tests

The interactive loop is exercised by a test that emulates a TTY (via `pty`) against a fake Ollama server, so no real model is needed. Voice input is covered too, with fake `sounddevice`/`faster_whisper` modules:

```sh
python3 test_agent.py
```

## How to debug 

- start app in debug mode executing `run_debug.sh `
- open agent.py  
- open dap_ui 
- set breakpoint 
- start new dap session 

## License

MIT
