# ai-python-agent

A simple CLI AI agent with conversation history, built on Python and Ollama. It keeps the whole chat history in context so the model can answer follow-up questions, while trimming the history to stay within the context length. Prompts can be typed or dictated, and the replies can be read out loud by a local neural voice.

## Requirements

- Python 3
- [Ollama](https://ollama.com) running locally
- Model `gemma4:latest` pulled (`ollama pull gemma4`)

Optional, for voice input and voice output (the system Python is externally managed on Debian/Ubuntu, so use a venv):

```sh
python3 -m venv .venv
.venv/bin/python -m pip install sounddevice piper-tts
sudo apt install libportaudio2   # PortAudio backend for sounddevice (macOS: brew install portaudio)
```

Run the agent with the venv interpreter (`.venv/bin/python agent.py`); `run_debug.sh` picks it up automatically. Without these packages the agent still works for typed prompts: `/v` prints the voice input install hint, and `/s` prints the voice output one.

## Usage

```sh
.venv/bin/python agent.py
```

Type your prompt and press Enter. The agent sends the full conversation history to the model, so it remembers earlier turns. End the dialog by pressing Enter on an empty prompt (or typing `/q`, or pressing Ctrl+C).

Press Up/Down to walk through the commands you already entered and repeat one (standard readline editing, so Ctrl+A/E/K and other shortcuts work too). The command history is saved to `~/.local/share/ai-python-agent/history` when the agent exits, so prompts from previous sessions are also available with the Up arrow; the last `HISTORY_LIMIT` (32) entries are kept. Set `HISTORY_FILE` in `agent.py` to store it elsewhere.

Example session:

```
AI agent — using model: gemma4:latest
Type /v to speak your prompt.
Type /s to hear the replies.
Press Enter with an empty prompt to exit.

You: hi
Thinking...

Assistant: Hello! How can I help you?

You:
Bye!
```

## Voice input

Type `/v` instead of a prompt to dictate it: recording starts immediately, you press Enter again to stop, and the transcript is sent to the model as your prompt (echoed as `You (voice): ...`, and kept in the input history so Up recalls it). Ctrl+C cancels the recording. A transcript that comes back empty is ignored, so the dialog just shows the prompt again.

There is no local speech recognition: the recording is packed into a mono 16-bit 16 kHz WAV and handed to the model itself, which transcribes it. The audio travels as an `input_audio` block to Ollama's OpenAI-compatible `/v1/chat/completions` endpoint — the native `/api/chat` endpoint accepts `images` but silently drops audio (see [ollama#17730](https://github.com/ollama/ollama/issues/17730)), which is why that endpoint is used for the whole conversation, not just for the audio turn. The transcription request asks for the transcript only, with thinking disabled (`reasoning_effort: "none"`) and a fixed temperature, so it stays short and fast. The transcript then becomes a plain text user message, so no audio is kept in the history.

Audio-related constants at the top of `agent.py`:

| Constant | Default | Meaning |
| --- | --- | --- |
| `VOICE_COMMAND` | `/v` | command that starts voice input |
| `TRANSCRIBE_PROMPT` | `Transcribe the audio. Output the transcription only.` | instruction sent along with the recording |
| `VOICE_SAMPLE_RATE` | `16000` | sample rate of the recorded WAV |
| `VOICE_BLOCK_MS` | `100` | microphone read block size |

Without `sounddevice` installed the agent works for typed prompts; `/v` just prints the install hint. A microphone that cannot be opened, or a model that cannot transcribe the recording, is reported and the dialog continues.

## Voice output

Type `/s` to toggle spoken replies: from then on every answer is read out loud, in addition to being printed. Type `/s` again to mute. The toggle lasts for the session, and `/q` or an empty prompt still ends the dialog.

The voice is [Piper](https://github.com/OHF-Voice/piper1-gpl) — a small ONNX neural TTS that runs on the CPU, so it stays well inside 16 GB of RAM and synthesizes roughly 15–30x faster than real time (playback itself takes as long as the reply is long). The language of every reply is detected from its script, and the matching voice is used: a mostly Cyrillic reply is read by `ru_RU-denis-medium`, anything else by `en_US-lessac-medium`. A reply that mixes both is named after the script it mostly uses, so a Russian answer with a few English words still gets the Russian voice.

Each voice (~60 MB) is downloaded the first time it is needed into `TTS_DIR` and cached on disk, so later sessions need no internet and only the languages you actually chat in are downloaded. The default voice is loaded while the toggle is confirmed, so the first answer is spoken without a delay; switching to another language for the first time prints a short download notice.

Replies are cleaned up before speaking: code blocks are dropped, raw URLs are replaced by a word in the spoken language ("link" / "ссылка"), inline markup like `**bold**` and `[text](url)` is reduced to its text, and only the first `TTS_MAX_CHARS` (1200) characters are read out, so a long answer is not a minute and a half of talking. Press Ctrl+C to stop the voice and end the dialog.

Related constants at the top of `agent.py`:

| Constant | Default | Meaning |
| --- | --- | --- |
| `SPEAK_COMMAND` | `/s` | command that toggles voice output |
| `PIPER_VOICE_EN` | `en_US-lessac-medium` | voice for replies that are not mostly Cyrillic |
| `PIPER_VOICE_RU` | `ru_RU-denis-medium` | voice for mostly Cyrillic replies (`irina`, `dmitri` and `ruslan` also exist) |
| `PIPER_USE_CUDA` | `False` | set to `True` to run the voice on an NVIDIA GPU (needs `onnxruntime-gpu`) |
| `TTS_DIR` | `~/.local/share/ai-python-agent/voices` | where the downloaded voices are cached |
| `TTS_MAX_CHARS` | `1200` | how much of a long reply is spoken |

Run `python -m piper.download_voices` for the full list of voices; any language can be added by pointing `TTS_VOICES` at a new name and a matching entry in `TTS_LINK_WORDS`.

Without `piper-tts` and `sounddevice` installed the agent works as before; `/s` just prints the install hint. If synthesis or playback fails mid-session, the error is printed and the voice is muted, while the dialog continues.

## How it works

- The full message history is sent to Ollama's OpenAI-compatible `/v1/chat/completions` endpoint on each turn, so the model has the previous context and, unlike `/api/chat`, an audio prompt.
- The history is capped at `HISTORY_LIMIT` (32) messages (`agent.py:12`), dropping the oldest turns so the prompt stays within the model's context length.
- The model URL and name are configurable via the `OLLAMA_URL` and `MODEL` constants at the top of `agent.py`.
- Prompts are persisted to `HISTORY_FILE` (`~/.local/share/ai-python-agent/history`): loaded on start, saved on exit, capped at `HISTORY_LIMIT` entries.
- Voice prompts are recorded with `sounddevice` (16 kHz mono, Enter stops the recording) and the WAV is sent to the model, which returns the transcript; that transcript is just another text user message in the same history.
- Replies are spoken with a cached Piper voice, chosen per reply from the language detected in its text, and played through the default output device with `sounddevice`; only the speakable part of the reply is rendered.

## Tests

The interactive loop is exercised by a test that emulates a TTY (via `pty`) against a fake Ollama server, so no real model is needed. Voice input and voice output are covered too, with a fake `sounddevice`/`piper` and a fake server that answers both the transcription and the chat request:

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
