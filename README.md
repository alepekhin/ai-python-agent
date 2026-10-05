# ai-python-agent

A simple CLI AI agent with conversation history, built on Python and Ollama. It keeps the whole chat history in context so the model can answer follow-up questions, while trimming the history to stay within the context length. The model can search the web and read local files, prompts can be typed or dictated, and the replies can be read out loud by a local neural voice.

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

Type `/e` when you start writing prompts in English: the agent sends `translate next prompts from English to Russian` (`TRANSLATE_PROMPT`) instead of the command itself, so the model translates the prompts to follow and keeps answering in Russian. Type `/r` for the other way round (`TRANSLATE_PROMPT_RU`, `translate next prompts from Russian to English`): Russian prompts are answered in English.

Press Up/Down to walk through the commands you already entered and repeat one (standard readline editing, so Ctrl+A/E/K and other shortcuts work too). The command history is saved to `~/.local/share/ai-python-agent/history` when the agent exits, so prompts from previous sessions are also available with the Up arrow; the last `HISTORY_LIMIT` (32) entries are kept. Set `HISTORY_FILE` in `agent.py` to store it elsewhere.

Every request sent to the model is also written to `~/.local/share/ai-python-agent/last_prompt.json` (constant `LAST_PROMPT_FILE`), replaced on each turn: it holds the system prompt, the whole history and the last user message, indented, so you can see exactly what the model was asked. The file is overwritten even when a turn runs a tool or transcribes a recording, so it always shows the request behind the answer you are looking at.

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

## Tools

The model can use three tools by writing a tag into its reply. The agent runs the tool, feeds its result back to the model as the next message and asks again, so the tag never reaches you — you only see the final answer. A reply is a tool call at most `MAX_TOOL_ROUNDS` (4) times per prompt, so a model that keeps asking for the same file cannot loop forever.

| Tag | What it does | What the model gets back |
| --- | --- | --- |
| `[SEARCH: query]` | searches DuckDuckGo, and adds the current weather when the query mentions it | up to 5 results as numbered title + snippet pairs |
| `[READ: path]` | reads a local file | the contents prefixed with line numbers, so it can cite them; a binary file is not read, only its path and size |
| `[IMAGE: path]` | attaches an image file | the picture itself, as a base64 `image_url` part, which the model can look at |

Ask in plain language — "what is in README.md?", "compare agent.py and test_agent.py" — and the model decides whether it needs a tool. You can see what it did while it thinks: the agent prints `[read] agent.py` or `[search] ...` for every call.

A tag counts only when it is alone on its line, which is how the system prompt asks for it: an answer that merely mentions the syntax (a summary of this very file, for instance) does not run anything.

Reads are confined to `READ_ROOTS`, which defaults to the directory the agent was started in: a relative path is taken from there, `~` is expanded, symlinks are followed, and the resolved path must still be inside one of the roots, so `../` cannot reach the rest of the filesystem. A missing file, a directory or an unreadable file is reported back to the model as a short sentence, and a file bigger than `READ_MAX_BYTES` is sent truncated with a note, so a huge file cannot blow up the context. A binary file is recognized by the first `READ_PROBE_BYTES` (8192) bytes and is never read past that: the model gets the path and the size, so it can tell you where the file is and what it is, but no bytes of it reach the context. For the same reason a prompt naming a binary file carries the path with it: typing "compare blood.jpg and notes.txt" sends the prompt plus `Files named in this prompt that are binary, given by path and size only: blood.jpg (342822 bytes). An image among them can be attached with [IMAGE: path] to be seen.`, so the model knows what you meant and can ask to see the picture. Names that are not files, and text files, are left out and the prompt is sent as typed. The same path check applies to a dictated prompt.

### Images

Gemma 4 is a multimodal model, so the agent can hand it a picture: the model writes `[IMAGE: blood.jpg]` and the next request carries the file as a base64 `image_url` part, which the model then describes — ask "что показывает blood.jpg?" and the answer comes from the pixels. Ask in plain language, as with the other tools.

Only the formats in `IMAGE_MIME_TYPES` are attached (`.jpg`, `.jpeg`, `.png`, `.gif`, `.webp`, `.bmp`); anything else, a file outside `READ_ROOTS` or one bigger than `IMAGE_MAX_BYTES` (5 MB) is reported to the model, which can tell you why it cannot see it. A file that is not an image format is not converted.

An image costs hundreds of kilobytes of base64 in every later request, so only the newest `IMAGE_HISTORY_KEEP` (1) images keep their bytes: older ones stay in the history as a one-line note saying the picture is no longer attached, and the model can ask for the file again. Setting it to 0 attaches an image for one turn only. The saved `last_prompt.json` never holds image or audio data — those parts are written as a placeholder, so the file stays readable.

Constants at the top of `agent.py`:

| Constant | Default | Meaning |
| --- | --- | --- |
| `READ_ROOTS` | `None` (the working directory) | directories the model may read; a list of paths widens it, `[]` opens it to the whole filesystem |
| `READ_MAX_BYTES` | `200000` | how much of a bigger file is sent before it is cut off |
| `READ_PROBE_BYTES` | `8192` | how much of a file is looked at to tell text from binary |
| `IMAGE_MIME_TYPES` | jpg, jpeg, png, gif, webp, bmp | the image formats the model is given; anything else is refused |
| `IMAGE_MAX_BYTES` | `5000000` | an image bigger than this is not attached |
| `IMAGE_HISTORY_KEEP` | `1` | how many images stay in the history with their bytes, the older ones are kept as a note |
| `MAX_TOOL_ROUNDS` | `4` | how many tool results are fed back for one prompt |
| `ANSWER_FROM_RESULT` | `Answer from this result, do not write a tool tag.` | reminder appended to a tool result, since a small model tends to ask for another tool instead of answering |

## Voice input

Type `/v` instead of a prompt to dictate it: recording starts immediately and ends on a pause of 2 seconds, so you just start talking and stop; Enter also stops it right away. The transcript is sent to the model as your prompt (echoed as `You (voice): ...`, and kept in the input history so Up recalls it). Ctrl+C cancels the recording. A transcript that comes back empty is ignored, so the dialog just shows the prompt again.

A block of audio counts as speech only when it is `VOICE_NOISE_FACTOR` (2.5) times louder than the room itself, where the room level is the quiet of the last `VOICE_NOISE_BLOCKS` blocks. A fan, a hiss or a microphone gain turned up therefore cannot drown out the pause, and a quiet microphone is not deafened by the threshold either; the bar jumps up at once when the room gets louder and comes down slowly, so a soft syllable does not lower it. Two seconds of such blocks in a row end the prompt — after at least one loud block, and the pause is counted only from that block on, so a quiet room never cuts a recording short and the silence before the first word is not part of it. The closing silence is dropped, so the model gets the words and not the pause. The first `VOICE_NOISE_BLOCKS` blocks only go to measuring the room, since a word cannot be told from the room before the room is known: without that a microphone that comes up with a bang in its first block passes for a word and sets the bar so high that no word after it is ever heard, leaving a recording of 0.1 s of the bang. A recording with no word in it is not sent to the model at all — the levels involved are printed instead (`0.0180 is the noise, 0.0450 is what counts as speech`), which is what to tune if a room is too loud to be heard past. Whatever is typed while the recording runs belongs to the recording, not to the prompt that comes after it.

There is no local speech recognition: the recording is packed into a mono 16-bit 16 kHz WAV and handed to the model itself, which transcribes it. A quiet recording is turned up to `VOICE_PEAK` first, at most `VOICE_MAX_GAIN` times over: speech spoken softly from across the desk reaches the model's audio tower as a whisper, and what comes back then is a guess (a stray word in another language) rather than what was said. The audio travels as an `input_audio` block to Ollama's OpenAI-compatible `/v1/chat/completions` endpoint — the native `/api/chat` endpoint accepts `images` but silently drops audio (see [ollama#17730](https://github.com/ollama/ollama/issues/17730)), which is why that endpoint is used for the whole conversation, not just for the audio turn. The transcription request asks for the transcript only, with thinking disabled (`reasoning_effort: "none"`) and a fixed temperature, so it stays short and fast. The transcript then becomes a plain text user message, so no audio is kept in the history.

Audio-related constants at the top of `agent.py`:

| Constant | Default | Meaning |
| --- | --- | --- |
| `VOICE_COMMAND` | `/v` | command that starts voice input |
| `TRANSCRIBE_PROMPT` | `Transcribe the audio. Output the transcription only.` | instruction sent along with the recording |
| `VOICE_SAMPLE_RATE` | `16000` | sample rate of the recorded WAV |
| `VOICE_BLOCK_MS` | `100` | microphone read block size |
| `VOICE_SILENCE_MS` | `2000` | pause that ends the prompt without Enter |
| `VOICE_NOISE_BLOCKS` | `10` | blocks the room level is measured over (1 s) |
| `VOICE_NOISE_FACTOR` | `2.5` | how much louder than the room a block must be to be speech |
| `VOICE_SPEECH_RMS` | `0.003` | level that is speech even where the room is dead silent |
| `VOICE_NOISE_FALL` | `0.1` | how fast the speech bar may drop when the room gets quieter |
| `VOICE_PEAK` | `0.95` | level a quiet recording is turned up to before it is sent |
| `VOICE_MAX_GAIN` | `100.0` | most a recording is amplified, so noise is not blown up |

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
- A command is never sent to the model as typed: `/q` ends the dialog, `/s` toggles the voice, `/v` is replaced by the transcript of a recording, and `/e` / `/r` are replaced by `TRANSLATE_PROMPT` / `TRANSLATE_PROMPT_RU`, which tell the model which language the prompts to follow are written in and that they are to be answered in the other one.
- Every request is also written to `LAST_PROMPT_FILE` (`~/.local/share/ai-python-agent/last_prompt.json`), overwritten on each turn, so the last prompt the model was given — system prompt, whole history and the last user message, as indented JSON — can be read afterwards to see why it answered the way it did. A file that cannot be written is ignored.
- Voice prompts are recorded with `sounddevice` (16 kHz mono, ended by Enter or by a 2-second pause) and the WAV is sent to the model, which returns the transcript; that transcript is just another text user message in the same history.
- A reply carrying a `[SEARCH: ...]`, `[READ: ...]` or `[IMAGE: ...]` tag is answered by running that tool (`agent_turn()`), appending its result to the history and asking the model again; the reply without a tag is the one that is printed and spoken. The `[IMAGE: ...]` result is not text but content parts — a line naming the file and the picture as an `image_url` — so a message can be a list of parts, and only the newest few images keep their bytes.
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
