# Simple AI agent with conversation history 

## Goal 

Use LLM conversation history in next prompt, typed or dictated 
Print and say response 

## Requirements 

- Use Python3, Ollama, model `gemma4:latest`, reached over `/v1/chat/completions` (the only endpoint that takes an audio prompt)  
- CLI 
- Web server on `--serve`: a prompt in an HTTP request, the reply in the response, conversation history kept per session id; `POST /transcribe` turns a WAV recording into text, so a phone can dictate a prompt without speech recognition on the device  
- Android client in `android/`: records a prompt, posts it to `/chat` with its session id, reads the reply out loud  
- End dialog with empty prompt (also `/q` or Ctrl+C) 
- History length should be less than context length (trimmed to `HISTORY_LIMIT` = 32 messages) 
- Show "Thinking..." while the model responds 
- Tools the model can ask for by writing a tag in its reply: `[DATE]` (current date and time, another timezone with `[DATE: Europe/Moscow]`), `[WEATHER]` (forecast for Химози from https://ru-meteo.ru/khimozi), `[SEARCH: query]` (web search), `[READ: path]` (local file) and `[IMAGE: path]` (attach a picture for the model to look at); reads are limited to `READ_ROOTS` (the working directory by default) 
- Voice input on `/v`: a 2-second pause (or Enter) ends the recording, which is sent to the model as a WAV and transcribed by the model itself 
- Voice output with local Piper TTS, on by default and muted/unmuted by `/s` (text is always printed too), voice picked per reply: Russian for Cyrillic replies, English otherwise; Ctrl+C while a reply is being spoken stops the audio and returns to the prompt 
- Answers are in Russian by default; `/e` sends the prompt "translate next prompts from English to Russian", `/r` the same from Russian to English, so prompts to follow are answered in the other language — the language is one line of the system prompt (`ANSWER_IN`), rewritten by the command, so the two instructions never disagree 
- Works locally without internet (the tools that need it report the failure and the dialog goes on) 
- Works on computer with 16 Gb memory 

## Prerequisite 

- Python3, Ollama and model are installed
- Optional for voice: `sounddevice`, `piper-tts` installed
