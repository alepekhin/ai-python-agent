# Simple AI agent with conversation history 

## Goal 

Use LLM conversation history in next prompt, typed or dictated 
Print and say response 

## Requirements 

- Use Python3, Ollama, model ollama/gemma4:latest, reached over `/v1/chat/completions` (the only endpoint that takes an audio prompt)  
- CLI 
- End dialog with empty prompt (also `/q` or Ctrl+C) 
- History length should be less than context length (trimmed to `HISTORY_LIMIT` = 32 messages) 
- Show "Thinking..." while the model responds 
- Tools the model can ask for by writing a tag in its reply: `[SEARCH: query]` (web search) and `[READ: path]` (local file), limited to `READ_ROOTS` (the working directory by default)
- Voice input on `/v`: a 2-second pause (or Enter) ends the recording, which is sent to the model as a WAV and transcribed by the model itself
- Voice output with local Piper TTS, on by default and muted/unmuted by `/s` (text is always printed too), voice picked per reply: Russian for Cyrillic replies, English otherwise; Ctrl+C while a reply is being spoken stops the audio and returns to the prompt
- `/e` sends the prompt "translate next prompts from English to Russian", `/r` the same from Russian to English, so prompts to follow are answered in the other language
- Works locally without internet
- Works on computer with 16 Gb memory

## Prerequisite 

- Python3, Ollama and model are installed
- Optional for voice: `sounddevice`, `piper-tts` installed
