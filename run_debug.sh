#/nin/sh
# Prefer the project venv, it has faster-whisper/sounddevice for voice input.
PY=python3
[ -x "$(dirname "$0")/.venv/bin/python" ] && PY="$(dirname "$0")/.venv/bin/python"
$PY -Xfrozen_modules=off -m debugpy --listen 127.0.0.1:5678 agent.py
