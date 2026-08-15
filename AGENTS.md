# Simple AI agent with conversation history 

## Goal 

Use LLM conversation history in next prompt 

## Requirements 

- Use Python3, Ollama, model carstenuhlig/omnicoder-2-9b:latest  
- CLI 
- End dialog with empty prompt (also `/q` or Ctrl+C) 
- History length should be less than context length (trimmed to `HISTORY_LIMIT` = 32 messages) 
- Show "Thinking..." while the model responds 

## Prerequisite 

- Python3, Ollama and model are installed