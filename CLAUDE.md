# periplus-cli

Terminal chat harness for a local LLM served by llama-server (llama.cpp).

## Stack

- Python 3.14, uv for package management
- `openai` SDK pointed at `http://127.0.0.1:8080/v1` (llama-server's OpenAI-compatible API)
- Model: Qwen3 — thinking mode disabled via `chat_template_kwargs: {enable_thinking: false}`

## Run

```bash
uv run main.py
```

## Key files

| File | Purpose |
|------|---------|
| `main.py` | Entry point — chat loop, tool dispatch, streaming |
| `SYSTEM_PROMPT.md` | System prompt loaded fresh each run |
| `MAP.md` | Agent memory (appended to system prompt at startup; created on first `remember` call) |

## Architecture

- `build_system_prompt()` — reads `SYSTEM_PROMPT.md`, appends `MAP.md` contents if it exists
- `complete(history)` — one streaming turn; accumulates tool call chunks, returns `(reply, tool_calls)`
- `chat()` — outer loop; inner `while True` handles tool → response cycles
- `TOOL_HANDLERS` dict maps tool names to callables; add new tools here + to `TOOLS` list

## Adding a tool

1. Add handler to `TOOL_HANDLERS`
2. Add OpenAI function schema to `TOOLS`
3. Document it in `SYSTEM_PROMPT.md` if the agent needs to know when to use it

## Constants (top of main.py)

| Constant | Default |
|----------|---------|
| `MEMORY_FILE` | `MAP.md` |
| `SYSTEM_PROMPT_FILE` | `SYSTEM_PROMPT.md` |
| `MODEL` | `qwen3` |
| `BASE_URL` | `http://127.0.0.1:8080/v1` |
