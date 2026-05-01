import itertools
import json
import threading
import time
from pathlib import Path
from openai import OpenAI

MEMORY_FILE = "MAP.md"
SYSTEM_PROMPT_FILE = "SYSTEM_PROMPT.md"
MODEL = "qwen3"
BASE_URL = "http://127.0.0.1:8080/v1"

client = OpenAI(base_url=BASE_URL, api_key="none")

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "Save important information to persistent memory for future conversations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The information to save."}
                },
                "required": ["content"],
            },
        },
    }
]


def _remember(content: str) -> str:
    memory = Path(MEMORY_FILE)
    if not memory.exists():
        memory.write_text("# Memory\n\n", encoding="utf-8")
    with memory.open("a", encoding="utf-8") as f:
        f.write(content.rstrip() + "\n")
    preview = content[:80] + ("..." if len(content) > 80 else "")
    return f"Saved: {preview}"


TOOL_HANDLERS = {
    "remember": lambda args: _remember(args["content"]),
}


def execute_tool(name: str, arguments: str) -> str:
    handler = TOOL_HANDLERS.get(name)
    if not handler:
        return f"Unknown tool: {name}"
    try:
        return handler(json.loads(arguments))
    except Exception as e:
        return f"Tool error: {e}"


def run_tool_with_spinner(name: str, arguments: str) -> str:
    frames = itertools.cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])
    stop = threading.Event()

    def spin():
        while not stop.is_set():
            print(f"\r\x1b[2m{next(frames)} {name}...\x1b[0m", end="", flush=True)
            time.sleep(0.08)

    t = threading.Thread(target=spin, daemon=True)
    t.start()
    try:
        result = execute_tool(name, arguments)
    finally:
        stop.set()
        t.join()
        print(f"\r\x1b[2K", end="")  # clear spinner line
    return result


# ---------------------------------------------------------------------------
# System prompt + memory loading
# ---------------------------------------------------------------------------

def build_system_prompt() -> str:
    prompt = Path(SYSTEM_PROMPT_FILE).read_text(encoding="utf-8").strip()
    memory = Path(MEMORY_FILE)
    if memory.exists():
        content = memory.read_text(encoding="utf-8").strip()
        if content:
            prompt += f"\n\n## Saved memories ({MEMORY_FILE})\n\n{content}"
    return prompt


# ---------------------------------------------------------------------------
# Chat loop
# ---------------------------------------------------------------------------

def complete(history: list) -> tuple[str, list]:
    """Stream one completion turn. Returns (reply_text, tool_calls_list)."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=history,
        tools=TOOLS,
        stream=True,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    reply = ""
    tool_calls_acc: dict[int, dict] = {}

    print("Assistant: ", end="", flush=True)
    for chunk in response:
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True)
            reply += delta.content
        if delta.tool_calls:
            for tc in delta.tool_calls:
                entry = tool_calls_acc.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                if tc.id:
                    entry["id"] = tc.id
                if tc.function.name:
                    entry["name"] = tc.function.name
                if tc.function.arguments:
                    entry["arguments"] += tc.function.arguments
    print()

    tool_calls = [
        {"id": v["id"], "type": "function", "function": {"name": v["name"], "arguments": v["arguments"]}}
        for v in (tool_calls_acc[k] for k in sorted(tool_calls_acc))
    ]
    return reply, tool_calls


def chat():
    history = [{"role": "system", "content": build_system_prompt()}]
    print(f"Chat with Qwen3 — memories stored in {MEMORY_FILE}  (type 'exit' or Ctrl+C to quit)\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break

        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Bye.")
            break

        history.append({"role": "user", "content": user_input})

        while True:
            reply, tool_calls = complete(history)

            if not tool_calls:
                history.append({"role": "assistant", "content": reply})
                break

            history.append({"role": "assistant", "content": reply or None, "tool_calls": tool_calls})

            for tc in tool_calls:
                result = run_tool_with_spinner(tc["function"]["name"], tc["function"]["arguments"])
                print(f"\x1b[2m[{tc['function']['name']}] {result}\x1b[0m")
                history.append({"role": "tool", "tool_call_id": tc["id"], "content": result})


if __name__ == "__main__":
    chat()
