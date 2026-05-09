import argparse
import json
from pathlib import Path
from openai import OpenAI

MEMORY_FILE = "MAP.md"
SYSTEM_PROMPT_FILE = "SYSTEM_PROMPT.md"
MODEL = "qwen3"
BASE_URL = "http://127.0.0.1:8080/v1"
THINKING_BUDGET = 4096

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

def complete(history: list, show_thinking: bool = False) -> tuple[str, list]:
    """Stream one completion turn. Returns (reply_text, tool_calls_list)."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=history,
        tools=TOOLS,
        stream=True,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": True},
            "thinking_budget_tokens": THINKING_BUDGET,
        },
    )

    reply = ""
    tool_calls_acc: dict[int, dict] = {}
    announced: set[int] = set()
    printed_prefix = False
    in_thinking = False

    for chunk in response:
        delta = chunk.choices[0].delta
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning and show_thinking:
            if not in_thinking:
                in_thinking = True
                print("\x1b[2m", end="", flush=True)
            print(reasoning, end="", flush=True)
        if delta.content:
            if in_thinking:
                in_thinking = False
                if show_thinking:
                    print("\x1b[0m", end="", flush=True)
            if not printed_prefix:
                print("Assistant: ", end="", flush=True)
                printed_prefix = True
            print(delta.content, end="", flush=True)
            reply += delta.content
        if delta.tool_calls:
            for tc in delta.tool_calls:
                entry = tool_calls_acc.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                if tc.id:
                    entry["id"] = tc.id
                if tc.function.name:
                    entry["name"] = tc.function.name
                    if tc.index not in announced:
                        announced.add(tc.index)
                        if printed_prefix:
                            print()
                        print(f"  → {tc.function.name}...", flush=True)
                if tc.function.arguments:
                    entry["arguments"] += tc.function.arguments

    if in_thinking and show_thinking:
        print("\x1b[0m", end="", flush=True)
    if printed_prefix:
        print()

    tool_calls = [
        {"id": v["id"], "type": "function", "function": {"name": v["name"], "arguments": v["arguments"]}}
        for v in (tool_calls_acc[k] for k in sorted(tool_calls_acc))
    ]
    return reply, tool_calls


def chat(show_thinking: bool = False):
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
            reply, tool_calls = complete(history, show_thinking)

            if not tool_calls:
                history.append({"role": "assistant", "content": reply})
                break

            history.append({"role": "assistant", "content": reply or None, "tool_calls": tool_calls})

            for tc in tool_calls:
                result = execute_tool(tc["function"]["name"], tc["function"]["arguments"])
                print(f"\x1b[2m[{tc['function']['name']}] {result}\x1b[0m")
                history.append({"role": "tool", "tool_call_id": tc["id"], "content": result})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-thinking", action="store_true", help="Display model thinking output (dimmed)")
    args = parser.parse_args()
    chat(show_thinking=args.show_thinking)
