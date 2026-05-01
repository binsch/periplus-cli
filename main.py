from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")

history = []

print("Chat with Qwen3 (type 'exit' or Ctrl+C to quit)\n")

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

    response = client.chat.completions.create(
        model="qwen3",
        messages=history,
        stream=True,
    )

    print("Assistant: ", end="", flush=True)
    reply = ""
    for chunk in response:
        delta = chunk.choices[0].delta.content or ""
        print(delta, end="", flush=True)
        reply += delta
    print()

    history.append({"role": "assistant", "content": reply})
