import ollama

system_prompt = (
    "You are a direct, honest, and helpful assistant. "
    "Be concise, clear, and practical."
)

chat_history = []

while True:
    try:
        user_input = input("User: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chat.")
            break

        if user_input.lower() == "clear":
            chat_history = []
            print("Chat history cleared.")
            continue

        if user_input.lower() == "history":
            print("Chat History:")
            if not chat_history:
                print("No chat history available.")
                continue
            else:
                for message in chat_history:
                    print(f"{message['role'].capitalize()}: {message['content']}")
                continue
        
        if not user_input:
            print("Please enter a message.")
            continue

        chat_history.append({"role": "user", "content": user_input})
        
        response = ollama.chat(
            model="gemma3:4b",
            messages=[
                {"role": "system", "content": system_prompt}
            ] + chat_history,
            stream=True
        )

        assistant_response = ""
        print("Assistant: ", end="", flush=True)
        for chunk in response:
            assistant_response += chunk["message"]["content"]
            print(chunk["message"]["content"], end="", flush=True)
        print()

        chat_history.append({"role": "assistant", "content": assistant_response})
    except KeyboardInterrupt:
        print("\nExiting chat.")
        break
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        break
