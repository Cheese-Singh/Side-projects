from mlx_vlm import load, generate

model, processor = load("mlx-community/Qwen3.5-9B-MLX-4bit")

def clean_output(text):
    if not isinstance(text, str):
        text = str(text)

    stop_tokens = ["<|im_end|>", "<|endoftext|>", "<|im_start|>", "<|im_start|>user"]
    
    for token in stop_tokens:
        if token in text:
            text = text.split(token)[0].strip()
    return text.strip()

def build_prompt(chat_history):
    prompt = ""
    for message in chat_history:
        role = message["role"]
        content = message["content"]

        prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"

    prompt += "<|im_start|>assistant\n"
    return prompt

def start_conversation():
    chat_history = [
        {
            "role": "system",
            "content": (
                "You are Qwen, a precise and practical AI assistant. "
                "Answer clearly and concisely. "
                "If you do not know something, say so. "
                "Do not invent facts, code behavior, citations, file names, or package APIs."
            )
        }
    ]
    
    while True:
        try:
            user_input = input("User: ").strip()

            if user_input.lower() in ["exit", "quit"]:
                print("Exiting chat.")
                break

            if user_input.lower() == "clear":
                chat_history = chat_history[:1]
                print("Chat history cleared.")
                continue

            if user_input.lower() == "history":
                print("Chat History:")
                if len(chat_history) <= 1:
                    print("No chat history available.")
                    continue
                else:
                    for message in chat_history:
                        print(f"{message['role'].capitalize()}: {message['content']}")
                    continue
                
            if not user_input:
                print("Please enter a message.")
                continue

            print("Qwen is thinking...", end="\r", flush=True)

            chat_history.append({"role": "user", "content": user_input})
            response = generate(
                model,
                processor,
                prompt=build_prompt(chat_history),
                max_tokens=512,
                temperature=0.2,
                repetition_penalty=1.1
            )

            cleaned_response = clean_output(response.text)
            chat_history.append({"role": "assistant", "content": cleaned_response})
            print(f"Qwen: {cleaned_response}\n")

        except Exception as e:
            print(f"An error occurred: {e}")
            continue

if __name__ == "__main__":
    print("Welcome to the Qwen Chat! Type 'exit' or 'quit' to end the conversation.")
    print("Type 'clear' to clear chat history, 'history' to view chat history.\n")
    start_conversation()