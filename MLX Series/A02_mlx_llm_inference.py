from mlx_vlm import load, generate

model, processor = load("mlx-community/Qwen3.5-9B-MLX-4bit")

def clean_output(text):
    if isinstance(text, str):
        text = str(text)

    stop_tokens = ["<|im_end|>", "<|endoftext|>", "<|im_start|>", "<|im_start|>user"]
    
    for token in stop_tokens:
        if token in text:
            text = text.split(token)[0].strip()
    return text.strip()

def inspect_model():
    print(model, "\n")
    print(model.parameters(), "\n")
    print(dir(model.layers[0]))

def run_single_turn_chat():
    while True:
        prompt = input("User prompt: ")

        if prompt.lower() in ["exit", "quit"]: break

        print("Qwen is thinking...", end="\r", flush=True)

        response = generate(
            model,
            processor,
            prompt=prompt,
            verbose=False,
            max_tokens=512,
            temperature=0.1,
            repetition_penalty = 1.1
        )
        
        reply = clean_output(response.text)
        print(f"\nQwen: {reply}\n")

def main():
    run_single_turn_chat()
    
if __name__ == "__main__":
    main()