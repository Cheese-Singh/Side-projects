from mlx_vlm import load, generate

model, processor = load("mlx-community/Qwen3.5-9B-MLX-4bit")

while True:
    prompt = input("User prompt: ")

    if prompt.lower() in ["exit", "quit"]: 
        break

    text = generate(
        model,
        processor,
        prompt=prompt,
        verbose=True,
        max_tokens=512,
        temperature=0.7,
        repetition_penalty = 1.1)
        
inspect = input("Inspect model? (yes/no): ")
if inspect.lower() == "yes":
    print(model, "\n")
    print(model.parameters(), "\n")
    print(dir(model.layers[0]))