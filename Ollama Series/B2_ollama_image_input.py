import ollama
from pathlib import Path

image_path = input("Enter the path to the image: ").strip()
path = Path(image_path)

if not path.exists():
    raise FileNotFoundError(f"Error: The file '{image_path}' does not exist.")

if not path.is_file():
    raise ValueError(f"Error: The file '{image_path}' does not exist.")

response = ollama.generate(
    model="qwen3-vl:2b",
    prompt="Describe this image.",
    images=[image_path],
    stream=True
)

for chunk in response:
    print(chunk["response"], end="")

print("\nDone")