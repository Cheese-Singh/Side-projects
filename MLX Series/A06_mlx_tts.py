from mlx_audio.tts.models.kokoro import KokoroPipeline
from mlx_audio.tts.utils import load_model
import sounddevice as sd
import numpy as np
from mlx_vlm import load, generate
import re

voice_model_id = 'prince-canuma/Kokoro-82M'
voice_model = load_model(voice_model_id)
model, processor = load("mlx-community/Qwen3.5-9B-MLX-4bit")
pipeline = KokoroPipeline(lang_code='a', model=voice_model, repo_id=voice_model_id)

chat_history = [
    {
        "role": "system",
        "content": (
            "You are Qwen, a precise and practical AI assistant. "
            "Answer clearly and concisely. "
            "If you do not know something, say so. "
            "Do not invent facts, code behavior, citations, file names, or package APIs."
            "Do not overthink or include internal monologues in your responses."
        )
    }
]

def build_prompt(chat_history):
    prompt = ""
    for message in chat_history:
        role = message["role"]
        content = message["content"]

        prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"

    prompt += "<|im_start|>assistant\n"
    return prompt

def get_response(prompt: str) -> str:
    response = generate(
        model,
        processor,
        prompt=build_prompt(chat_history + [{"role": "user", "content": prompt}]),
        temperature=0.3,
        repetition_penalty=1.2,
        verbose=False
    )
    return response.text if hasattr(response, "text") else str(response)

def sanitize_response(text: str) -> str:
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think[^>]*>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'\*{1,3}(.+?)\*{1,3}', r'\1', text)
    text = re.sub(r'#{1,6}\s*', '', text)
    text = re.sub(r'`{1,3}[^`]*`{1,3}', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'[_~>|*]', ' ', text)
    text = text.encode('ascii', errors='ignore').decode('ascii')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def speak(text: str):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        for _, _, audio in pipeline(sentence, voice='af_heart', speed=1.2):
            sd.play(np.array(audio[0]), samplerate=24000)
            sd.wait()

def start_chat():
    while True:
        prompt = input("You: ")
        if prompt.lower() == "exit":
            break
        if prompt.lower() == "clear":
            chat_history[:] = chat_history[:1]
            print("Chat history cleared.")
            continue

        chat_history.append({"role": "user", "content": prompt})
        print("Thinking...        ", end="\r", flush=True)

        response = get_response(prompt)
        clean = sanitize_response(response)
        chat_history.append({"role": "assistant", "content": clean})
        print(f"Assistant: {clean}")
        speak(clean)

if __name__ == "__main__":
    start_chat()