import sounddevice as sd
from mlx_audio.tts.models.kokoro import KokoroPipeline
from mlx_audio.tts.utils import load_model
import mlx_whisper
from mlx_vlm import load, generate
import numpy as np
import re

SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
SILENCE_THRESHOLD = 0.01
SILENCE_DURATION = 1.5

model, processor = load("mlx-community/Qwen3.5-9B-MLX-4bit")
HF_REPO = "mlx-community/distil-whisper-large-v3"
voice_model_id = 'prince-canuma/Kokoro-82M'
voice_model = load_model(voice_model_id)
pipeline = KokoroPipeline(lang_code='a', model=voice_model, repo_id=voice_model_id)

def record_audio()->np.ndarray:
    frames = []
    silent_chunks = 0
    max_silent = int(SILENCE_DURATION * SAMPLE_RATE / CHUNK_SIZE)

    print("Listening... Speak now.")
    with sd.InputStream(samplerate=SAMPLE_RATE, dtype="float32", channels=1, blocksize=CHUNK_SIZE) as stream:
        while True:
            data, _ = stream.read(CHUNK_SIZE)
            frames.append(data.copy())
            rms = np.sqrt(np.mean(data**2))
            peak = np.max(np.abs(data))
            is_silent = rms < SILENCE_THRESHOLD and peak < SILENCE_THRESHOLD * 3
            if is_silent:
                silent_chunks += 1
                if silent_chunks >= max_silent:
                    break
            else:
                silent_chunks = 0

    print("Recording stopped.")
    return np.concatenate(frames).flatten()

def transcribe_audio(audio_data: np.ndarray) -> str:
    result = mlx_whisper.transcribe(
        audio_data,
        path_or_hf_repo=HF_REPO,
        language="en",
        task="transcribe",
        word_timestamps=True,
        verbose=False,
    )
    return result["text"] if "text" in result else ""

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
        for _, _, audio in pipeline(sentence, voice='af_heart', speed=1.3):
            sd.play(np.array(audio[0]), samplerate=24000)
            sd.wait()

def start_chat():
    while True:
        audio_data = record_audio()
        transcription = transcribe_audio(audio_data)
        print(f"You said: {transcription}")
        response = get_response(transcription)
        sanitized_response = sanitize_response(response)
        print(f"Assistant: {sanitized_response}")
        speak(sanitized_response)
        chat_history.append({"role": "user", "content": transcription})
        chat_history.append({"role": "assistant", "content": sanitized_response})
        continue_chat = input("Do you want to continue the conversation? (yes/no): ").strip().lower()
        if continue_chat not in {"yes", "y"}:
            print("Ending conversation.")
            break 

if __name__ == "__main__":
    start_chat()