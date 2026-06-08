import mlx_whisper
from mlx_vlm import load, generate
import numpy as np
import re
import os
import wave
import subprocess
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

model, processor = load("mlx-community/Qwen3.5-9B-MLX-4bit")
HF_REPO = "mlx-community/distil-whisper-large-v3"

app = FastAPI()

system_prompt = {
    "role": "system",
    "content": (
        "You are a precise summarisation assistant. The user will provide a transcription of spoken audio input."
        "Your job is to return a clean, concise summary of what was said — preserving all key points, removing filler words, false starts, and repetition."
        "Output only the summary. No preamble, no labels, no explanation."
        "Write in clear third-person or neutral prose unless the content is clearly a personal note or reminder, in which case preserve first-person."
        "Keep it under 5 sentences unless the input is long enough to warrant more."
        "Do not overthink or include internal monologues in your responses."
    )
}

def convert_to_wav(input_path: str, output_path: str):
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", output_path],
        check=True,
        capture_output=True
    )
    print("ffmpeg stdout:", result.stdout.decode())
    print("ffmpeg stderr:", result.stderr.decode())

def check_wav(wav_path: str):
    with wave.open(wav_path, 'r') as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        print(f"WAV duration: {duration:.2f}s, frames: {frames}, rate: {rate}")

def transcribe_audio(wav_path: str) -> str:
    result = mlx_whisper.transcribe(
        wav_path,
        path_or_hf_repo=HF_REPO,
        language="en",
        task="transcribe",
        word_timestamps=True,
        verbose=False,
    )
    return result["text"] if "text" in result else ""

def build_prompt(chat_history):
    prompt = ""
    for message in chat_history:
        prompt += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt

def get_response(transcription: str) -> str:
    chat_history = [
        system_prompt,
        {"role": "user", "content": transcription}
    ]
    response = generate(
        model,
        processor,
        prompt=build_prompt(chat_history),
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

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/transcribe-and-summarise")
async def transcribe_and_summarise(audio: UploadFile = File(...)):
    debug_dir = "/Users/ekamveersingh/Documents/GitHub/Side-projects/Mini/debug_audio"
    os.makedirs(debug_dir, exist_ok=True)

    raw_path = os.path.join(debug_dir, "input" + os.path.splitext(audio.filename)[1])
    wav_path = os.path.join(debug_dir, "converted.wav")

    with open(raw_path, "wb") as f:
        f.write(await audio.read())

    convert_to_wav(raw_path, wav_path)
    check_wav(wav_path)

    transcription = transcribe_audio(wav_path)
    print(f"Transcription: {transcription}")

    summary = sanitize_response(get_response(transcription))
    print(f"Summary: {summary}")

    return JSONResponse({"transcription": transcription, "summary": summary})