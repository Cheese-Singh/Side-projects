import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from mlx_vlm import load, generate

device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float16 if device == "mps" else torch.float32

model_id = "openai/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    dtype=dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True,
)

model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    dtype=dtype,
    device=device,
)

llm_model, llm_processor = load("mlx-community/Qwen3.5-9B-MLX-4bit")

def get_response(prompt):
    response = generate(
        llm_model,
        llm_processor,
        prompt=prompt,
        temperature=0.1,
        repetition_penalty=1.2,
        max_tokens=512,
        verbose=False
    )

    if hasattr(response, "text"):
        return response.text
    return str(response)

def record_audio(duration=5):
    import pyaudio
    import wave

    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("Recording...")
    frames = []
    for _ in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Finished recording.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open("prompt.wav", 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    print("Recording saved as prompt.wav")

def transcribe(audio_file_path):
    print("Transcribing...")
    result = pipe(
        audio_file_path,
        generate_kwargs={
            "language": "en",
            "task": "transcribe"
        }
    )
    return result["text"]

if __name__ == "__main__":
    record_audio()
    text_prompt = transcribe("./prompt.wav")
    print(f"User: {text_prompt}")
    reply = get_response(text_prompt)
    print(f"Qwen: {reply}")