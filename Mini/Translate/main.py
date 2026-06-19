from mlx_lm import load, generate
from mlx_audio.tts.models.kokoro import KokoroPipeline
from mlx_audio.tts.utils import load_model
import sounddevice as sd
import re
import numpy as np
import mlx_whisper

model, tokenizer = load("mlx-community/gemma-3-4b-it-4bit")

voice_model_id = "prince-canuma/Kokoro-82M"
voice_model = load_model(voice_model_id)

pipelines = {
    'spanish': KokoroPipeline(lang_code='e', model=voice_model, repo_id=voice_model_id),
    'french':  KokoroPipeline(lang_code='f', model=voice_model, repo_id=voice_model_id),
    'italian': KokoroPipeline(lang_code='i', model=voice_model, repo_id=voice_model_id),
}

VOICES = {
    'spanish': 'ef_dora',
    'french':  'ff_siwis',
    'italian': 'if_sara',
}

SUPPORTED_LANGS = list(pipelines.keys())

def translate(text, target_lang):
    messages = [
        {"role": "system", "content": "You are a translator. Output only the translation, nothing else. No preamble, no explanation."},
        {"role": "user",   "content": f"Translate to {target_lang}: {text}"}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=256,
        verbose=False
    )

    return response

def speak(text, target_lang):
    pipeline = pipelines[target_lang]
    voice    = VOICES[target_lang]

    sentences = re.split(r'(?<=[.!?])\s+', text)
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        for _, _, audio in pipeline(sentence, voice=voice, speed=1.0):
            sd.play(np.array(audio[0]), samplerate=24000)
            sd.wait()

def record_voice(duration=5, samplerate=16000):
    print(f"Recording for {duration}s... speak now.")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    print("Done recording.")
    audio_np = np.squeeze(audio)
    result = mlx_whisper.transcribe(audio_np, path_or_hf_repo="mlx-community/whisper-large-v3-turbo")
    return result["text"]

def chat_loop():
    print("VoiceTranslator — supported languages: Spanish, French, Italian")
    print("Type 'quit' to exit.\n")

    target_lang = ""
    while target_lang not in SUPPORTED_LANGS:
        target_lang = input("Enter target language (Spanish / French / Italian): ").strip().lower()
        if target_lang not in SUPPORTED_LANGS:
            print(f"Unsupported. Choose from: {', '.join(SUPPORTED_LANGS)}.")

    print(f"\nTarget language set to: {target_lang.capitalize()}")
    print("Input mode: type 't' for text, 'v' for voice.\n")

    while True:
        mode = input("[t/v/quit]: ").strip().lower()

        if mode in ['quit', 'exit', 'end']:
            print("Exiting.")
            break

        elif mode == 't':
            text = input("Enter text: ").strip()
            if not text:
                continue

        elif mode == 'v':
            text = record_voice(duration=5)
            print(f"Transcribed: {text}")

        else:
            print("Invalid mode. Type 't', 'v', or 'quit'.")
            continue

        translation = translate(text, target_lang)
        print(f"Translation: {translation}\n")
        speak(translation, target_lang)

if __name__ == "__main__":
    chat_loop()