import os
import subprocess
import json
import re
import requests
from urllib.parse import quote
from bs4 import BeautifulSoup
from ddgs import DDGS
from mlx_vlm import load, generate
from mlx_audio.tts.models.kokoro import KokoroPipeline
from mlx_audio.tts.utils import load_model
import mlx_whisper
import sounddevice as sd
import numpy as np
from datetime import datetime
import time

model, processor = load("mlx-community/Qwen3.5-9B-MLX-4bit")

HF_REPO = "mlx-community/distil-whisper-large-v3"

voice_model_id = 'prince-canuma/Kokoro-82M'
voice_model = load_model(voice_model_id)
pipeline = KokoroPipeline(lang_code='a', model=voice_model, repo_id=voice_model_id)

SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
SILENCE_THRESHOLD = 0.01
SILENCE_DURATION = 1.5

def get_installed_apps() -> list[str]:
    app_dirs = [
        "/Applications",
        "/System/Applications",
        "/System/Applications/Utilities",
        os.path.expanduser("~/Applications")
    ]
    apps = []
    for d in app_dirs:
        if os.path.exists(d):
            apps += [f.replace(".app", "") for f in os.listdir(d) if f.endswith(".app")]
    return apps

BROWSER_APPS = {"Google Chrome", "Safari", "Firefox", "Arc", "Brave Browser"}
MUSIC_APPS = {"Music"}

installed = set(get_installed_apps())
browsers = installed & BROWSER_APPS
music_apps = installed & MUSIC_APPS

tools = {
    "open_app": {
        "description": "Open any installed application",
        "args": {"app": "name of any installed application"}
    },
    "quit_app": {
        "description": "Force quit an application",
        "args": {"app": "name of the application to quit"}
    },
    "web_search": {
        "description": "Search the web and return an accurate answer based on live results. Use for current events, distances, prices, scores, or anything where training data may be inaccurate.",
        "args": {"query": "search query string"}
    }
}

if browsers:
    tools["search_web"] = {
        "description": "Open a URL or search query in a browser",
        "args": {"query": "search term or full URL", "browser": f"one of: {browsers}"}
    }

if music_apps:
    tools["play_music"] = {
        "description": "Play a song or artist. Accepts natural queries like 'Churchill Downs by Jack Harlow' or 'songs by Drake'. Optionally shuffle.",
        "args": {
            "query": "full search string e.g. 'song name by artist'",
            "app": f"one of: {music_apps}",
            "shuffle": "true or false (optional, default false)"
        }
    }
    tools["pause_music"] = {
        "description": "Pause or resume currently playing music",
        "args": {"app": f"one of: {music_apps}"}
    }
    tools["shuffle_music"] = {
        "description": "Toggle shuffle on or off",
        "args": {"enabled": "true or false", "app": f"one of: {music_apps}"}
    }
    tools["repeat_music"] = {
        "description": "Set repeat mode",
        "args": {"mode": "one of: one, all, off", "app": f"one of: {music_apps}"}
    }
    tools["play_library"] = {
        "description": "Play a specific album or playlist from your local Music library. Optionally shuffle.",
        "args": {
            "type": "one of: album, playlist",
            "query": "album or playlist name",
            "app": f"one of: {music_apps}",
            "shuffle": "true or false (optional, default false)"
        }
    }

chat_history = [
    {
        "role": "system",
        "content": f'''
            Today's date is {datetime.now().strftime("%B %d, %Y")}.
            You are Qwen, a precise and practical AI assistant.

            You have access to tools, but you must ONLY output a JSON tool call when the user is explicitly and unambiguously requesting a system action — such as opening an app, playing music, or searching the web.

            NEVER output a JSON tool call for:
            - Greetings ("hey", "hi", "how are you", "what's up")
            - Casual conversation or small talk
            - Thank you messages or acknowledgements
            - General knowledge questions
            - Anything that is not a direct system action request

            If you are unsure whether the user wants a tool call, default to a plain text response.

            When a tool call IS appropriate, output exactly one raw JSON object and nothing else:
            {{"name": "tool_name", "arguments": {{"key": "value"}}}}

            Available tools:
            {json.dumps(tools, indent=2)}

            Tool names must be exactly: open_app, quit_app, search_web, play_music, pause_music, shuffle_music, repeat_music, play_library, web_search.

            Examples of tool calls:
            - "play Blinding Lights by The Weeknd" -> {{"name": "play_music", "arguments": {{"query": "Blinding Lights by The Weeknd", "app": "Music", "shuffle": "false"}}}}
            - "play my favorites on shuffle" -> {{"name": "play_library", "arguments": {{"type": "favorites", "query": "", "app": "Music", "shuffle": "true"}}}}
            - "play Certified Lover Boy album" -> {{"name": "play_library", "arguments": {{"type": "album", "query": "Certified Lover Boy", "app": "Music", "shuffle": "false"}}}}
            - "open YouTube on Chrome" -> {{"name": "search_web", "arguments": {{"query": "https://youtube.com", "browser": "Google Chrome"}}}}
            - "pause music" -> {{"name": "pause_music", "arguments": {{"app": "Music"}}}}
            - "stop playing" -> {{"name": "pause_music", "arguments": {{"app": "Music"}}}}
            - "close Music" -> {{"name": "quit_app", "arguments": {{"app": "Music"}}}}

            Examples of plain text responses (NO tool call):
            - "hey how are you" -> respond conversationally
            - "what's up" -> respond conversationally
            - "thanks" -> respond conversationally
            - "what is the capital of France" -> answer directly

            IMPORTANT: For current events, distances, prices, scores, or anything where your training data may be inaccurate — use web_search with today's date in the query where relevant.
            For all other questions, answer clearly and concisely. Do not invent facts or APIs. No internal monologues.
        '''
    }
]

def build_prompt(history):
    prompt = ""
    for message in history:
        prompt += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt

def strip_prefill(text: str) -> str:
    text = re.sub(r'<\|im_start\|>\s*assistant\s*', '', text)
    text = re.sub(r'<\|im_end\|>', '', text)
    return text.strip()

def get_response(prompt: str) -> str:
    response = generate(
        model,
        processor,
        prompt=build_prompt(chat_history + [{"role": "user", "content": prompt}]),
        temperature=0.3,
        repetition_penalty=1.2,
        verbose=False
    )
    raw = response.text if hasattr(response, "text") else str(response)
    return strip_prefill(raw)

def get_response_with_context(user_input: str, context: str) -> str:
    augmented = f"{user_input}\n\n[Web context]:\n{context}"
    response = generate(
        model,
        processor,
        prompt=f"<|im_start|>system\nYou are a helpful assistant. Answer the user's question using the provided web context. Be concise and accurate.<|im_end|>\n<|im_start|>user\n{augmented}<|im_end|>\n<|im_start|>assistant\n",
        temperature=0.3,
        repetition_penalty=1.2,
        verbose=False
    )
    raw = response.text if hasattr(response, "text") else str(response)
    return strip_prefill(raw)

def parse_tool_call(text: str) -> dict | None:
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    start = text.find('{')
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                try:
                    data = json.loads(text[start:i+1])
                    if "name" in data and "arguments" in data:
                        return data
                except json.JSONDecodeError:
                    return None
    return None

def fetch_page_text(url: str, max_chars: int = 1500) -> str:
    try:
        r = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return text[:max_chars]
    except Exception:
        return ""

deep_search_mode = False

def escape_applescript_string(text: str) -> str:
    return text.replace('\\', '\\\\').replace('"', '\\"')

def run_applescript(script: str) -> str:
    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        text=True,
        check=False
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "AppleScript failed")
    return result.stdout.strip()

def execute_tool(name: str, args: dict, user_input: str = "") -> str:
    name = name.replace(" ", "_").lower()
    try:
        if name == "open_app":
            app = args.get("app", "").strip()
            if app not in installed:
                return f"App '{app}' not found."
            subprocess.Popen(["open", "-a", app])
            return f"Opened {app}."

        elif name == "pause_music":
            app = args.get("app", next(iter(music_apps))).strip()
            if app == "Music":
                run_applescript('tell application "Music" to pause')
            elif app == "Spotify":
                run_applescript('tell application "Spotify" to pause')
            return f"Paused {app}."

        elif name == "quit_app":
            app = args.get("app", "").strip()
            run_applescript(f'tell application "{app}" to quit')
            return f"Quit {app}."

        elif name == "search_web":
            query = args.get("query", "").strip()
            browser = args.get("browser", next(iter(browsers))).strip()
            if not query.startswith("http"):
                query = "https://www.google.com/search?q=" + query.replace(" ", "+")
            subprocess.Popen(["open", "-a", browser, query])
            return f"Opened '{query}' in {browser}."

        elif name == "web_search":
            query = args.get("query", "").strip()
            print(f"[Searching web for: {query}]")
            variants = [
                query,
                f"{query} {datetime.now().strftime('%Y')}",
                f"{query} today {datetime.now().strftime('%B %d %Y')}"
            ]
            all_context = []
            seen_urls = set()
            for q in variants:
                with DDGS() as ddgs:
                    try:
                        results = list(ddgs.text(q, max_results=2))
                        for r in results:
                            url = r.get("href", "")
                            if url in seen_urls:
                                continue
                            seen_urls.add(url)
                            snippet = r.get("body", "")
                            page_text = fetch_page_text(url) if deep_search_mode else ""
                            all_context.append(f"Source: {url}\nSnippet: {snippet}\n{page_text}")
                    except Exception:
                        continue
            context = "\n\n---\n\n".join(all_context) if all_context else "No results found."
            answer = get_response_with_context(user_input, context)
            return sanitize_response(answer)

        elif name == "play_music":
            query = args.get("query", "").strip()
            if not query:
                return "No query specified for play_music."
            shuffle_raw = args.get("shuffle", "false")
            shuffle = str(shuffle_raw).strip().lower() == "true"
            if shuffle:
                run_applescript('tell application "Music" to set shuffle enabled to true')
            result = subprocess.run(
                ["shortcuts", "run", "Play Apple Music", "--input-string", query],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                subprocess.Popen(["open", f"music://music.apple.com/search?term={quote(query)}"])
                return f"Couldn't auto-play '{query}'. Opened Music search instead."
            return f"Playing '{query}' on Music{' with shuffle' if shuffle else ''}."

        elif name == "shuffle_music":
            enabled = args.get("enabled", "true").strip().lower()
            app = args.get("app", next(iter(music_apps))).strip()
            value = "true" if enabled == "true" else "false"
            if app == "Music":
                run_applescript(f'tell application "Music" to set shuffle enabled to {value}')
            return f"Shuffle {'enabled' if value == 'true' else 'disabled'} on {app}."

        elif name == "repeat_music":
            mode = args.get("mode", "all").strip().lower()
            app = args.get("app", next(iter(music_apps))).strip()
            applescript_mode = {"one": "one", "all": "all", "off": "off"}.get(mode, "off")
            if app == "Music":
                run_applescript(f'tell application "Music" to set song repeat to {applescript_mode}')
            return f"Repeat set to '{applescript_mode}' on {app}."

        elif name == "play_library":
            ptype = args.get("type", "playlist").strip().lower()
            query = args.get("query", "").strip()
            app = args.get("app", next(iter(music_apps))).strip()
            shuffle_raw = args.get("shuffle", "false")
            shuffle = str(shuffle_raw).strip().lower() == "true"
            if app == "Music":
                run_applescript('tell application "Music" to activate')
                time.sleep(2)
                if shuffle:
                    run_applescript('tell application "Music" to set shuffle enabled to true')
                if ptype == "album":
                    run_applescript(f'''tell application "Music"
                        activate
                        set sr to (every track of playlist "Library" whose album is "{query}")
                        if sr is not {{}} then play item 1 of sr
                    end tell''')
                elif ptype == "playlist":
                    run_applescript(f'''tell application "Music"
                        activate
                        play playlist "{query}"
                    end tell''')
                else:
                    return f"Unknown library type: {ptype}. Use album or playlist."
            return f"Playing {ptype}{' — ' + query if query else ''} on {app}{' with shuffle' if shuffle else ''}."

    except Exception as e:
        return f"Error executing {name}: {str(e)}"

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

def record_audio() -> np.ndarray:
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
        if not transcription.strip():
            print("Nothing detected, listening again.")
            continue

        print(f"You said: {transcription}")

        if transcription.strip().lower() in {"exit", "quit", "goodbye", "bye"}:
            speak("Goodbye!")
            break

        if "toggle deep search" in transcription.strip().lower():
            global deep_search_mode
            deep_search_mode = not deep_search_mode
            msg = f"Deep search mode {'enabled' if deep_search_mode else 'disabled'}."
            print(f"[System]: {msg}")
            speak(msg)
            continue

        response = get_response(transcription)
        tool_call = parse_tool_call(response)

        if tool_call:
            result = execute_tool(tool_call["name"], tool_call["arguments"], transcription)
            print(f"[Tool]: {result}")
            speak(result)
            chat_history.append({"role": "user", "content": transcription})
            chat_history.append({"role": "assistant", "content": result})
        else:
            sanitized = sanitize_response(response)
            print(f"Assistant: {sanitized}")
            speak(sanitized)
            chat_history.append({"role": "user", "content": transcription})
            chat_history.append({"role": "assistant", "content": sanitized})
        continue_chat = input("Do you want to continue the conversation? (yes/no): ").strip().lower()
        if continue_chat not in {"yes", "y"}:
            print("Ending conversation.")
            break

if __name__ == "__main__":
    start_chat()