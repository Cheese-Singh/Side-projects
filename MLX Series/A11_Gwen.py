import os
import subprocess
import json
import re
import sqlite3
import hashlib
import time
import requests
import numpy as np
from urllib.parse import quote
from bs4 import BeautifulSoup
from ddgs import DDGS
from mlx_vlm import load, generate
from mlx_lm import load as load_lm, generate as generate_lm
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from mlx_audio.tts.models.kokoro import KokoroPipeline
from mlx_audio.tts.utils import load_model
import mlx_whisper
import sounddevice as sd
from datetime import datetime

# -------
# MODELS
# -------

model, processor = load("mlx-community/Qwen3.5-9B-MLX-4bit")

SMALL_MODEL_REPO = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
try:
    small_model, small_tokenizer = load_lm(SMALL_MODEL_REPO)
    SMALL_MODEL_AVAILABLE = True
except Exception:
    print(f"[Router] Small model not found at {SMALL_MODEL_REPO}. Falling back to primary for all queries.")
    small_model, small_tokenizer = model, processor
    SMALL_MODEL_AVAILABLE = False

HF_REPO = "mlx-community/distil-whisper-large-v3"

voice_model_id = 'prince-canuma/Kokoro-82M'
voice_model = load_model(voice_model_id)
pipeline = KokoroPipeline(lang_code='a', model=voice_model, repo_id=voice_model_id)

SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
SILENCE_THRESHOLD = 0.01
SILENCE_DURATION = 1.5

# -----------------
# WAKE WORD CONFIG
# -----------------

WAKE_WORDS = {"gwen", "hey gwen", "hi gwen", "okay gwen"}

WAKE_TIMEOUT = 30.0

CONVERSATION_TIMEOUT = 15.0 

DEACTIVATION_PHRASES = {
    "deactivate", "go to sleep", "stop listening", "sleep", "go to sleep gwen"
}

EXIT_PHRASES = {
    "exit", "quit", "shut down", "shutdown"
}

FORCE_SEARCH_PATTERNS = [
    r"\b(world cup|premier league|champions league|nba|nfl)\b",
    r"\b(score|standings|match|tournament|who (won|is winning|is leading))\b",
    r"\b(price|stock|weather|news|latest|current|right now|today)\b",
    r"\b(born on|birthday|who is|how old is|when did|what year)\b",
    r"\b(travel time|distance|travel|cost)\b"
]

# ----------------
# DATABASE SETUP
# -----------------

DB_PATH = os.path.expanduser("~/daughter_ai.db")


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS user_facts (
            key         TEXT PRIMARY KEY,
            value       TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS user_facts_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            key         TEXT NOT NULL,
            old_value   TEXT,
            new_value   TEXT NOT NULL,
            changed_at  TEXT NOT NULL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS search_cache (
            query_hash  TEXT PRIMARY KEY,
            query       TEXT NOT NULL,
            answer      TEXT NOT NULL,
            cached_at   TEXT NOT NULL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS session_summaries (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            summary     TEXT NOT NULL,
            created_at  TEXT NOT NULL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS latency_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      TEXT NOT NULL,
            query_preview   TEXT NOT NULL,
            complexity      TEXT NOT NULL,
            stt_ms          REAL,
            memory_ms       REAL,
            classifier_ms   REAL,
            inference_ms    REAL,
            tts_ms          REAL,
            total_ms        REAL,
            logged_at       TEXT NOT NULL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS memory_retrieval_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      TEXT NOT NULL,
            query_preview   TEXT NOT NULL,
            facts_injected  TEXT,
            semantic_hits   TEXT,
            logged_at       TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


init_db()

# ---------------------------
# CHROMADB — SEMANTIC MEMORY
# ----------------------------

CHROMA_AVAILABLE = False
chroma_collection = None

try:
    import chromadb
    from chromadb.utils import embedding_functions

    chroma_client = chromadb.PersistentClient(path=os.path.expanduser("~/daughter_ai_chroma"))
    ef = embedding_functions.DefaultEmbeddingFunction()
    chroma_collection = chroma_client.get_or_create_collection(
        name="daughter_memory",
        embedding_function=ef
    )
    CHROMA_AVAILABLE = True
    print("[Memory] ChromaDB loaded.")
except ImportError:
    print("[Memory] ChromaDB not installed. Run: pip install chromadb sentence-transformers")
except Exception as e:
    print(f"[Memory] ChromaDB init failed: {e}")


def store_semantic_memory(text: str, metadata: dict = None):
    if not CHROMA_AVAILABLE:
        return
    doc_id = hashlib.md5((text + datetime.now().isoformat()).encode()).hexdigest()
    chroma_collection.add(
        documents=[text],
        metadatas=[metadata or {}],
        ids=[doc_id]
    )


def retrieve_semantic_memory(query: str, n_results: int = 3) -> list[str]:
    if not CHROMA_AVAILABLE:
        return []
    try:
        count = chroma_collection.count()
        if count == 0:
            return []
        results = chroma_collection.query(
            query_texts=[query],
            n_results=min(n_results, count)
        )
        return results["documents"][0] if results["documents"] else []
    except Exception as e:
        print(f"[Memory] Semantic retrieval failed: {e}")
        return []

# ------------------
# GLOBAL USER STATE
# ------------------

def get_all_facts() -> dict:
    conn = get_db()
    rows = conn.execute("SELECT key, value FROM user_facts").fetchall()
    conn.close()
    return {r["key"]: r["value"] for r in rows}


def set_fact(key: str, value: str):
    conn = get_db()
    now = datetime.now().isoformat()
    existing = conn.execute("SELECT value FROM user_facts WHERE key = ?", (key,)).fetchone()
    old_value = existing["value"] if existing else None
    conn.execute(
        "INSERT OR REPLACE INTO user_facts (key, value, updated_at) VALUES (?, ?, ?)",
        (key, value, now)
    )
    conn.execute(
        "INSERT INTO user_facts_history (key, old_value, new_value, changed_at) VALUES (?, ?, ?, ?)",
        (key, old_value, value, now)
    )
    conn.commit()
    conn.close()


def build_global_context() -> str:
    facts = get_all_facts()
    if not facts:
        return ""
    lines = ["[User context]"]
    for k, v in facts.items():
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)

# ------------------
# AUTO-UPDATE LOGIC
# ------------------

AUTO_UPDATE_PROMPT = """You are a fact extractor. Given a user message, extract any personal facts the user is stating about themselves (name, location, current projects, stopped projects, preferences, occupation, etc.).

Return a JSON array of objects with "key" and "value". Use snake_case keys. If no facts are stated, return [].
Return ONLY the JSON array, nothing else.

Examples:
- "I've stopped working on project X" → [{{"key": "stopped_projects", "value": "project X"}}]
- "I moved to Chandigarh" → [{{"key": "location", "value": "Chandigarh"}}]
- "my name is Veer" → [{{"key": "name", "value": "Veer"}}]
- "hey what's up" → []

User message: {message}"""


def detect_and_apply_fact_updates(user_message: str):
    prompt = AUTO_UPDATE_PROMPT.format(message=user_message)
    try:
        response = generate_lm(
            small_model,
            small_tokenizer,
            prompt=f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
            max_tokens=200,
            sampler=make_sampler(temp=0.0),
            verbose=False
        )
        raw = response.text if hasattr(response, "text") else str(response)
        raw = strip_think_tags(raw).strip()
        start = raw.find('[')
        end = raw.rfind(']')
        if start == -1 or end == -1:
            return
        facts = json.loads(raw[start:end+1])
        for f in facts:
            if "key" in f and "value" in f:
                set_fact(f["key"], f["value"])
                store_semantic_memory(
                    f"User fact — {f['key']}: {f['value']}",
                    {"type": "user_fact", "key": f["key"]}
                )
                print(f"[Memory] Updated fact: {f['key']} = {f['value']}")
    except Exception as e:
        print(f"[Memory] Auto-update failed: {e}")

# ------------------
# WEB SEARCH CACHE
# ------------------

CACHE_TTL_HOURS = 6


def _query_hash(query: str) -> str:
    return hashlib.md5(query.strip().lower().encode()).hexdigest()


def get_cached_answer(query: str) -> str | None:
    conn = get_db()
    row = conn.execute(
        "SELECT answer, cached_at FROM search_cache WHERE query_hash = ?",
        (_query_hash(query),)
    ).fetchone()
    conn.close()
    if not row:
        return None
    cached_at = datetime.fromisoformat(row["cached_at"])
    age_hours = (datetime.now() - cached_at).total_seconds() / 3600
    if age_hours > CACHE_TTL_HOURS:
        return None
    return row["answer"]


def cache_answer(query: str, answer: str):
    conn = get_db()
    conn.execute(
        "INSERT OR REPLACE INTO search_cache (query_hash, query, answer, cached_at) VALUES (?, ?, ?, ?)",
        (_query_hash(query), query, answer, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

# ------------------
# LATENCY LOGGER
# ------------------

SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")


def log_latency(query: str, complexity: str, timings: dict):
    conn = get_db()
    conn.execute("""
        INSERT INTO latency_log
            (session_id, query_preview, complexity, stt_ms, memory_ms, classifier_ms, inference_ms, tts_ms, total_ms, logged_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        SESSION_ID, query[:80], complexity,
        timings.get("stt_ms"), timings.get("memory_ms"), timings.get("classifier_ms"),
        timings.get("inference_ms"), timings.get("tts_ms"), timings.get("total_ms"),
        datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()


def log_memory_retrieval(query: str, facts: dict, semantic_hits: list[str]):
    conn = get_db()
    conn.execute("""
        INSERT INTO memory_retrieval_log
            (session_id, query_preview, facts_injected, semantic_hits, logged_at)
        VALUES (?, ?, ?, ?, ?)
    """, (
        SESSION_ID, query[:80],
        json.dumps(facts), json.dumps(semantic_hits),
        datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()

# ----------------------
# COMPLEXITY CLASSIFIER
# ----------------------

TOOL_TRIGGER_WORDS = {
    "open", "launch", "start", "close", "quit", "kill",
    "play", "pause", "stop", "resume", "shuffle", "repeat",
    "search", "google", "look up", "find", "browse",
    "go to", "navigate"
}

SIMPLE_QUESTION_PATTERNS = [
    r"^(hi|hey|hello|what'?s up|how are you|good morning|good night|thanks|thank you|bye|goodbye)[\W]*$",
    r"^what (time|day|date) is it",
    r"^(set|create) (a )?(timer|alarm|reminder)",
    r"^(what is|what's|whats) the (capital|currency|language|population) of",
    r"^(convert|translate) .{0,40}$",
    r"^(how (do|does)|what (does|is)) .{0,30}\?$",
]


def classify_complexity(query: str) -> str:
    q = query.strip().lower()
    first_word = q.split()[0] if q.split() else ""
    if first_word in TOOL_TRIGGER_WORDS:
        return "simple"
    word_count = len(q.split())
    if word_count <= 25:
        return "simple"
    for pattern in SIMPLE_QUESTION_PATTERNS:
        if re.match(pattern, q, re.IGNORECASE):
            return "simple"
        
    complex_signals = [
        "explain", "why", "how does", "compare", "difference between",
        "write", "code", "implement", "design", "analyze", "summarize",
        "what are the", "pros and cons", "help me", "should i",
        "plan", "strategy", "research", "paper", "essay",
        "who", "born", "birthday", "when did", "what year",
        "famous", "history", "did you know" 
    ]
    for signal in complex_signals:
        if signal in q:
            return "complex"
    return "complex" if word_count > 25 else "simple"

# ---------------
# APP DETECTION
# ---------------

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

# ------
# TOOLS
# ------

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
        "description": "Search the web and return an accurate answer. Use for current events, prices, scores, or anything time-sensitive.",
        "args": {"query": "search query string"}
    },
    "update_memory": {
        "description": "Explicitly store or update a fact about the user in long-term memory.",
        "args": {"key": "fact key in snake_case", "value": "fact value"}
    }
}

if browsers:
    tools["search_web"] = {
        "description": "Open a URL or search query in a browser",
        "args": {"query": "search term or full URL", "browser": f"one of: {browsers}"}
    }

if music_apps:
    tools["play_music"] = {
        "description": "Play a song or artist in Apple Music.",
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
        "description": "Play a specific album or playlist from your local Music library.",
        "args": {
            "type": "one of: album, playlist",
            "query": "album or playlist name",
            "app": f"one of: {music_apps}",
            "shuffle": "true or false (optional, default false)"
        }
    }

# --------------
# SYSTEM PROMPT
# --------------
SYSTEM_PROMPT_BASE = """Today's date is {date}. Current time is {time}
You are Gwen, a precise and practical AI assistant running entirely on-device.

{global_context}

{semantic_context}

You have access to tools, but ONLY output a JSON tool call when the user is explicitly requesting a system action.

NEVER output a JSON tool call for greetings, casual conversation, or thank-you messages.

When a tool call IS appropriate, output exactly one raw JSON object and nothing else:
{{"name": "tool_name", "arguments": {{"key": "value"}}}}

Available tools:
{tools}

Tool names must be exactly: {tool_names}.

Examples of tool calls:
- "play Blinding Lights by The Weeknd" -> {{"name": "play_music", "arguments": {{"query": "Blinding Lights by The Weeknd", "app": "Music", "shuffle": "false"}}}}
- "open YouTube on Chrome" -> {{"name": "search_web", "arguments": {{"query": "https://youtube.com", "browser": "Google Chrome"}}}}
- "pause music" -> {{"name": "pause_music", "arguments": {{"app": "Music"}}}}
- "remember I stopped working on project X" -> {{"name": "update_memory", "arguments": {{"key": "stopped_projects", "value": "project X"}}}}

Examples of plain text (NO tool call):
- "hey how are you" -> respond conversationally
- "what is the capital of France" -> answer directly

STRICT RULES — follow these without exception:
- For ANY question about people, birthdays, dates, facts, history, news, prices, scores, or events — ALWAYS call web_search. NEVER answer from memory.
- If you are not 100 percent certain of a fact, call web_search instead of guessing.
- Answer in 1-3 sentences maximum. No lists unless explicitly asked.
- No internal monologue. No self-correction mid-response. No hedging.
- If you start to be unsure, stop and call web_search."""


def build_system_prompt(global_context: str, semantic_context: list[str]) -> str:
    return SYSTEM_PROMPT_BASE.format(
        date=datetime.now().strftime("%B %d, %Y"),
        time=datetime.now().strftime("%I:%M %p"),
        global_context=global_context if global_context else "",
        semantic_context=f"[Relevant memories]\n{chr(10).join(semantic_context)}" if semantic_context else "",
        tools=json.dumps(tools, indent=2),
        tool_names=", ".join(tools.keys())
    )

# ------------------------
# IN-SESSION CHAT HISTORY
# ------------------------

chat_history: list[dict] = []


def build_prompt(user_input: str, system_prompt: str) -> str:
    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    for message in chat_history:
        prompt += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
    prompt += f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
    return prompt

# ---------------------------------------
# RESPONSE GENERATION — two-tier router
# ---------------------------------------

def strip_think_tags(text: str) -> str:
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def strip_prefill(text: str) -> str:
    text = re.sub(r'<\|im_start\|>\s*assistant\s*', '', text)
    text = re.sub(r'<\|im_end\|>', '', text)
    return text.strip()


def get_response(user_input: str, complexity: str, system_prompt: str) -> str:
    if complexity == "simple":
        response = generate_lm(
            small_model,
            small_tokenizer,
            prompt=build_prompt(user_input, system_prompt),
            max_tokens=512,
            sampler=make_sampler(temp=0.3),
            logits_processors=make_logits_processors(repetition_penalty=1.3),
            verbose=False
        )
    else:
        response = generate(
            model, processor,
            prompt=build_prompt(user_input, system_prompt),
            temperature=0.1,
            max_tokens = 1024,
            verbose=False
        )
    raw = response.text if hasattr(response, "text") else str(response)
    return strip_prefill(strip_think_tags(raw))


def get_response_with_context(user_input: str, context: str) -> str:
    augmented = f"{user_input}\n\n[Web context]:\n{context}"
    response = generate(
        model, processor,
        prompt=f"<|im_start|>system\nYou are a helpful assistant. Answer using the provided web context. Be concise and accurate.<|im_end|>\n<|im_start|>user\n{augmented}<|im_end|>\n<|im_start|>assistant\n",
        temperature=0.1,
        max_tokens = 1024,
        verbose=False
    )
    raw = response.text if hasattr(response, "text") else str(response)
    return strip_prefill(strip_think_tags(raw))

# -------------------
# TOOL CALL PARSING
# -------------------

def parse_tool_call(text: str) -> dict | None:
    text = strip_think_tags(text)
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

# -----------
# WEB SEARCH
# -----------

deep_search_mode = False


def fetch_page_text(url: str, max_chars: int = 1500) -> str:
    try:
        r = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)[:max_chars]
    except Exception:
        return ""


def do_web_search(query: str, user_input: str, force_fresh: bool = False) -> str:
    if not force_fresh:
        cached = get_cached_answer(query)
        if cached:
            print(f"[Search] Cache hit for: {query}")
            return cached

    print(f"[Search] Fetching: {query}")
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
    answer = sanitize_response(answer)
    cache_answer(query, answer)
    store_semantic_memory(
        f"Web search: {query} → {answer[:200]}",
        {"type": "web_search", "query": query}
    )
    return answer

# ------------
# APPLESCRIPT
# ------------

def run_applescript(script: str) -> str:
    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "AppleScript failed")
    return result.stdout.strip()

# --------------
# TOOL EXECUTOR
# --------------

def execute_tool(name: str, args: dict, user_input: str = "") -> str:
    name = name.replace(" ", "_").lower()
    try:
        if name == "open_app":
            app = args.get("app", "").strip()
            if app not in installed:
                return f"App '{app}' not found."
            subprocess.Popen(["open", "-a", app])
            return f"Opened {app}."

        elif name == "quit_app":
            app = args.get("app", "").strip()
            run_applescript(f'tell application "{app}" to quit')
            return f"Quit {app}."

        elif name == "pause_music":
            app = args.get("app", next(iter(music_apps))).strip()
            run_applescript(f'tell application "{app}" to pause')
            return f"Paused {app}."

        elif name == "search_web":
            query = args.get("query", "").strip()
            browser = args.get("browser", next(iter(browsers))).strip()
            if not query.startswith("http"):
                query = "https://www.google.com/search?q=" + query.replace(" ", "+")
            subprocess.Popen(["open", "-a", browser, query])
            return f"Opened '{query}' in {browser}."

        elif name == "web_search":
            query = args.get("query", "").strip()
            force_fresh = any(w in user_input.lower() for w in ["latest", "again", "refresh", "new search"])
            return do_web_search(query, user_input, force_fresh=force_fresh)

        elif name == "update_memory":
            key = args.get("key", "").strip()
            value = args.get("value", "").strip()
            if key and value:
                set_fact(key, value)
                store_semantic_memory(
                    f"User fact — {key}: {value}",
                    {"type": "user_fact", "key": key}
                )
                return f"Got it, I'll remember that {key.replace('_', ' ')}: {value}."
            return "Couldn't store that — missing key or value."

        elif name == "play_music":
            query = args.get("query", "").strip()
            if not query:
                return "No query specified for play_music."
            shuffle = str(args.get("shuffle", "false")).strip().lower() == "true"
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
            run_applescript(f'tell application "{app}" to set shuffle enabled to {value}')
            return f"Shuffle {'enabled' if value == 'true' else 'disabled'} on {app}."

        elif name == "repeat_music":
            mode = args.get("mode", "all").strip().lower()
            app = args.get("app", next(iter(music_apps))).strip()
            applescript_mode = {"one": "one", "all": "all", "off": "off"}.get(mode, "off")
            run_applescript(f'tell application "{app}" to set song repeat to {applescript_mode}')
            return f"Repeat set to '{applescript_mode}' on {app}."

        elif name == "play_library":
            ptype = args.get("type", "playlist").strip().lower()
            query = args.get("query", "").strip()
            app = args.get("app", next(iter(music_apps))).strip()
            shuffle = str(args.get("shuffle", "false")).strip().lower() == "true"
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

# -------------------
# RESPONSE SANITIZER
# -------------------

def sanitize_response(text: str) -> str:
    text = strip_think_tags(text)
    text = re.sub(r'\*{1,3}(.+?)\*{1,3}', r'\1', text)
    text = re.sub(r'#{1,6}\s*', '', text)
    text = re.sub(r'`{1,3}[^`]*`{1,3}', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'[_~>|*]', ' ', text)
    text = text.encode('ascii', errors='ignore').decode('ascii')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -----------------------------------------
# AUDIO — beeps, record, transcribe, speak
# -----------------------------------------

def play_beep(style: str = "activate"):
    """
    Generate and play a beep sound. No external audio files needed.
    activate   → two ascending tones (Siri-style, "I'm listening")
    deactivate → single low tone    ("going to sleep")
    timeout    → two descending tones ("timed out")
    """
    sr = 24000
    duration = 0.12
    t = np.linspace(0, duration, int(sr * duration), False)
    fade = int(0.01 * sr)

    def make_tone(freq: float, vol: float = 0.35) -> np.ndarray:
        tone = np.sin(2 * np.pi * freq * t) * vol
        tone[:fade] *= np.linspace(0, 1, fade)
        tone[-fade:] *= np.linspace(1, 0, fade)
        return tone.astype(np.float32)

    gap = np.zeros(int(sr * 0.05), dtype=np.float32)

    if style == "activate":
        beep = np.concatenate([make_tone(880), gap, make_tone(1100)])
    elif style == "deactivate":
        beep = make_tone(440)
    elif style == "timeout":
        beep = np.concatenate([make_tone(880), gap, make_tone(660)])
    else:
        beep = make_tone(880)

    sd.play(beep, samplerate=sr)
    sd.wait()


def transcribe_audio(audio_data: np.ndarray) -> str:
    if audio_data.size == 0:
        return ""
    result = mlx_whisper.transcribe(
        audio_data,
        path_or_hf_repo=HF_REPO,
        language="en",
        task="transcribe",
        word_timestamps=True,
        verbose=False,
    )
    return result.get("text", "")


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if target_sr == orig_sr or audio.size == 0:
        return audio
    ratio = target_sr / orig_sr
    n_samples = int(round(audio.shape[0] * ratio))
    indices = np.arange(n_samples) / ratio
    left = np.floor(indices).astype(np.int64)
    right = np.minimum(left + 1, audio.shape[0] - 1)
    frac = indices - left
    return ((1 - frac) * audio[left] + frac * audio[right]).astype(audio.dtype)


def speak(text: str):
    sd.stop()
    time.sleep(0.1) 
    sentences = re.split(r'(?<=[.!?])\s+', text)
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        try:
            for _, _, audio in pipeline(sentence, voice='af_heart', speed=1.3):
                samples = np.array(audio[0], dtype=np.float32)
                device_index = sd.default.device[1] if isinstance(sd.default.device, (tuple, list)) else sd.default.device
                device_info = sd.query_devices(device_index, 'output')
                device_rate = int(device_info['default_samplerate'])
                if device_rate != 24000:
                    samples = resample_audio(samples, 24000, device_rate)
                sd.play(samples, samplerate=device_rate)
                sd.wait()
        except Exception as e:
            print(f"[TTS] Failed on sentence: {e}")
            sd.stop()
            time.sleep(0.1) 
            continue

# -------------------
# WAKE WORD HELPERS
# -------------------

def is_wake_word(text: str) -> bool:
    t = text.strip().lower()
    for w in WAKE_WORDS:
        if w in t:
            return True
    # Fuzzy: any standalone "gwen"
    if re.search(r'\bgwen\b', t):
        return True
    return False


def is_deactivation(text: str) -> bool:
    t = text.strip().lower()
    return any(phrase in t for phrase in DEACTIVATION_PHRASES)


def is_exit(text: str) -> bool:
    t = text.strip().lower()
    return any(phrase in t for phrase in EXIT_PHRASES)

# ----------------------------------------
# WAKE WORD LOOP — lightweight, always-on
# ----------------------------------------

def listen_for_wake_word():
    """
    Stays dormant, monitoring audio energy.
    When energy is detected, captures a short clip and checks for wake word.
    Returns only when wake word is confirmed.
    """
    print("[Gwen] Dormant — say 'Hey Gwen' to activate.")

    with sd.InputStream(samplerate=SAMPLE_RATE, dtype="float32", channels=1, blocksize=CHUNK_SIZE) as stream:
        while True:
            data, _ = stream.read(CHUNK_SIZE)
            rms = np.sqrt(np.mean(data**2))

            if rms <= SILENCE_THRESHOLD:
                continue

            # Energy detected — capture wake word clip (max 2.5s)
            frames = [data.copy()]
            silent_chunks = 0
            max_silent = int(0.7 * SAMPLE_RATE / CHUNK_SIZE)
            max_chunks = int(2.5 * SAMPLE_RATE / CHUNK_SIZE)
            total = 1

            while total < max_chunks:
                chunk, _ = stream.read(CHUNK_SIZE)
                frames.append(chunk.copy())
                total += 1
                chunk_rms = np.sqrt(np.mean(chunk**2))
                if chunk_rms < SILENCE_THRESHOLD:
                    silent_chunks += 1
                    if silent_chunks >= max_silent:
                        break
                else:
                    silent_chunks = 0

            audio_clip = np.concatenate(frames).flatten()
            text = transcribe_audio(audio_clip)

            if not text.strip():
                continue

            print(f"[Wake] Heard: {text.strip()}")

            if is_wake_word(text):
                return  # Wake word confirmed — exit loop

# -------------------------------------------
# ACTIVE PROMPT LISTENER — with 30s timeout
# -------------------------------------------

def listen_for_prompt() -> tuple[str | None, float]:
    """
    Listens for the user's prompt after activation.
    Waits for speech energy (like Siri — doesn't immediately start recording).
    Returns (transcription, stt_ms) or (None, 0) on timeout.
    """
    deadline = time.time() + WAKE_TIMEOUT
    print(f"[Gwen] Active — waiting for your prompt ({int(WAKE_TIMEOUT)}s timeout)...")

    with sd.InputStream(samplerate=SAMPLE_RATE, dtype="float32", channels=1, blocksize=CHUNK_SIZE) as stream:
        # Phase 1: wait for speech to begin (energy threshold), respecting timeout
        while time.time() < deadline:
            data, _ = stream.read(CHUNK_SIZE)
            rms = np.sqrt(np.mean(data**2))

            if rms > SILENCE_THRESHOLD:
                # Speech started — capture the full prompt
                frames = [data.copy()]
                silent_chunks = 0
                max_silent = int(SILENCE_DURATION * SAMPLE_RATE / CHUNK_SIZE)

                while True:
                    chunk, _ = stream.read(CHUNK_SIZE)
                    frames.append(chunk.copy())
                    chunk_rms = np.sqrt(np.mean(chunk**2))
                    peak = np.max(np.abs(chunk))
                    is_silent = chunk_rms < SILENCE_THRESHOLD and peak < SILENCE_THRESHOLD * 3
                    if is_silent:
                        silent_chunks += 1
                        if silent_chunks >= max_silent:
                            break
                    else:
                        silent_chunks = 0

                audio = np.concatenate(frames).flatten()
                t_stt = time.time()
                transcription = transcribe_audio(audio)
                stt_ms = (time.time() - t_stt) * 1000
                return transcription, stt_ms

    # Timeout
    return None, 0.0

# -------------------
# SESSION SUMMARIZER
# -------------------

def summarize_and_save_session():
    if not chat_history:
        return
    history_text = "\n".join(
        f"{m['role'].capitalize()}: {m['content'][:200]}"
        for m in chat_history[-20:]
    )
    prompt = f"""Summarize this conversation in 2-3 sentences. Focus on what the user asked, what was accomplished, and any important facts mentioned.

Conversation:
{history_text}

Summary:"""
    try:
        response = generate_lm(
            small_model,
            small_tokenizer,
            prompt=f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
            max_tokens=150,
            sampler=make_sampler(temp=0.3),
            verbose=False
        )
        summary = strip_prefill(strip_think_tags(
            response.text if hasattr(response, "text") else str(response)
        ))
        conn = get_db()
        conn.execute(
            "INSERT INTO session_summaries (summary, created_at) VALUES (?, ?)",
            (summary, datetime.now().isoformat())
        )
        conn.commit()
        conn.close()
        store_semantic_memory(
            f"Session summary ({datetime.now().strftime('%Y-%m-%d')}): {summary}",
            {"type": "session_summary"}
        )
        print(f"[Session] Saved summary: {summary[:80]}...")
    except Exception as e:
        print(f"[Session] Summary failed: {e}")

# ----------
# MAIN LOOP
# ----------

def process_query(transcription: str, stt_ms: float) -> bool | None:
    """
    Process one query.
    Returns True  → stay active (listen for follow-up)
            False → go dormant
            None  → full exit
    """
    global deep_search_mode

    transcription = transcription.strip()
    if not transcription:
        return False
    
    t_start = time.time()
    timings = {"stt_ms": stt_ms}

    now_str = datetime.now().strftime("%H:%M:%S")
    print(f"[{now_str}] You: {transcription}")

    if is_exit(transcription):
        speak("Shutting down. Goodbye.")
        play_beep("deactivate")
        summarize_and_save_session()
        return None

    if is_deactivation(transcription):
        speak("Going to sleep.")
        play_beep("deactivate")
        summarize_and_save_session()
        chat_history.clear()
        return False

    if "toggle deep search" in transcription.lower():
        deep_search_mode = not deep_search_mode
        msg = f"Deep search {'enabled' if deep_search_mode else 'disabled'}."
        print(f"[System] {msg}")
        speak(msg)
        return True

    detect_and_apply_fact_updates(transcription)

    t_mem = time.time()
    facts = get_all_facts()
    global_context = build_global_context()
    semantic_hits = retrieve_semantic_memory(transcription, n_results=3)
    timings["memory_ms"] = (time.time() - t_mem) * 1000

    log_memory_retrieval(transcription, facts, semantic_hits)

    system_prompt = build_system_prompt(global_context, semantic_hits)

    force_search = any(re.search(p, transcription.lower()) for p in FORCE_SEARCH_PATTERNS)
    if force_search:
        print("[Router] Force search triggered.")
        final_response = do_web_search(transcription, transcription)
        final_response = sanitize_response(final_response)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Gwen: {final_response}")

        t_tts = time.time()
        try:
            speak(final_response)
        except Exception as e:
            print(f"[TTS] speak() failed: {e}")
            sd.stop() 
        timings["tts_ms"] = (time.time() - t_tts) * 1000
        timings["total_ms"] = (time.time() - t_start) * 1000 

        print(
            f"[Latency] STT:{timings.get('stt_ms', 0):.0f}ms | Mem:{timings.get('memory_ms', 0):.0f}ms | "
            f"Cls:{timings.get('classifier_ms', 0):.0f}ms | Inf:{timings.get('inference_ms', 0):.0f}ms | "
            f"TTS:{timings.get('tts_ms', 0):.0f}ms | Total:{timings.get('total_ms', 0)/1000:.2f}s"
        )
        log_latency(transcription, "forced_search", timings)
        chat_history.append({"role": "user", "content": transcription})
        chat_history.append({"role": "assistant", "content": final_response})
        return True
    
    t_cls = time.time()
    complexity = classify_complexity(transcription)
    timings["classifier_ms"] = (time.time() - t_cls) * 1000
    print(f"[Router] Complexity: {complexity} ({timings['classifier_ms']:.1f}ms)")

    t_inf = time.time()
    response = get_response(transcription, complexity, system_prompt)
    timings["inference_ms"] = (time.time() - t_inf) * 1000

    tool_call = parse_tool_call(response)
    if tool_call:
        final_response = execute_tool(tool_call["name"], tool_call["arguments"], transcription)
        print(f"[Tool] {tool_call['name']}: {final_response}")
    else:
        final_response = sanitize_response(response)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Gwen: {final_response}")

    t_tts = time.time()
    try:
        speak(final_response)
    except Exception as e:
        print(f"[TTS] speak() failed: {e}")
        sd.stop() 
    timings["tts_ms"] = (time.time() - t_tts) * 1000
    timings["total_ms"] = (time.time() - t_start) * 1000

    print(
        f"[Latency] STT:{timings['stt_ms']:.0f}ms | Mem:{timings['memory_ms']:.0f}ms | "
        f"Cls:{timings['classifier_ms']:.0f}ms | Inf:{timings['inference_ms']:.0f}ms | "
        f"TTS:{timings['tts_ms']:.0f}ms | Total:{timings['total_ms']/1000:.2f}s"
    )

    log_latency(transcription, complexity, timings)
    chat_history.append({"role": "user", "content": transcription})
    chat_history.append({"role": "assistant", "content": final_response})
    return True


def start_chat():
    global deep_search_mode

    print(f"\n[Gwen / Daughter AI — A11] Session: {SESSION_ID}")
    print(f"[Memory] DB: {DB_PATH}")
    print(f"[Memory] ChromaDB: {'enabled' if CHROMA_AVAILABLE else 'disabled'}")
    print(f"[Router] Small model: {'enabled' if SMALL_MODEL_AVAILABLE else 'disabled (using primary)'}\n")

    while True:
        # DORMANT — wait for wake word
        listen_for_wake_word()
        play_beep("activate")
        print("[Gwen] Activated.")

        # ACTIVE — conversational loop, no wake word needed between turns
        while True:
            transcription, stt_ms = listen_for_prompt()

            if transcription is None:
                print("[Gwen] Timeout — going dormant.")
                play_beep("timeout")
                break  # back to dormant

            result = process_query(transcription, stt_ms)

            if result is None:
                return  # full exit

            if not result:
                break  # soft deactivation — back to dormant

            # True → stay active, play ready beep, listen again
            play_beep("activate")


if __name__ == "__main__":
    start_chat()
