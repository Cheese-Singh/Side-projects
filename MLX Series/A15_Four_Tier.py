import re
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import ollama
from mlx_lm import load as load_lm, generate as generate_lm
from mlx_vlm import load as load_vlm, generate as generate_vlm

# --------------
# SYSTEM PROMPT
# --------------

SYSTEM_PROMPT = """
You are Gwen, a precise local AI assistant.

Rules:
- Give only the final answer.
- Do not output hidden reasoning, chain-of-thought, analysis, scratchpad, drafts, or <think> blocks.
- Do not explain how you arrived at the answer unless the user explicitly asks.
- Answer in the shortest complete form that satisfies the request.
- If the user asks for one sentence, output exactly one sentence.
- If the user asks for code, output only the necessary code unless explanation is requested.
- Do not invent facts.
- If the answer is uncertain, say so clearly.
- If information is missing, say what is missing.
- Do not claim to have checked something unless it was actually provided.
""".strip()

# --------------
# MODEL CONFIG
# --------------

class ModelTier(Enum):
    VERY_LIGHT = "Very Light"
    LIGHT = "Light"
    REGULAR = "Regular"
    MAX = "MAX"

@dataclass(frozen=True)
class ModelConfig:
    name: str
    backend: str          # "mlx_lm", "mlx_vlm", "ollama"
    family: str           # "qwen", "gemma", "kimi"
    temperature: float = 0.0
    think: bool = False   # only used by Ollama

MODEL_REGISTRY = {
    ModelTier.VERY_LIGHT: ModelConfig(
        name="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        backend="mlx_lm",
        family="qwen",
        temperature=0.0,
    ),

    ModelTier.LIGHT: ModelConfig(
        name="mlx-community/Qwen3.5-9B-MLX-4bit",
        backend="mlx_vlm",
        family="qwen",
        temperature=0.0,
    ),

    ModelTier.REGULAR: ModelConfig(
        name="gemma4:cloud",
        backend="ollama",
        family="gemma",
        temperature=0.1,
        think=False,
    ),

    ModelTier.MAX: ModelConfig(
        name="kimi-k2.7-code:cloud",
        backend="ollama",
        family="kimi",
        temperature=0.1,
        think=False,
    ),
}

# ---------
# LOGGING
# ---------

def timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")

def log_success(message: str) -> None:
    print(f"[SUCCESS {timestamp()}] {message}")

def log_info(message: str) -> None:
    print(f"[INFO {timestamp()}] {message}")

def log_error(message: str) -> None:
    print(f"[ERROR {timestamp()}] {message}")

def log_failed(message: str) -> None:
    print(f"[FAILED {timestamp()}] {message}")

def log_skipped(message: str) -> None:
    print(f"[SKIPPED {timestamp()}] {message}")

# ---------
# CLEANING
# ---------

REASONING_PATTERNS = {
    "qwen": [
        r"<think>.*?</think>",
    ],

    "gemma": [
        r"<think>.*?</think>",
        r"<thinking>.*?</thinking>",
        r"<reasoning>.*?</reasoning>",
    ],

    "kimi": [
        r"<think>.*?</think>",
        r"<thinking>.*?</thinking>",
        r"<reasoning>.*?</reasoning>",
    ],
}

def clean_response(text: str, family: str) -> str:
    patterns = REASONING_PATTERNS.get(family, [])

    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)

    return text.strip()

def extract_mlx_text(response: Any) -> str:
    if hasattr(response, "text"):
        return response.text

    return str(response)

def extract_ollama_text(response: Any) -> str:
    if hasattr(response, "message"):
        message = response.message

        if hasattr(message, "content"):
            return message.content or ""

    if isinstance(response, dict):
        message = response.get("message", {})

        if isinstance(message, dict):
            return message.get("content", "") or ""

    return str(response)

def make_mlx_chat_prompt(user_prompt: str) -> str:
    return (
        f"<|im_start|>system\n"
        f"{SYSTEM_PROMPT}"
        f"<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{user_prompt}"
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"Final answer:"
    )

def make_ollama_messages(user_prompt: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

# --------
# ROUTER
# --------

class ModelRouter:
    def __init__(self):
        self.loaded_models: dict[str, tuple[Any, Any]] = {}
        self.current_tier: Optional[ModelTier] = None

    def load(self, tier: ModelTier) -> None:
        config = MODEL_REGISTRY[tier]

        if config.backend == "ollama":
            self.current_tier = tier
            log_success(f"Ollama model selected: {config.name}")
            return

        if config.name in self.loaded_models:
            self.current_tier = tier
            log_info(f"Using cached model: {config.name}")
            return

        try:
            if config.backend == "mlx_lm":
                model, tokenizer = load_lm(config.name)
                self.loaded_models[config.name] = (model, tokenizer)

            elif config.backend == "mlx_vlm":
                model, processor = load_vlm(config.name)
                self.loaded_models[config.name] = (model, processor)

            else:
                raise ValueError(f"Unknown backend: {config.backend}")

            self.current_tier = tier
            log_success(f"Loaded model: {config.name}")

        except Exception as e:
            log_error(f"Failed to load {config.name}. Exception: {e}")
            raise

    def generate(self, prompt: str, tier: Optional[ModelTier] = None) -> str:
        if tier is not None:
            self.load(tier)

        if self.current_tier is None:
            raise RuntimeError("No model selected. Call load() first or pass tier=...")

        config = MODEL_REGISTRY[self.current_tier]

        if config.backend == "mlx_lm":
            return self._generate_mlx_lm(prompt, config)

        if config.backend == "mlx_vlm":
            return self._generate_mlx_vlm(prompt, config)

        if config.backend == "ollama":
            return self._generate_ollama_chat(prompt, config)

        raise ValueError(f"Unknown backend: {config.backend}")

    def _generate_mlx_lm(self, prompt: str, config: ModelConfig) -> str:
        model, tokenizer = self.loaded_models[config.name]

        response = generate_lm(
            model=model,
            tokenizer=tokenizer,
            prompt=make_mlx_chat_prompt(prompt),
            verbose=False,
        )

        raw = extract_mlx_text(response)
        return clean_response(raw, config.family)

    def _generate_mlx_vlm(self, prompt: str, config: ModelConfig) -> str:
        model, processor = self.loaded_models[config.name]

        response = generate_vlm(
            model=model,
            processor=processor,
            prompt=make_mlx_chat_prompt(prompt),
            verbose=False,
            temperature=config.temperature,
        )

        raw = extract_mlx_text(response)
        return clean_response(raw, config.family)

    def _generate_ollama_chat(self, prompt: str, config: ModelConfig) -> str:
        try:
            response = ollama.chat(
                model=config.name,
                messages=make_ollama_messages(prompt),
                think=config.think,
                options={
                    "temperature": config.temperature,
                },
            )

            raw = extract_ollama_text(response)
            return clean_response(raw, config.family)

        except ollama.ResponseError as e:
            if e.status_code == 401:
                raise RuntimeError("Ollama auth failed. Run: ollama signin") from e

            if e.status_code == 429:
                raise RuntimeError("Ollama cloud limit hit. Use a local fallback or try later.") from e

            raise RuntimeError(f"Ollama error {e.status_code}: {e}") from e

# --------
# TESTING
# --------

def test_all_except_max(
    prompt: str = "Reply with exactly one short sentence.",
) -> None:
    router = ModelRouter()

    for tier in ModelTier:
        if tier == ModelTier.MAX:
            log_skipped(tier.value)
            continue

        print("\n" + "=" * 60)
        print(f"Testing: {tier.value}")
        print("=" * 60)

        start = time.perf_counter()

        try:
            response = router.generate(prompt, tier=tier)
            elapsed = time.perf_counter() - start

            print(response)
            log_success(f"{tier.value} completed in {elapsed:.2f}s")

        except Exception as e:
            elapsed = time.perf_counter() - start
            log_failed(f"{tier.value} failed after {elapsed:.2f}s: {e}")

# ----------
# DEBUGGING
# ----------

def debug_ollama_raw(
    model_name: str,
    prompt: str = "Think briefly and answer: 2 + 2?",
) -> None:
    response = ollama.chat(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        think=True,
    )

    print("\nFULL RESPONSE:")
    print(response)

    if hasattr(response, "message"):
        print("\nMESSAGE CONTENT:")
        print(getattr(response.message, "content", None))

        print("\nMESSAGE THINKING:")
        print(getattr(response.message, "thinking", None))

# ------------
# ENTRY POINT
# ------------

if __name__ == "__main__":
    test_all_except_max(
        prompt="Explain what a vector database is in one sentence."
    )