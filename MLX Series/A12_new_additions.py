import random
import re
import threading
import time

import numpy as np
import sounddevice as sd

# ----
# TTS
# ----

try:
    from mlx_audio.tts.models.kokoro import KokoroPipeline
    from mlx_audio.tts.utils import load_model

    _voice_model_id = "prince-canuma/Kokoro-82M"
    _voice_model = load_model(_voice_model_id)
    _pipeline = KokoroPipeline(
        lang_code="a", model=_voice_model, repo_id=_voice_model_id
    )
    TTS_AVAILABLE = True
    print("[A12] Kokoro pipeline loaded.")
except Exception as e:
    _pipeline = None
    TTS_AVAILABLE = False
    print(f"[A12] TTS unavailable ({e}). Phrase tests will print only.")


# ------------------------
# Shared audio constants
# ------------------------

SAMPLE_RATE    = 16_000   
TTS_RATE       = 24_000   
CHUNK_SIZE     = 1_024    
SILENCE_THRESH = 0.01     

# ----------------
# Bell-tone beeps
# ----------------

# MERGE REMINDERS
# 1. Copy _bell_tone(), beep_activate(), beep_deactivate(), beep_timeout() above play_beep().
# 2. Replace every call to play_beep("activate")    → beep_activate()
#                          play_beep("deactivate")  → beep_deactivate()
#                          play_beep("timeout")     → beep_timeout()
# 3. Delete the old play_beep() definition.


def _bell_tone(
    frequencies: list[float],
    duration: float = 0.18,
    sample_rate: int = 44_100,
    amplitude: float = 0.25,
) -> np.ndarray:
    
    n = int(sample_rate * duration)
    t = np.linspace(0, duration, n, endpoint=False)

    attack = int(0.002 * sample_rate)
    envelope = np.exp(-t / (duration * 0.45))
    envelope[:attack] = np.linspace(0, 1, attack)

    harmonic_weights = [1.0, 0.35, 0.15, 0.06]
    wave = np.zeros(n)
    for freq in frequencies:
        for h, w in enumerate(harmonic_weights, start=1):
            wave += w * np.sin(2 * np.pi * freq * h * t)

    wave *= envelope
    peak = np.max(np.abs(wave))
    if peak > 0:
        wave = wave / peak * amplitude

    stereo = np.stack([wave, wave], axis=1).astype(np.float32)
    return stereo


def beep_activate(sample_rate: int = 44_100) -> None:
    gap  = np.zeros((int(sample_rate * 0.04), 2), dtype=np.float32)
    clip = np.concatenate([
        _bell_tone([880.0],   duration=0.16, sample_rate=sample_rate),
        gap,
        _bell_tone([1_320.0], duration=0.20, sample_rate=sample_rate),
    ])
    sd.play(clip, samplerate=sample_rate)
    sd.wait()


def beep_deactivate(sample_rate: int = 44_100) -> None:
    sd.play(_bell_tone([523.25], duration=0.22, sample_rate=sample_rate),
            samplerate=sample_rate)
    sd.wait()


def beep_timeout(sample_rate: int = 44_100) -> None:
    gap  = np.zeros((int(sample_rate * 0.05), 2), dtype=np.float32)
    clip = np.concatenate([
        _bell_tone([1_047.0], duration=0.16, sample_rate=sample_rate),
        gap,
        _bell_tone([659.0],   duration=0.20, sample_rate=sample_rate),
    ])
    sd.play(clip, samplerate=sample_rate)
    sd.wait()

# ------------------------------
# Randomised activation phrases
# ------------------------------

# MERGE REMINDERS
# 1. Add ACTIVATION_PHRASES and pick_activation_phrase() to the constants
# 2. In start_chat(), call speak(pick_activation_phrase())


ACTIVATION_PHRASES = [
    "I'm listening.",
    "Hey Kam.",
    "Yeah?",
    "Uh huh.",
    "Go ahead.",
    "How can I help?",
    "What's up?",
    "Tell me.",
]


def pick_activation_phrase() -> str:
    """Call this once per activation instead of a hardcoded phrase."""
    return random.choice(ACTIVATION_PHRASES)


# -------------------------
# Text / voice mode toggle
# -------------------------

# MERGE REMINDER
# 1. ModeController class below the constants section.
# 2. In start_chat(), instantiate once before the outer while-loop:
#        mode_ctrl = ModeController()
# 3. At the top of process_query(), before complexity classification,
#    add:
#        if mode_ctrl.process(transcription):
#            return True          
# 4. Where I read from the mic, gate on mode_ctrl.is_voice:
#        if mode_ctrl.is_voice:
#            transcription, stt_ms = listen_for_prompt()
#        else:
#            transcription = input("You (text): ").strip()
#            stt_ms = 0.0

class ModeController:

    VOICE = "voice"
    TEXT  = "text"

    _TO_TEXT  = {"text mode", "switch to text", "switch to text mode", "type mode"}
    _TO_VOICE = {"voice mode", "default mode", "back to voice", "voice input"}

    def __init__(self, initial: str = VOICE) -> None:
        self.mode = initial
        self._print_mode()

    def _print_mode(self) -> None:
        icon = "🎙" if self.mode == self.VOICE else "⌨️ "
        print(f"  [{icon} MODE: {self.mode.upper()}]")

    def process(self, utterance: str) -> bool:
        lowered = utterance.strip().lower()
        if lowered in self._TO_TEXT and self.mode != self.TEXT:
            self.mode = self.TEXT
            self._print_mode()
            _speak("Switching to text mode.")
            return True
        if lowered in self._TO_VOICE and self.mode != self.VOICE:
            self.mode = self.VOICE
            self._print_mode()
            _speak("Back to voice mode.")
            return True
        return False

    @property
    def is_voice(self) -> bool:
        return self.mode == self.VOICE

# -------------------------
# Interrupt-while-speaking
# -------------------------

# MERGE REMINDER
# 1. Add _interrupt_event, _mic_rms_monitor(), and speak_interruptible() near the existing speak() function.
# 2. In process_query(), replace:
#        speak(final_response)
#    with:
#        interrupted = speak_interruptible(final_response)
#        if interrupted:
#            play_beep("activate")   # or beep_activate() after Component 1 merge
#            # optionally: re-listen immediately without replaying full cycle
# 3. The RMS threshold is shared with SILENCE_THRESHOLD — no new constant needed.
#    Interrupt sensitivity is set by INTERRUPT_RMS_THRESHOLD below; tune if needed.


INTERRUPT_RMS_THRESHOLD: float = SILENCE_THRESH * 3.0
MIC_BLOCK_SIZE: int = 512

_interrupt_event = threading.Event()


def _mic_rms_monitor(stop_event: threading.Event) -> None:
    def _callback(indata, frames, time_info, status):
        if stop_event.is_set():
            raise sd.CallbackStop()
        rms = float(np.sqrt(np.mean(indata ** 2)))
        if rms > INTERRUPT_RMS_THRESHOLD:
            print(f"\n  [INTERRUPT] RMS={rms:.4f} — stopping playback")
            sd.stop()
            _interrupt_event.set()
            raise sd.CallbackStop()

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=MIC_BLOCK_SIZE,
            callback=_callback,
        ):
            while not stop_event.is_set() and not _interrupt_event.is_set():
                time.sleep(0.01)
    except Exception:
        pass


def speak_interruptible(text: str) -> bool:
    _interrupt_event.clear()
    stop_monitor = threading.Event()

    monitor = threading.Thread(
        target=_mic_rms_monitor, args=(stop_monitor,), daemon=True
    )
    monitor.start()

    interrupted = False
    try:
        _speak(text)                    # delegates to local _speak()
    except Exception as e:
        print(f"[TTS] speak_interruptible: {e}")
        sd.stop()
    finally:
        interrupted = _interrupt_event.is_set()
        stop_monitor.set()
        monitor.join(timeout=1.0)
        _interrupt_event.clear()

    return interrupted


# ----------------------------
# _speak() - just for testing
# ----------------------------

# REMINDER : Delete _speak() when merging.

def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr or audio.size == 0:
        return audio
    ratio = target_sr / orig_sr
    n = int(round(audio.shape[0] * ratio))
    idx = np.arange(n) / ratio
    left = np.floor(idx).astype(np.int64)
    right = np.minimum(left + 1, audio.shape[0] - 1)
    frac = idx - left
    return ((1 - frac) * audio[left] + frac * audio[right]).astype(audio.dtype)


def _speak(text: str) -> None:
    if not TTS_AVAILABLE or _pipeline is None:
        print(f"  [TTS stub] {text}")
        time.sleep(0.5)
        return

    sd.stop()
    time.sleep(0.05)

    for sentence in re.split(r"(?<=[.!?])\s+", text):
        sentence = sentence.strip()
        if not sentence:
            continue
        try:
            for _, _, audio in _pipeline(sentence, voice="af_heart", speed=1.3):
                samples = np.array(audio[0], dtype=np.float32)
                dev = sd.default.device
                dev_idx = dev[1] if isinstance(dev, (tuple, list)) else dev
                dev_rate = int(sd.query_devices(dev_idx, "output")["default_samplerate"])
                if dev_rate != TTS_RATE:
                    samples = _resample(samples, TTS_RATE, dev_rate)
                sd.play(samples, samplerate=dev_rate)
                sd.wait()
                if _interrupt_event.is_set():
                    return
        except Exception as e:
            print(f"[TTS] sentence failed: {e}")
            sd.stop()
            time.sleep(0.05)


# ----------------------
# STANDALONE TEST SUITE
# ----------------------

def _separator(title: str) -> None:
    print("\n" + "═" * 58)
    print(f"  {title}")
    print("═" * 58)


def test_beeps() -> None:
    _separator("TEST 1 — Bell-tone beeps")
    for label, fn in [
        ("activate  (ascending ↑↑)", beep_activate),
        ("deactivate (single low ↓)", beep_deactivate),
        ("timeout   (descending ↓↓)", beep_timeout),
    ]:
        print(f"  ▶ {label}")
        fn()
        time.sleep(0.4)
    print("  ✓ Done.\n")


def test_activation_phrases() -> None:
    _separator("TEST 2 — Randomised activation phrases")
    phrases = ACTIVATION_PHRASES[:]
    random.shuffle(phrases)
    for phrase in phrases:
        print(f"  ▶ '{phrase}'")
        beep_activate()
        time.sleep(0.08)
        _speak(phrase)
        time.sleep(0.4)
    print("  ✓ Done.\n")


def test_mode_toggle() -> None:
    _separator("TEST 3 — Text / voice mode toggle  (simulated)")
    mc = ModeController()
    cases = [
        ("text mode",         True,  "text"),
        ("text mode",         False, "text"),   
        ("voice mode",        True,  "voice"),
        ("switch to text",    True,  "text"),
        ("default mode",      True,  "voice"),
        ("what is 2 plus 2",  False, "voice"),
    ]
    for utterance, expect_switch, expect_mode in cases:
        print(f'\n  utterance: "{utterance}"')
        switched = mc.process(utterance)
        assert switched == expect_switch, f"Switch mismatch: got {switched}"
        assert mc.mode  == expect_mode,   f"Mode mismatch: got {mc.mode}"
        if not switched:
            print(f"    → no switch (stays in {mc.mode} mode)")
        time.sleep(0.25)
    print("\n  ✓ Done.\n")


def test_interrupt() -> None:
    _separator("TEST 4 — Interrupt-while-speaking")
    long_text = (
        "This is a long sentence Gwen is speaking right now. "
        "You can interrupt me at any time by making a sound or speaking. "
        "Just say something or clap, and I will stop immediately. "
        "Go ahead, try it — I am waiting for you to cut me off."
    )
    print(f"  RMS interrupt threshold: {INTERRUPT_RMS_THRESHOLD:.4f}")
    print("  ➜ Speak or clap to interrupt Gwen mid-sentence.\n")
    time.sleep(1.0)

    beep_activate()
    interrupted = speak_interruptible(long_text)

    if interrupted:
        print("  ✓ Interrupted — returned to listening state.")
        beep_deactivate()
    else:
        print("  ✓ Playback finished without interruption.")
    print("  ✓ Done.\n")


def main() -> None:
    print("\n" + "█" * 58)
    print("  Daughter AI / Gwen — A12 Component Test Suite")
    print("█" * 58)

    test_beeps()
    input("  Press Enter → Test 2 (activation phrases)…\n")

    test_activation_phrases()
    input("  Press Enter → Test 3 (mode toggle)…\n")

    test_mode_toggle()
    input("  Press Enter → Test 4 (interrupt)…\n")

    test_interrupt()

    print("█" * 58)
    print("  All four tests complete.")
    print("█" * 58 + "\n")


if __name__ == "__main__":
    main()