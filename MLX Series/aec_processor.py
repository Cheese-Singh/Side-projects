from __future__ import annotations

import threading
from collections import deque
from typing import Optional

import numpy as np

try:
    from aec_audio_processing import AudioProcessor
    AEC_AVAILABLE = True
except Exception as e:
    AudioProcessor = None
    AEC_AVAILABLE = False
    print(f"[AEC] aec_audio_processing unavailable ({e}). "
          f"Falling back to raw mic input, no echo cancellation.")


AEC_SAMPLE_RATE = 16000
FRAME_SAMPLES = 160
FRAME_BYTES = FRAME_SAMPLES * 2  


def _float32_to_pcm16_bytes(samples: np.ndarray) -> bytes:
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


def _pcm16_bytes_to_float32(data: bytes) -> np.ndarray:
    ints = np.frombuffer(data, dtype=np.int16)
    return (ints.astype(np.float32) / 32768.0)


def _resample_int(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr or audio.size == 0:
        return audio
    ratio = target_sr / orig_sr
    n = int(round(audio.shape[0] * ratio))
    if n <= 0:
        return np.array([], dtype=audio.dtype)
    idx = np.arange(n) / ratio
    left = np.floor(idx).astype(np.int64)
    right = np.minimum(left + 1, audio.shape[0] - 1)
    frac = idx - left
    return ((1 - frac) * audio[left] + frac * audio[right]).astype(audio.dtype)


class EchoCanceller:

    def __init__(self, stream_delay_ms: int = 50):
        self._lock = threading.Lock()
        self._enabled = AEC_AVAILABLE
        self._stream_delay_ms = stream_delay_ms
        self._ap: Optional["AudioProcessor"] = None
        self._ref_carry = np.array([], dtype=np.float32)
        self._mic_carry = np.array([], dtype=np.float32)
        self._mic_out_carry = deque()

        if self._enabled:
            try:
                self._ap = AudioProcessor(
                    enable_aec=True,
                    enable_ns=True,
                    enable_agc=False,
                    enable_vad=False,
                )
                self._ap.set_stream_format(AEC_SAMPLE_RATE, 1)
                self._ap.set_reverse_stream_format(AEC_SAMPLE_RATE, 1)
                self._ap.set_stream_delay(self._stream_delay_ms)
            except Exception as e:
                print(f"[AEC] Failed to initialize AudioProcessor ({e}). Disabling AEC.")
                self._enabled = False
                self._ap = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def start_reference_stream(self) -> None:
        with self._lock:
            self._ref_carry = np.array([], dtype=np.float32)

    def stop_reference_stream(self) -> None:
        with self._lock:
            self._ref_carry = np.array([], dtype=np.float32)

    def push_reference(self, samples: np.ndarray, source_sr: int) -> None:
        if not self._enabled or self._ap is None:
            return
        if samples is None or samples.size == 0:
            return

        mono = samples if samples.ndim == 1 else samples.mean(axis=-1)
        resampled = _resample_int(mono.astype(np.float32), source_sr, AEC_SAMPLE_RATE)

        with self._lock:
            combined = np.concatenate([self._ref_carry, resampled])
            n_frames = combined.shape[0] // FRAME_SAMPLES
            for i in range(n_frames):
                frame = combined[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES]
                try:
                    self._ap.process_reverse_stream(_float32_to_pcm16_bytes(frame))
                except Exception as e:
                    print(f"[AEC] process_reverse_stream failed: {e}")
                    self._enabled = False
                    return
            self._ref_carry = combined[n_frames * FRAME_SAMPLES:]

    def process_mic_frame(self, samples: np.ndarray) -> np.ndarray:
        if not self._enabled or self._ap is None or samples is None or samples.size == 0:
            return samples

        mono = samples if samples.ndim == 1 else samples.mean(axis=-1)
        original_len = mono.shape[0]

        with self._lock:
            combined = np.concatenate([self._mic_carry, mono.astype(np.float32)])
            n_frames = combined.shape[0] // FRAME_SAMPLES
            processed_chunks = []
            for i in range(n_frames):
                frame = combined[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES]
                try:
                    out_bytes = self._ap.process_stream(_float32_to_pcm16_bytes(frame))
                    processed_chunks.append(_pcm16_bytes_to_float32(out_bytes))
                except Exception as e:
                    print(f"[AEC] process_stream failed: {e}")
                    self._enabled = False
                    return samples
            self._mic_carry = combined[n_frames * FRAME_SAMPLES:]

        if not processed_chunks:
            return np.array([], dtype=np.float32)

        out = np.concatenate(processed_chunks)
        if out.shape[0] > original_len:
            out = out[:original_len]
        elif out.shape[0] < original_len:
            out = np.pad(out, (0, original_len - out.shape[0]))
        return out