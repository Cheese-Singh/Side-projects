from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np


def _coerce_device_index(value: Any) -> Optional[int]:
    if value is None or value == -1:
        return None
    if isinstance(value, (tuple, list)):
        for item in value:
            if isinstance(item, (int, np.integer)):
                return int(item)
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return None


def _device_supports_kind(device_info: Any, kind: str) -> bool:
    if not isinstance(device_info, dict):
        return False
    if kind == "output":
        return int(device_info.get("max_output_channels", 0) or 0) > 0
    if kind == "input":
        return int(device_info.get("max_input_channels", 0) or 0) > 0
    return False


def select_audio_device(sounddev: Any, kind: str = "output") -> Optional[int]:
    """Pick a usable output/input device index with a safe fallback."""
    try:
        default_device = getattr(getattr(sounddev, "default", None), "device", None)
    except Exception:
        default_device = None

    candidates: list[int] = []
    default_index = _coerce_device_index(default_device)
    if default_index is not None:
        candidates.append(default_index)

    if isinstance(default_device, (tuple, list)):
        for item in default_device:
            device_index = _coerce_device_index(item)
            if device_index is not None and device_index not in candidates:
                candidates.append(device_index)

    for idx in candidates:
        try:
            info = sounddev.query_devices(idx, kind)
        except Exception:
            continue
        if _device_supports_kind(info, kind):
            return int(idx)

    try:
        devices = sounddev.query_devices()
    except Exception:
        return None

    if isinstance(devices, (list, tuple)):
        for idx, info in enumerate(devices):
            if _device_supports_kind(info, kind):
                return int(idx)

    return None


def get_device_default_rate(sounddev: Any, device_index: Optional[int], fallback: int) -> int:
    if device_index is None:
        return fallback
    try:
        info = sounddev.query_devices(device_index, "output")
        rate = info.get("default_samplerate")
        if rate:
            return int(round(float(rate)))
    except Exception:
        pass
    return fallback

_DEVICE_SETTLE_SECONDS = 0.03

_last_stop_time: float = 0.0


def _settle_after_stop() -> None:
    global _last_stop_time
    elapsed = time.monotonic() - _last_stop_time
    if elapsed < _DEVICE_SETTLE_SECONDS:
        time.sleep(_DEVICE_SETTLE_SECONDS - elapsed)


def _mark_stopped() -> None:
    global _last_stop_time
    _last_stop_time = time.monotonic()


def play_audio(
    sounddev: Any,
    audio: np.ndarray,
    sample_rate: int,
    device_index: Optional[int] = None,
    blocking: bool = True,
) -> bool:
    if audio is None or np.size(audio) == 0:
        return False

    audio = np.asarray(audio, dtype=np.float32)

    candidate_devices = []
    if device_index is not None:
        candidate_devices.append(device_index)
    selected_device = select_audio_device(sounddev, kind="output")
    if selected_device is not None and selected_device not in candidate_devices:
        candidate_devices.append(selected_device)
    candidate_devices.append(None)

    last_error: Optional[Exception] = None

    for dev_idx in candidate_devices:
        device_default = get_device_default_rate(sounddev, dev_idx, fallback=int(sample_rate))
        candidate_rates = []
        for r in (device_default, int(sample_rate), 48000, 44100, 24000, 16000):
            if r not in candidate_rates:
                candidate_rates.append(r)

        for rate in candidate_rates:
            try:
                try:
                    sounddev.stop()
                except Exception:
                    pass
                _mark_stopped()
                _settle_after_stop()

                if dev_idx is None:
                    sounddev.play(audio, samplerate=rate)
                else:
                    sounddev.play(audio, samplerate=rate, device=dev_idx)
                if blocking:
                    sounddev.wait()
                return True
            except Exception as exc:
                last_error = exc
                try:
                    sounddev.stop()
                except Exception:
                    pass
                _mark_stopped()

    if last_error is not None:
        raise last_error
    return False