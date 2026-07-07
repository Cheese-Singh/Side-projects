from __future__ import annotations

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


def play_audio(sounddev: Any, audio: np.ndarray, sample_rate: int, device_index: Optional[int] = None, blocking: bool = True) -> bool:
    """Play audio with fallback device and sample rate selection."""
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

    candidate_rates = [int(sample_rate)]
    if sample_rate != 48000:
        candidate_rates.append(48000)
    if sample_rate != 44100:
        candidate_rates.append(44100)
    if sample_rate != 24000:
        candidate_rates.append(24000)
    if sample_rate != 16000:
        candidate_rates.append(16000)

    last_error: Optional[Exception] = None
    
    for dev_idx in candidate_devices:
        for rate in candidate_rates:
            try:
                try:
                    sounddev.stop()
                except Exception:
                    pass
                
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

    if last_error is not None:
        raise last_error
    return False
